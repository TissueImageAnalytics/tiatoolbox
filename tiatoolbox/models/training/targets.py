"""Annotation-to-target builders for training datasets."""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import cv2
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

if TYPE_CHECKING:  # pragma: no cover
    from shapely.geometry.base import BaseGeometry

    from tiatoolbox.annotation import Annotation, AnnotationStore
    from tiatoolbox.type_hints import Predicate

_SPATIAL_DIMS = 2

Bounds = tuple[float, float, float, float]
TargetType = np.ndarray | int | float | dict[str, "TargetType"]
DEFAULT_GEOMETRY_PREDICATE = "intersects"
DEFAULT_DISTANCE = 0.0


class AnnotationQuery(TypedDict, total=False):
    """Store query options accepted by every target builder.

    ``where`` filters annotation properties, ``geometry_predicate`` relates
    annotations to the patch box, ``min_area`` drops small annotations, and
    ``distance`` expands the queried region. See
    :meth:`tiatoolbox.annotation.AnnotationStore.query`.
    """

    where: Predicate | None
    geometry_predicate: str
    min_area: float | None
    distance: float


def _positive_int(value: int, name: str) -> int:
    """Validate a positive integer drawing parameter."""
    value = int(value)
    if value <= 0:
        msg = f"`{name}` must be a positive integer."
        raise ValueError(msg)
    return value


def _as_bounds(bounds: tuple[float, ...] | list[float] | np.ndarray) -> Bounds:
    """Validate and normalize a patch bounds tuple."""
    if len(bounds) != 4:  # noqa: PLR2004
        msg = "`patch_bounds` must be of length 4: (x_min, y_min, x_max, y_max)."
        raise ValueError(msg)

    x_min, y_min, x_max, y_max = [float(value) for value in bounds]
    if x_max <= x_min or y_max <= y_min:
        msg = "Invalid `patch_bounds`, expected x_max > x_min and y_max > y_min."
        raise ValueError(msg)

    return x_min, y_min, x_max, y_max


def _normalize_output_shape(
    output_shape: tuple[int, int] | list[int] | np.ndarray,
) -> tuple[int, int]:
    """Validate and normalize output shape to (height, width)."""
    if len(output_shape) != 2:  # noqa: PLR2004
        msg = "`output_shape` must be of length 2: (height, width)."
        raise ValueError(msg)

    height, width = [int(value) for value in output_shape]
    if height <= 0 or width <= 0:
        msg = "`output_shape` dimensions must be positive integers."
        raise ValueError(msg)

    return height, width


def _normalize_min_fraction(min_fraction: float) -> float:
    """Validate and normalize a fractional threshold in the interval [0, 1]."""
    value = float(min_fraction)
    if not 0.0 <= value <= 1.0:
        msg = "`min_fraction` must be in the interval [0, 1]."
        raise ValueError(msg)
    return value


def _normalize_overlap_method(
    overlap_method: str,
) -> Literal["exact", "rasterized"]:
    """Validate overlap method values used by coverage-based builders."""
    if overlap_method not in {"exact", "rasterized"}:
        msg = "`overlap_method` must be either `'exact'` or `'rasterized'`."
        raise ValueError(msg)
    return overlap_method


def _patch_box(bounds: Bounds) -> Polygon:
    """Create a shapely box from bounds."""
    return box(*bounds)


def _point_within_bounds(point: tuple[float, float], bounds: Bounds) -> bool:
    """Return whether an XY point lies within half-open patch bounds."""
    x_coord, y_coord = point
    x_min, y_min, x_max, y_max = bounds
    return x_min <= x_coord < x_max and y_min <= y_coord < y_max


def _point_to_pixel_coordinates(
    point: tuple[float, float],
    bounds: Bounds,
    output_shape: tuple[int, int],
) -> tuple[float, float] | None:
    """Map a point from patch coordinates into output pixel coordinates."""
    if not _point_within_bounds(point, bounds):
        return None

    x_min, y_min, x_max, y_max = bounds
    height, width = output_shape
    x_coord, y_coord = point

    scale_x = width / (x_max - x_min)
    scale_y = height / (y_max - y_min)
    pixel_x = (x_coord - x_min) * scale_x
    pixel_y = (y_coord - y_min) * scale_y
    # Scalar clamp; same result as ``np.clip`` without the array round trip.
    pixel_x = min(max(float(pixel_x), 0.0), float(max(width - 1, 0)))
    pixel_y = min(max(float(pixel_y), 0.0), float(max(height - 1, 0)))
    return pixel_x, pixel_y


def _annotation_centroids_in_patch(
    annotations: dict[str, Annotation],
    patch_bounds: Bounds,
    output_shape: tuple[int, int],
) -> list[tuple[float, float]]:
    """Return annotation centroids that fall inside the patch in pixel space."""
    points: list[tuple[float, float]] = []
    for annotation in annotations.values():
        centroid = annotation.geometry.centroid
        pixel_point = _point_to_pixel_coordinates(
            (float(centroid.x), float(centroid.y)),
            patch_bounds,
            output_shape,
        )
        if pixel_point is not None:
            points.append(pixel_point)
    return points


def _rescale_geometry(
    geometry: BaseGeometry,
    bounds: Bounds,
    output_shape: tuple[int, int],
) -> BaseGeometry:
    """Map a geometry from patch coordinates into output pixel space.

    Translates by ``(-x_min, -y_min)`` and scales about the origin in a single
    pass over the coordinate array, preserving geometry type and holes. The
    per-coordinate arithmetic matches ``shapely.affinity.translate`` followed
    by ``shapely.affinity.scale`` exactly.
    """
    x_min, y_min, x_max, y_max = bounds
    height, width = output_shape
    offset = np.array([x_min, y_min])
    factor = np.array([width / (x_max - x_min), height / (y_max - y_min)])
    return shapely.transform(geometry, lambda coords: (coords - offset) * factor)


def _drop_zero_area_parts(geometry: BaseGeometry) -> BaseGeometry | None:
    """Drop degenerate (zero-area) polygon parts the clipping path would discard."""
    if geometry.geom_type == "Polygon":
        return None if geometry.area == 0 else geometry
    if geometry.geom_type == "MultiPolygon":
        parts = [part for part in geometry.geoms if part.area > 0]
        if not parts:
            return None
        if len(parts) != len(geometry.geoms):
            return MultiPolygon(parts)
    return geometry


def _clip_and_rescale_geometry(
    geometry: BaseGeometry,
    bounds: Bounds,
    output_shape: tuple[int, int],
) -> BaseGeometry | None:
    """Clip geometry to patch bounds and map into output pixel space.

    Geometries whose bounds already lie within the patch are mapped directly,
    skipping the comparatively expensive shapely intersection. Geometries that
    cross the patch edge are clipped first; polygon parts that collapse to
    lines or points on the crop boundary are dropped.
    """
    x_min, y_min, x_max, y_max = bounds
    geom_x_min, geom_y_min, geom_x_max, geom_y_max = geometry.bounds
    if (
        x_min <= geom_x_min
        and y_min <= geom_y_min
        and geom_x_max <= x_max
        and geom_y_max <= y_max
    ):
        # Fully inside the patch: clipping would return the geometry unchanged.
        # (Empty geometries have NaN bounds and take the clipping path below.)
        contained = _drop_zero_area_parts(geometry)
        if contained is None:
            return None
        return _rescale_geometry(contained, bounds, output_shape)

    clipped = geometry.intersection(_patch_box(bounds))
    if geometry.geom_type in {"Polygon", "MultiPolygon"} and clipped.geom_type not in {
        "Polygon",
        "MultiPolygon",
    }:
        # Touching the crop boundary can collapse polygon parts to lines/points.
        # Keep only area-bearing parts, including in mixed intersection results.
        clipped = unary_union(
            [
                part
                for part in getattr(clipped, "geoms", (clipped,))
                if part.geom_type == "Polygon"
            ]
        )
    if clipped.is_empty:
        return None

    return _rescale_geometry(clipped, bounds, output_shape)


def _coords_to_int(coords: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert XY coordinates to clipped int32 coordinates for OpenCV."""
    rounded = np.rint(coords)
    np.clip(rounded, 0, [width - 1, height - 1], out=rounded)
    return rounded.astype(np.int32)


def _rasterize_geometry(
    mask: np.ndarray,
    geometry: BaseGeometry,
    label: float,
    *,
    line_width: int,
    point_radius: int,
) -> None:
    """Rasterize a single shapely geometry onto a label mask."""
    height, width = mask.shape
    geom_type = geometry.geom_type

    if geom_type == "Polygon":
        if not geometry.interiors:
            # Without holes the polygon's coordinates are exactly its exterior.
            exterior = _coords_to_int(shapely.get_coordinates(geometry), width, height)
            cv2.fillPoly(mask, [exterior], label)
            return
        exterior = _coords_to_int(np.asarray(geometry.exterior.coords), width, height)
        # Holes are outside this annotation, not background annotations. Preserve
        # any previously rasterized geometry within them.
        support = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(support, [exterior], 1)
        for interior in geometry.interiors:
            interior_coords = _coords_to_int(
                np.asarray(interior.coords),
                width,
                height,
            )
            cv2.fillPoly(support, [interior_coords], 0)
        mask[support != 0] = label
        return

    if geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geometry.geoms:
            _rasterize_geometry(
                mask,
                part,
                label,
                line_width=line_width,
                point_radius=point_radius,
            )
        return

    if geom_type == "LineString":
        points = _coords_to_int(np.asarray(geometry.coords), width, height)
        cv2.polylines(
            mask,
            [points],
            isClosed=False,
            color=label,
            thickness=line_width,
        )
        return

    if geom_type == "MultiLineString":
        for part in geometry.geoms:
            _rasterize_geometry(
                mask,
                part,
                label,
                line_width=line_width,
                point_radius=point_radius,
            )
        return

    if geom_type == "Point":
        point_coords = _coords_to_int(np.asarray(geometry.coords), width, height)[0]
        cv2.circle(mask, tuple(point_coords), point_radius, label, thickness=-1)
        return

    if geom_type == "MultiPoint":
        for part in geometry.geoms:
            _rasterize_geometry(
                mask,
                part,
                label,
                line_width=line_width,
                point_radius=point_radius,
            )


class TargetBuilderABC(ABC):
    """Base interface for building patch targets from annotation stores.

    Annotation geometries and ``patch_bounds`` are expected to use the same
    slide/baseline coordinate space. Dense targets returned by builders are in
    patch pixel coordinates: NumPy arrays that start with ``output_shape``, which
    lets augmentation find them. Every builder accepts the
    :class:`AnnotationQuery` options ``where``, ``geometry_predicate``,
    ``min_area`` and ``distance``.
    """

    def __init__(
        self: TargetBuilderABC,
        *,
        where: Predicate | None = None,
        geometry_predicate: str = DEFAULT_GEOMETRY_PREDICATE,
        min_area: float | None = None,
        distance: float = DEFAULT_DISTANCE,
    ) -> None:
        """Initialize :class:`TargetBuilderABC`."""
        self.where = where
        self.geometry_predicate = geometry_predicate
        self.min_area = None if min_area is None else float(min_area)
        self.distance = float(distance)

    def query_annotations(
        self: TargetBuilderABC,
        store: AnnotationStore,
        patch_bounds: Bounds,
    ) -> dict[str, Annotation]:
        """Query relevant annotations from a store for one patch bounds."""
        return store.query(
            geometry=patch_bounds,
            where=self.where,
            geometry_predicate=self.geometry_predicate,
            min_area=self.min_area,
            distance=self.distance,
        )

    def create_target(
        self: TargetBuilderABC,
        *,
        store: AnnotationStore,
        patch_bounds: tuple[float, ...] | list[float] | np.ndarray,
        output_shape: tuple[int, int] | list[int] | np.ndarray,
    ) -> TargetType:
        """Query annotations and build a target for one patch.

        ``patch_bounds`` are interpreted in annotation slide/baseline
        coordinates. Dense target arrays are rescaled to ``output_shape`` in
        patch pixel coordinates.
        """
        normalized_bounds = _as_bounds(patch_bounds)
        normalized_shape = _normalize_output_shape(output_shape)
        annotations = self.query_annotations(store, normalized_bounds)
        return self.build_target(annotations, normalized_bounds, normalized_shape)

    @abstractmethod
    def build_target(
        self: TargetBuilderABC,
        annotations: dict[str, Annotation],
        patch_bounds: Bounds,
        output_shape: tuple[int, int],
    ) -> TargetType:
        """Build target array/scalar from queried annotations."""


def _has_query_settings(builder: TargetBuilderABC) -> bool:
    """Return whether a builder has non-default annotation query settings."""
    return (
        builder.where is not None
        or builder.geometry_predicate != DEFAULT_GEOMETRY_PREDICATE
        or builder.min_area is not None
        or builder.distance != DEFAULT_DISTANCE
    )


def _queried_descendants(builders: dict[str, TargetBuilderABC]) -> list[str]:
    """Name the builders, at any depth, that have their own query settings."""
    names = []
    for name, builder in builders.items():
        if _has_query_settings(builder):
            names.append(name)
        nested = getattr(builder, "builders", {})
        names.extend(f"{name}.{child}" for child in _queried_descendants(nested))
    return names


class _CombinedTargetBuilder(TargetBuilderABC):
    """Combine the targets of named child builders.

    With query settings of its own, a combiner runs one store query for all its
    children, which may then have no query settings, at any depth. Without them,
    each child runs its own query.
    """

    def __init__(
        self: _CombinedTargetBuilder,
        builders: dict[str, TargetBuilderABC],
        **query: Unpack[AnnotationQuery],
    ) -> None:
        """Initialize a combination of child builders."""
        super().__init__(**query)
        if not builders:
            msg = "`builders` must contain at least one target builder."
            raise ValueError(msg)
        self.builders = builders
        self._queried_children = _queried_descendants(builders)
        if self._queried_children and _has_query_settings(self):
            msg = (
                f"{type(self).__name__} has query settings of its own, so all its "
                "children share one query and cannot have their own. Move the "
                "settings to the parent, or remove the parent's so that each child "
                f"queries for itself. Conflicting children: "
                f"{', '.join(self._queried_children)}."
            )
            raise ValueError(msg)

    @staticmethod
    def _combine(
        targets: dict[str, TargetType],
        output_shape: tuple[int, int],  # noqa: ARG004 - hook signature
    ) -> TargetType:
        """Combine the children's targets into this builder's target."""
        return targets

    def create_target(
        self: _CombinedTargetBuilder,
        *,
        store: AnnotationStore,
        patch_bounds: tuple[float, ...] | list[float] | np.ndarray,
        output_shape: tuple[int, int] | list[int] | np.ndarray,
    ) -> TargetType:
        """Build the target from one shared query, or from one query per child."""
        if _has_query_settings(self):
            return super().create_target(
                store=store, patch_bounds=patch_bounds, output_shape=output_shape
            )
        normalized_bounds = _as_bounds(patch_bounds)
        normalized_shape = _normalize_output_shape(output_shape)
        return self._combine(
            {
                name: builder.create_target(
                    store=store,
                    patch_bounds=normalized_bounds,
                    output_shape=normalized_shape,
                )
                for name, builder in self.builders.items()
            },
            normalized_shape,
        )

    def build_target(
        self: _CombinedTargetBuilder,
        annotations: dict[str, Annotation],
        patch_bounds: Bounds,
        output_shape: tuple[int, int],
    ) -> TargetType:
        """Build the target from annotations queried once for all children."""
        if self._queried_children:
            msg = (
                f"{type(self).__name__} cannot build a target from one set of "
                "annotations, because these children query for themselves: "
                f"{', '.join(self._queried_children)}. Use `create_target`."
            )
            raise ValueError(msg)
        return self._combine(
            {
                name: builder.build_target(annotations, patch_bounds, output_shape)
                for name, builder in self.builders.items()
            },
            output_shape,
        )


class CompositeTargetBuilder(_CombinedTargetBuilder):
    """Combine multiple target builders into a nested target dictionary.

    Without query settings on the composite, every child builder issues its
    own annotation query. When ``where``, ``geometry_predicate``, ``min_area``
    or ``distance`` is set on the composite, a single shared query is issued
    and its result is handed to every child, which then may not have query
    settings of its own.
    """


class StackedTargetBuilder(_CombinedTargetBuilder):
    """Stack multiple spatial target maps into a single multi-channel array.

    Children query as for :class:`CompositeTargetBuilder`: one shared query when
    the stack has query settings, which children then may not have, and
    otherwise one query per child.
    """

    def __init__(
        self: StackedTargetBuilder,
        builders: dict[str, TargetBuilderABC],
        *,
        dtype: np.dtype = np.float32,
        **query: Unpack[AnnotationQuery],
    ) -> None:
        """Initialize :class:`StackedTargetBuilder`."""
        super().__init__(builders, **query)
        self.dtype = np.dtype(dtype)

    def _combine(
        self: StackedTargetBuilder,
        targets: dict[str, TargetType],
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """Validate and stack named 2D channel targets, with channels last."""
        stacked_targets: list[np.ndarray] = []
        for name, child_target in targets.items():
            if (
                not isinstance(child_target, np.ndarray)
                or child_target.ndim != _SPATIAL_DIMS
            ):
                msg = (
                    "StackedTargetBuilder expects each child builder to return a "
                    f"2D NumPy array, but `{name}` returned "
                    f"`{type(child_target).__name__}`."
                )
                raise ValueError(msg)
            if child_target.shape != output_shape:
                msg = (
                    f"Stacked target `{name}` returned shape `{child_target.shape}`, "
                    f"expected `{output_shape}`."
                )
                raise ValueError(msg)
            stacked_targets.append(child_target.astype(self.dtype, copy=False))

        return np.stack(stacked_targets, axis=-1)


def _paint_order(annotations: dict[str, Annotation]) -> list[Annotation]:
    """Order annotations largest first so smaller, nested ones are painted on top.

    Store query order is not insertion order (SQLite follows its spatial index),
    so overlaps need an explicit rule to give the same mask for every store
    type. Ties, including zero-area points and lines, are broken by key.
    """
    return [
        annotations[key]
        for key in sorted(
            annotations, key=lambda key: (-annotations[key].geometry.area, key)
        )
    ]


class MaskTargetBuilder(TargetBuilderABC):
    """Build integer semantic or instance masks from annotations.

    Where annotations overlap, larger annotations are painted first, so
    smaller (typically nested) annotations take precedence; ties are broken by
    annotation key. The result does not depend on the store's query order.

    By default each annotation is painted with its class label. With
    ``instance_ids=True`` each painted annotation gets its own ID (1, 2, ... in
    paint order) and ``class_mapping`` only selects the annotations to include;
    derive distance or horizontal/vertical maps with ``target_transform``.
    """

    def __init__(
        self: MaskTargetBuilder,
        *,
        class_mapping: dict[object, int] | None = None,
        class_property: str = "class",
        default_label: int = 0,
        instance_ids: bool = False,
        line_width: int = 1,
        point_radius: int = 1,
        **query: Unpack[AnnotationQuery],
    ) -> None:
        """Initialize :class:`MaskTargetBuilder`."""
        super().__init__(**query)
        self.class_mapping = class_mapping
        self.class_property = class_property
        self.default_label = int(default_label)
        self.instance_ids = instance_ids
        self.line_width = _positive_int(line_width, "line_width")
        self.point_radius = _positive_int(point_radius, "point_radius")

    def _label_for_annotation(
        self: MaskTargetBuilder,
        annotation: Annotation,
    ) -> int | None:
        """Resolve annotation class label to an integer output id."""
        if self.class_mapping is None:
            return 1

        class_value = annotation.properties.get(self.class_property)
        if class_value not in self.class_mapping:
            return None
        return int(self.class_mapping[class_value])

    def build_target(
        self: MaskTargetBuilder,
        annotations: dict[str, Annotation],
        patch_bounds: Bounds,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """Build an integer mask from annotations."""
        height, width = output_shape
        mask = np.full((height, width), self.default_label, dtype=np.int32)
        instance_ids = itertools.count(1)

        for annotation in _paint_order(annotations):
            label = self._label_for_annotation(annotation)
            if label is None:
                continue

            geometry = _clip_and_rescale_geometry(
                annotation.geometry,
                patch_bounds,
                output_shape,
            )
            if geometry is None:
                continue

            _rasterize_geometry(
                mask,
                geometry,
                # itertools.count is infinite, so this cannot raise StopIteration.
                next(instance_ids) if self.instance_ids else label,  # skipcq: PTC-W0063
                line_width=self.line_width,
                point_radius=self.point_radius,
            )

        return mask.astype(np.int64)


class BoundaryTargetBuilder(TargetBuilderABC):
    """Rasterize original annotation contours, excluding artificial crop edges."""

    def __init__(
        self: BoundaryTargetBuilder,
        *,
        positive_value: float = 1.0,
        background_value: float = 0.0,
        line_width: int = 1,
        point_radius: int = 1,
        **query: Unpack[AnnotationQuery],
    ) -> None:
        """Initialize :class:`BoundaryTargetBuilder`."""
        super().__init__(**query)
        self.positive_value = float(positive_value)
        self.background_value = float(background_value)
        self.line_width = _positive_int(line_width, "line_width")
        self.point_radius = _positive_int(point_radius, "point_radius")

    def build_target(
        self: BoundaryTargetBuilder,
        annotations: dict[str, Annotation],
        patch_bounds: Bounds,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """Build a dense float32 boundary target map."""
        height, width = output_shape
        target = np.full((height, width), self.background_value, dtype=np.float32)

        for annotation in annotations.values():
            geometry = _clip_and_rescale_geometry(
                annotation.geometry.boundary,
                patch_bounds,
                output_shape,
            )
            if geometry is None:
                continue

            _rasterize_geometry(
                target,
                geometry,
                self.positive_value,
                line_width=self.line_width,
                point_radius=self.point_radius,
            )

        return target


class _CoverageTargetBuilder(TargetBuilderABC):
    """Share coverage thresholds between scalar and vector coverage targets.

    A zero ``min_fraction`` still requires positive coverage. Exact coverage
    measures area; rasterized coverage also counts point and line annotations.
    """

    def __init__(
        self: _CoverageTargetBuilder,
        *,
        class_mapping: dict[object, int] | None = None,
        class_property: str = "class",
        min_fraction: float = 0.0,
        overlap_method: Literal["exact", "rasterized"] = "exact",
        **query: Unpack[AnnotationQuery],
    ) -> None:
        """Store validated coverage options."""
        super().__init__(**query)
        self.class_mapping = class_mapping
        self.class_property = class_property
        self.min_fraction = _normalize_min_fraction(min_fraction)
        self.overlap_method = _normalize_overlap_method(overlap_method)

    def _covered(self: _CoverageTargetBuilder, fraction: float) -> bool:
        """Return whether a coverage fraction counts as present."""
        return fraction > 0 and fraction >= self.min_fraction

    def _class_coverages(
        self: _CoverageTargetBuilder,
        annotations: dict[str, Annotation],
        patch_bounds: Bounds,
        output_shape: tuple[int, int],
    ) -> dict[int, float]:
        """Measure each mapped class's coverage fraction of the patch."""
        return _per_class_coverage(
            annotations,
            patch_bounds,
            output_shape,
            class_property=self.class_property,
            class_mapping=self.class_mapping,
            overlap_method=self.overlap_method,
        )


class PresenceTargetBuilder(_CoverageTargetBuilder):
    """Build a scalar target from positive annotation coverage.

    A zero threshold still requires positive coverage. Exact coverage measures
    area; use rasterized coverage to include point or line annotations.
    """

    def __init__(
        self: PresenceTargetBuilder,
        *,
        min_fraction: float = 0.0,
        positive_label: int = 1,
        negative_label: int = 0,
        overlap_method: Literal["exact", "rasterized"] = "exact",
        **query: Unpack[AnnotationQuery],
    ) -> None:
        """Initialize :class:`PresenceTargetBuilder`."""
        super().__init__(
            min_fraction=min_fraction, overlap_method=overlap_method, **query
        )
        self.positive_label = int(positive_label)
        self.negative_label = int(negative_label)

    def build_target(
        self: PresenceTargetBuilder,
        annotations: dict[str, Annotation],
        patch_bounds: Bounds,
        output_shape: tuple[int, int],
    ) -> int:
        """Build scalar presence target from queried annotations."""
        fraction = _coverage_fraction(
            annotations,
            patch_bounds,
            output_shape,
            overlap_method=self.overlap_method,
        )
        return self.positive_label if self._covered(fraction) else self.negative_label


class CoverageClassTargetBuilder(_CoverageTargetBuilder):
    """Build class target from annotation class with highest area coverage."""

    def __init__(
        self: CoverageClassTargetBuilder,
        *,
        class_mapping: dict[object, int],
        class_property: str = "class",
        min_fraction: float = 0.0,
        default_label: int = 0,
        overlap_method: Literal["exact", "rasterized"] = "exact",
        **query: Unpack[AnnotationQuery],
    ) -> None:
        """Initialize :class:`CoverageClassTargetBuilder`."""
        super().__init__(
            class_mapping=class_mapping,
            class_property=class_property,
            min_fraction=min_fraction,
            overlap_method=overlap_method,
            **query,
        )
        self.default_label = int(default_label)

    def build_target(
        self: CoverageClassTargetBuilder,
        annotations: dict[str, Annotation],
        patch_bounds: Bounds,
        output_shape: tuple[int, int],
    ) -> int:
        """Build scalar class target from per-class overlap fractions."""
        class_coverages = self._class_coverages(annotations, patch_bounds, output_shape)
        if not class_coverages:
            return self.default_label
        best_label, best_fraction = max(
            sorted(class_coverages.items(), key=lambda item: item[0]),
            key=lambda item: item[1],
        )
        return int(best_label) if self._covered(best_fraction) else self.default_label


class MultiLabelTargetBuilder(_CoverageTargetBuilder):
    """Build a multi-label target vector from class-wise annotation coverage."""

    def __init__(
        self: MultiLabelTargetBuilder,
        *,
        class_mapping: dict[object, int],
        class_property: str = "class",
        min_fraction: float = 0.0,
        overlap_method: Literal["exact", "rasterized"] = "exact",
        **query: Unpack[AnnotationQuery],
    ) -> None:
        """Initialize :class:`MultiLabelTargetBuilder`."""
        super().__init__(
            class_mapping=class_mapping,
            class_property=class_property,
            min_fraction=min_fraction,
            overlap_method=overlap_method,
            **query,
        )
        mapped_values = list(class_mapping.values())
        if not mapped_values:
            msg = "`class_mapping` must contain at least one class entry."
            raise ValueError(msg)
        self.num_classes = int(max(mapped_values)) + 1

    def build_target(
        self: MultiLabelTargetBuilder,
        annotations: dict[str, Annotation],
        patch_bounds: Bounds,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """Build a multi-label binary vector from class coverage fractions."""
        target = np.zeros((self.num_classes,), dtype=np.int64)
        coverages = self._class_coverages(annotations, patch_bounds, output_shape)
        for class_label, fraction in coverages.items():
            if self._covered(fraction):
                target[int(class_label)] = 1
        return target


class BinaryDiskTargetBuilder(TargetBuilderABC):
    """Build a dense binary disk map centered on annotation centroids.

    Annotations are converted to point targets using their original geometry
    centroid. Only centroids whose original coordinates fall inside the patch
    bounds are included.
    """

    def __init__(
        self: BinaryDiskTargetBuilder,
        *,
        radius: int = 3,
        positive_value: float = 1.0,
        background_value: float = 0.0,
        **query: Unpack[AnnotationQuery],
    ) -> None:
        """Initialize :class:`BinaryDiskTargetBuilder`."""
        super().__init__(**query)
        self.radius = int(radius)
        self.positive_value = float(positive_value)
        self.background_value = float(background_value)
        if self.radius < 0:
            msg = "`radius` must be a non-negative integer."
            raise ValueError(msg)

    def build_target(
        self: BinaryDiskTargetBuilder,
        annotations: dict[str, Annotation],
        patch_bounds: Bounds,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """Build a dense float32 disk target map."""
        height, width = output_shape
        target = np.full((height, width), self.background_value, dtype=np.float32)

        for point_x, point_y in _annotation_centroids_in_patch(
            annotations,
            patch_bounds,
            output_shape,
        ):
            center = (round(point_x), round(point_y))
            cv2.circle(
                target,
                center,
                self.radius,
                color=self.positive_value,
                thickness=-1,
            )

        return target


class GaussianHeatmapTargetBuilder(TargetBuilderABC):
    """Build a dense Gaussian heatmap centered on annotation centroids."""

    def __init__(
        self: GaussianHeatmapTargetBuilder,
        *,
        sigma: float = 2.0,
        peak_value: float = 1.0,
        truncate: float = 3.0,
        background_value: float = 0.0,
        **query: Unpack[AnnotationQuery],
    ) -> None:
        """Initialize :class:`GaussianHeatmapTargetBuilder`."""
        super().__init__(**query)
        self.sigma = float(sigma)
        self.peak_value = float(peak_value)
        self.truncate = float(truncate)
        self.background_value = float(background_value)
        if self.sigma <= 0:
            msg = "`sigma` must be positive."
            raise ValueError(msg)
        if self.truncate <= 0:
            msg = "`truncate` must be positive."
            raise ValueError(msg)

    def build_target(
        self: GaussianHeatmapTargetBuilder,
        annotations: dict[str, Annotation],
        patch_bounds: Bounds,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """Build a dense float32 Gaussian target map."""
        height, width = output_shape
        target = np.full((height, width), self.background_value, dtype=np.float32)
        radius = int(np.ceil(self.truncate * self.sigma))

        for point_x, point_y in _annotation_centroids_in_patch(
            annotations,
            patch_bounds,
            output_shape,
        ):
            center_x = math.floor(point_x)
            center_y = math.floor(point_y)
            x_min = max(center_x - radius, 0)
            x_max = min(center_x + radius + 1, width)
            y_min = max(center_y - radius, 0)
            y_max = min(center_y + radius + 1, height)

            # Broadcast 1D pixel coordinates rather than building a dense
            # ``np.mgrid`` per centroid; the per-pixel arithmetic is unchanged.
            x_coords = np.arange(x_min, x_max, dtype=np.int64)[np.newaxis, :]
            y_coords = np.arange(y_min, y_max, dtype=np.int64)[:, np.newaxis]
            squared_distance = (x_coords - point_x) ** 2 + (y_coords - point_y) ** 2
            gaussian = self.peak_value * np.exp(
                -squared_distance / (2.0 * self.sigma**2)
            )
            target[y_min:y_max, x_min:x_max] = np.maximum(
                target[y_min:y_max, x_min:x_max],
                gaussian.astype(np.float32, copy=False),
            )

        return target


def _coverage_fraction(
    annotations: dict[str, Annotation],
    patch_bounds: Bounds,
    output_shape: tuple[int, int],
    *,
    overlap_method: Literal["exact", "rasterized"],
) -> float:
    """Compute total annotation coverage fraction inside a patch."""
    if not annotations:
        return 0.0

    if overlap_method == "rasterized":
        mask_builder = MaskTargetBuilder(class_mapping=None, default_label=0)
        mask = mask_builder.build_target(annotations, patch_bounds, output_shape)
        return float(np.mean(mask > 0))

    patch_geometry = _patch_box(patch_bounds)
    intersections = []
    for annotation in annotations.values():
        overlap = annotation.geometry.intersection(patch_geometry)
        if not overlap.is_empty:
            intersections.append(overlap)

    if not intersections:
        return 0.0

    x_min, y_min, x_max, y_max = patch_bounds
    patch_area = (x_max - x_min) * (y_max - y_min)
    if patch_area <= 0:
        return 0.0

    return float(unary_union(intersections).area / patch_area)


def _per_class_coverage(
    annotations: dict[str, Annotation],
    patch_bounds: Bounds,
    output_shape: tuple[int, int],
    *,
    class_property: str,
    class_mapping: dict[object, int],
    overlap_method: Literal["exact", "rasterized"],
) -> dict[int, float]:
    """Compute class-wise coverage fractions for a patch."""
    if not annotations:
        return {}

    if overlap_method == "rasterized":
        grouped: dict[int, dict[str, Annotation]] = {}
        for key, annotation in annotations.items():
            class_value = annotation.properties.get(class_property)
            if class_value in class_mapping:
                grouped.setdefault(int(class_mapping[class_value]), {})[key] = (
                    annotation
                )
        return {
            label: _coverage_fraction(
                members, patch_bounds, output_shape, overlap_method="rasterized"
            )
            for label, members in grouped.items()
        }

    patch_geometry = _patch_box(patch_bounds)
    x_min, y_min, x_max, y_max = patch_bounds
    patch_area = (x_max - x_min) * (y_max - y_min)
    if patch_area <= 0:
        return {}

    per_class_geometries: dict[int, list[Any]] = {}
    for annotation in annotations.values():
        class_value = annotation.properties.get(class_property)
        if class_value not in class_mapping:
            continue
        class_label = int(class_mapping[class_value])

        overlap = annotation.geometry.intersection(patch_geometry)
        if overlap.is_empty:
            continue

        per_class_geometries.setdefault(class_label, []).append(overlap)

    return {
        class_label: float(unary_union(overlaps).area / patch_area)
        for class_label, overlaps in per_class_geometries.items()
    }
