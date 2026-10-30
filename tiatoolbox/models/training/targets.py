"""Annotation-to-target builders for training datasets."""

from __future__ import annotations

import functools
import itertools
import math
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import cv2
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from tiatoolbox.annotation.dsl import PY_GLOBALS

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from shapely.geometry.base import BaseGeometry

    from tiatoolbox.annotation import Annotation, AnnotationStore
    from tiatoolbox.type_hints import Predicate, Properties

_SPATIAL_DIMS = 2

Bounds = tuple[float, float, float, float]
TargetType = np.ndarray | int | float | dict[str, "TargetType"]
SpatialTargetKind = Literal["mask", "image"]
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


@dataclass(frozen=True)
class StackedSpatialTargetSpec:
    """Describe per-channel interpolation for a stacked spatial target."""

    channels: tuple[tuple[str, SpatialTargetKind], ...]

    def __post_init__(self: StackedSpatialTargetSpec) -> None:
        """Validate stacked channel specs."""
        if not self.channels:
            msg = "Stacked spatial target specs must contain at least one channel."
            raise ValueError(msg)

        names = [name for name, _kind in self.channels]
        if len(names) != len(set(names)):
            msg = "Stacked spatial target channel names must be unique."
            raise ValueError(msg)

        for name, kind in self.channels:
            if not isinstance(name, str):
                msg = "Stacked spatial target channel names must be strings."
                raise ValueError(msg)  # noqa: TRY004 - preserve validation API
            if kind not in {"mask", "image"}:
                msg = (
                    f"Unsupported stacked spatial target kind `{kind}` for "
                    f"channel `{name}`. Choose from `mask` or `image`."
                )
                raise ValueError(msg)


SpatialTargetSpec = (
    SpatialTargetKind | StackedSpatialTargetSpec | dict[str, "SpatialTargetSpec"]
)


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
    patch pixel coordinates with shape ``output_shape``. Every builder accepts
    the :class:`AnnotationQuery` options ``where``, ``geometry_predicate``,
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

    @property
    def spatial_target_spec(self: TargetBuilderABC) -> SpatialTargetSpec | None:
        """Describe which target leaves should share spatial transforms."""
        return None

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


def _has_non_where_query_settings(builder: TargetBuilderABC) -> bool:
    """Return whether a builder has query settings that need a store query.

    Only ``where`` can be applied to an already-queried annotation mapping;
    ``geometry_predicate``, ``min_area`` and ``distance`` cannot.
    """
    return (
        builder.geometry_predicate != DEFAULT_GEOMETRY_PREDICATE
        or builder.min_area is not None
        or builder.distance != DEFAULT_DISTANCE
    )


def _validate_shared_query_children(
    parent: TargetBuilderABC,
    builders: dict[str, TargetBuilderABC],
) -> None:
    """Ensure children handed a shared query result only filter with ``where``."""
    conflicting = [
        name
        for name, builder in builders.items()
        if _has_non_where_query_settings(builder)
    ]
    if conflicting:
        msg = (
            f"{type(parent).__name__} cannot combine parent query settings with "
            "child `geometry_predicate`, `min_area` or `distance` settings: "
            "children of a shared annotation query may only filter with `where`. "
            "Move these settings to the parent builder or remove the parent "
            f"query settings. Conflicting children: {', '.join(conflicting)}."
        )
        raise ValueError(msg)


@functools.lru_cache(maxsize=128)
def _compile_where(where: str | bytes) -> Callable[[Properties], bool]:
    """Compile a string or pickled ``where`` predicate into a Python callable.

    Mirrors :meth:`AnnotationStore._eval_where`, which ``DictionaryStore`` uses
    to filter annotations in Python, but string predicates are compiled once
    and cached so per-annotation evaluation stays cheap.
    """
    if isinstance(where, bytes):
        predicate = pickle.loads(where)  # skipcq: BAN-B301  # noqa: S301
        return lambda properties: bool(predicate(properties))

    code = compile(where, "<where>", "eval")

    def evaluate(properties: Properties) -> bool:
        """Evaluate the compiled annotation predicate against its properties."""
        return bool(
            eval(  # skipcq: PYL-W0123,  # noqa: S307
                code,
                PY_GLOBALS,
                {"props": properties},
            ),
        )

    return evaluate


def _filter_annotations(
    annotations: dict[str, Annotation],
    where: Predicate | None,
) -> dict[str, Annotation]:
    """Filter a queried annotation mapping with a ``where`` predicate in Python."""
    if where is None:
        return annotations
    predicate = _compile_where(where) if isinstance(where, str | bytes) else where
    return {
        key: annotation
        for key, annotation in annotations.items()
        if predicate(annotation.properties)
    }


class CompositeTargetBuilder(TargetBuilderABC):
    """Combine multiple target builders into a nested target dictionary.

    Without query settings on the composite, every child builder issues its
    own annotation query. When ``where``, ``geometry_predicate``, ``min_area``
    or ``distance`` is set on the composite, a single shared query is issued
    and its result is handed to every child. Children may narrow the shared
    result with their own ``where`` predicate, which is then evaluated in
    Python per annotation, but may not set other query settings.
    """

    def __init__(
        self: CompositeTargetBuilder,
        builders: dict[str, TargetBuilderABC],
        **query: Unpack[AnnotationQuery],
    ) -> None:
        """Initialize :class:`CompositeTargetBuilder`."""
        super().__init__(**query)
        if not builders:
            msg = "`builders` must contain at least one target builder."
            raise ValueError(msg)
        self.builders = builders

    def create_target(
        self: CompositeTargetBuilder,
        *,
        store: AnnotationStore,
        patch_bounds: tuple[float, ...] | list[float] | np.ndarray,
        output_shape: tuple[int, int] | list[int] | np.ndarray,
    ) -> dict[str, TargetType]:
        """Build a structured target by delegating to named sub-builders.

        A parent-level query is shared by all children when the composite owns
        query settings; children may then narrow the shared result with their
        own ``where`` predicate. Otherwise, each child builder issues its own
        annotation query.
        """
        normalized_bounds = _as_bounds(patch_bounds)
        normalized_shape = _normalize_output_shape(output_shape)

        if _has_query_settings(self):
            _validate_shared_query_children(self, self.builders)
            annotations = self.query_annotations(store, normalized_bounds)
            return self.build_target(annotations, normalized_bounds, normalized_shape)

        return {
            name: builder.create_target(
                store=store,
                patch_bounds=normalized_bounds,
                output_shape=normalized_shape,
            )
            for name, builder in self.builders.items()
        }

    @property
    def spatial_target_spec(
        self: CompositeTargetBuilder,
    ) -> dict[str, SpatialTargetSpec] | None:
        """Describe the nested spatial leaves produced by child builders."""
        spatial_specs: dict[str, SpatialTargetSpec] = {}
        for name, builder in self.builders.items():
            child_spec = builder.spatial_target_spec
            if child_spec is not None:
                spatial_specs[name] = child_spec
        if not spatial_specs:
            return None
        return spatial_specs

    def build_target(
        self: CompositeTargetBuilder,
        annotations: dict[str, Annotation],
        patch_bounds: Bounds,
        output_shape: tuple[int, int],
    ) -> dict[str, TargetType]:
        """Build a structured target from shared queried annotations.

        Each child receives the annotations that satisfy its own ``where``
        predicate, evaluated in Python per annotation. Children with other
        query settings (``geometry_predicate``, ``min_area``, ``distance``)
        raise a :class:`ValueError`; use :meth:`create_target` without parent
        query settings when children need distinct queries.
        """
        _validate_shared_query_children(self, self.builders)
        return {
            name: builder.build_target(
                _filter_annotations(annotations, builder.where),
                patch_bounds,
                output_shape,
            )
            for name, builder in self.builders.items()
        }


class StackedTargetBuilder(TargetBuilderABC):
    """Stack multiple spatial target maps into a single multi-channel array."""

    def __init__(
        self: StackedTargetBuilder,
        builders: dict[str, TargetBuilderABC],
        *,
        dtype: np.dtype = np.float32,
        **query: Unpack[AnnotationQuery],
    ) -> None:
        """Initialize :class:`StackedTargetBuilder`."""
        super().__init__(**query)
        if not builders:
            msg = "`builders` must contain at least one target builder."
            raise ValueError(msg)
        self.builders = builders
        self.dtype = np.dtype(dtype)

    @property
    def spatial_target_spec(self: StackedTargetBuilder) -> StackedSpatialTargetSpec:
        """Describe per-channel interpolation for the stacked spatial map."""
        child_specs: list[tuple[str, SpatialTargetKind]] = []
        for name, builder in self.builders.items():
            child_spec = builder.spatial_target_spec
            if not isinstance(child_spec, str) or child_spec not in {"mask", "image"}:
                msg = (
                    "StackedTargetBuilder only supports child builders with "
                    f"leaf spatial specs, but `{name}` returned `{child_spec}`."
                )
                raise ValueError(msg)
            child_specs.append((name, child_spec))
        return StackedSpatialTargetSpec(tuple(child_specs))

    def create_target(
        self: StackedTargetBuilder,
        *,
        store: AnnotationStore,
        patch_bounds: tuple[float, ...] | list[float] | np.ndarray,
        output_shape: tuple[int, int] | list[int] | np.ndarray,
    ) -> np.ndarray:
        """Build and stack all child targets for one patch.

        A parent-level query is shared by all children when the stack owns query
        settings; children may then narrow the shared result with their own
        ``where`` predicate, evaluated in Python, but may not set other query
        settings. Otherwise, each child builder owns its annotation query
        configuration so per-channel filters such as ``where`` and
        ``geometry_predicate`` remain intact.
        """
        normalized_bounds = _as_bounds(patch_bounds)
        normalized_shape = _normalize_output_shape(output_shape)

        if _has_query_settings(self):
            _validate_shared_query_children(self, self.builders)
            annotations = self.query_annotations(store, normalized_bounds)
            return self.build_target(annotations, normalized_bounds, normalized_shape)

        channel_targets = {
            name: builder.create_target(
                store=store,
                patch_bounds=normalized_bounds,
                output_shape=normalized_shape,
            )
            for name, builder in self.builders.items()
        }
        return self._stack_channel_targets(channel_targets, normalized_shape)

    def _stack_channel_targets(
        self: StackedTargetBuilder,
        channel_targets: dict[str, TargetType],
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """Validate and stack named 2D channel targets."""
        stacked_targets: list[np.ndarray] = []
        for name, child_target in channel_targets.items():
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

    def build_target(
        self: StackedTargetBuilder,
        annotations: dict[str, Annotation],
        patch_bounds: Bounds,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """Build a dense stacked target with channels on the last axis.

        This method consumes an already-queried annotation mapping and hands
        each child the annotations that satisfy its own ``where`` predicate,
        evaluated in Python per annotation. Children with other query settings
        (``geometry_predicate``, ``min_area``, ``distance``) raise a
        :class:`ValueError`; use :meth:`create_target` without parent query
        settings when child builders need distinct queries.
        """
        _validate_shared_query_children(self, self.builders)
        channel_targets = {
            name: builder.build_target(
                _filter_annotations(annotations, builder.where),
                patch_bounds,
                output_shape,
            )
            for name, builder in self.builders.items()
        }
        return self._stack_channel_targets(channel_targets, output_shape)


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

    @property
    def spatial_target_spec(self: MaskTargetBuilder) -> Literal["mask"]:
        """Rasterized semantic masks should use mask interpolation."""
        return "mask"

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

    @property
    def spatial_target_spec(self: BoundaryTargetBuilder) -> Literal["mask"]:
        """Binary boundary maps should use mask interpolation."""
        return "mask"

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

    @property
    def spatial_target_spec(self: BinaryDiskTargetBuilder) -> Literal["mask"]:
        """Binary disk maps should use mask interpolation."""
        return "mask"

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

    @property
    def spatial_target_spec(
        self: GaussianHeatmapTargetBuilder,
    ) -> Literal["image"]:
        """Continuous heatmaps should use image interpolation."""
        return "image"

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
