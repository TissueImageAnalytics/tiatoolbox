"""Annotation-to-target builders for training datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from tiatoolbox.annotation import Annotation, AnnotationStore

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.type_hints import Predicate

Bounds = tuple[float, float, float, float]
TargetType = np.ndarray | int | float | dict[str, "TargetType"]


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
    pixel_x = float(np.clip(pixel_x, 0.0, max(width - 1, 0)))
    pixel_y = float(np.clip(pixel_y, 0.0, max(height - 1, 0)))
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


def _clip_and_rescale_geometry(
    geometry: Any,
    bounds: Bounds,
    output_shape: tuple[int, int],
) -> Any | None:
    """Clip geometry to patch bounds and map into output pixel space."""
    patch_box = _patch_box(bounds)
    clipped = geometry.intersection(patch_box)
    if clipped.is_empty:
        return None

    x_min, y_min, x_max, y_max = bounds
    height, width = output_shape
    scale_x = width / (x_max - x_min)
    scale_y = height / (y_max - y_min)

    translated = shapely_translate(clipped, xoff=-x_min, yoff=-y_min)
    return shapely_scale(translated, xfact=scale_x, yfact=scale_y, origin=(0, 0))


def _coords_to_int(coords: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert XY coordinates to clipped int32 coordinates for OpenCV."""
    coords = np.round(coords).astype(np.int32)
    coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
    return coords


def _rasterize_geometry(
    mask: np.ndarray,
    geometry: Any,
    label: int,
    *,
    background_label: int,
    line_width: int,
    point_radius: int,
) -> None:
    """Rasterize a single shapely geometry onto a label mask."""
    height, width = mask.shape
    geom_type = geometry.geom_type

    if geom_type == "Polygon":
        exterior = _coords_to_int(np.asarray(geometry.exterior.coords), width, height)
        cv2.fillPoly(mask, [exterior], int(label))
        for interior in geometry.interiors:
            interior_coords = _coords_to_int(
                np.asarray(interior.coords),
                width,
                height,
            )
            cv2.fillPoly(mask, [interior_coords], int(background_label))
        return

    if geom_type == "MultiPolygon":
        for part in geometry.geoms:
            _rasterize_geometry(
                mask,
                part,
                label,
                background_label=background_label,
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
            color=int(label),
            thickness=line_width,
        )
        return

    if geom_type == "MultiLineString":
        for part in geometry.geoms:
            _rasterize_geometry(
                mask,
                part,
                label,
                background_label=background_label,
                line_width=line_width,
                point_radius=point_radius,
            )
        return

    if geom_type == "Point":
        point_coords = _coords_to_int(np.asarray(geometry.coords), width, height)[0]
        cv2.circle(mask, tuple(point_coords), point_radius, int(label), thickness=-1)
        return

    if geom_type == "MultiPoint":
        for part in geometry.geoms:
            _rasterize_geometry(
                mask,
                part,
                label,
                background_label=background_label,
                line_width=line_width,
                point_radius=point_radius,
            )


class TargetBuilderABC(ABC):
    """Base interface for building patch targets from annotation stores."""

    def __init__(
        self: TargetBuilderABC,
        *,
        where: Predicate | None = None,
        geometry_predicate: str = "intersects",
    ) -> None:
        """Initialize :class:`TargetBuilderABC`."""
        self.where = where
        self.geometry_predicate = geometry_predicate

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
        )

    def create_target(
        self: TargetBuilderABC,
        *,
        store: AnnotationStore,
        patch_bounds: tuple[float, ...] | list[float] | np.ndarray,
        output_shape: tuple[int, int] | list[int] | np.ndarray,
    ) -> TargetType:
        """Query annotations and build a target for one patch."""
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


class CompositeTargetBuilder(TargetBuilderABC):
    """Combine multiple target builders into a nested target dictionary."""

    def __init__(
        self: CompositeTargetBuilder,
        builders: dict[str, TargetBuilderABC],
    ) -> None:
        """Initialize :class:`CompositeTargetBuilder`."""
        super().__init__()
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
        """Build a structured target by delegating to named sub-builders."""
        normalized_bounds = _as_bounds(patch_bounds)
        normalized_shape = _normalize_output_shape(output_shape)
        return {
            name: builder.create_target(
                store=store,
                patch_bounds=normalized_bounds,
                output_shape=normalized_shape,
            )
            for name, builder in self.builders.items()
        }

    def build_target(
        self: CompositeTargetBuilder,
        annotations: dict[str, Annotation],
        patch_bounds: Bounds,
        output_shape: tuple[int, int],
    ) -> dict[str, TargetType]:
        """Build a structured target from shared queried annotations."""
        return {
            name: builder.build_target(annotations, patch_bounds, output_shape)
            for name, builder in self.builders.items()
        }


class MaskTargetBuilder(TargetBuilderABC):
    """Build integer semantic masks from annotations.

    Overlapping annotations are rasterized in query iteration order, so later
    annotations overwrite earlier labels where they intersect.
    """

    def __init__(
        self: MaskTargetBuilder,
        *,
        class_mapping: dict[object, int] | None = None,
        class_property: str = "class",
        default_label: int = 0,
        where: Predicate | None = None,
        geometry_predicate: str = "intersects",
        line_width: int = 1,
        point_radius: int = 1,
    ) -> None:
        """Initialize :class:`MaskTargetBuilder`."""
        super().__init__(where=where, geometry_predicate=geometry_predicate)
        self.class_mapping = class_mapping
        self.class_property = class_property
        self.default_label = int(default_label)
        self.line_width = int(line_width)
        self.point_radius = int(point_radius)
        if self.line_width <= 0:
            msg = "`line_width` must be a positive integer."
            raise ValueError(msg)
        if self.point_radius <= 0:
            msg = "`point_radius` must be a positive integer."
            raise ValueError(msg)

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

        for annotation in annotations.values():
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
                label,
                background_label=self.default_label,
                line_width=self.line_width,
                point_radius=self.point_radius,
            )

        return mask.astype(np.int64)


class PresenceTargetBuilder(TargetBuilderABC):
    """Build a binary/class scalar target from annotation presence."""

    def __init__(
        self: PresenceTargetBuilder,
        *,
        min_fraction: float = 0.0,
        positive_label: int = 1,
        negative_label: int = 0,
        where: Predicate | None = None,
        geometry_predicate: str = "intersects",
        overlap_method: Literal["exact", "rasterized"] = "exact",
    ) -> None:
        """Initialize :class:`PresenceTargetBuilder`."""
        super().__init__(where=where, geometry_predicate=geometry_predicate)
        self.min_fraction = _normalize_min_fraction(min_fraction)
        self.positive_label = int(positive_label)
        self.negative_label = int(negative_label)
        self.overlap_method = _normalize_overlap_method(overlap_method)

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
        if fraction >= self.min_fraction:
            return self.positive_label
        return self.negative_label


class CoverageClassTargetBuilder(TargetBuilderABC):
    """Build class target from annotation class with highest area coverage."""

    def __init__(
        self: CoverageClassTargetBuilder,
        *,
        class_mapping: dict[object, int],
        class_property: str = "class",
        min_fraction: float = 0.0,
        default_label: int = 0,
        where: Predicate | None = None,
        geometry_predicate: str = "intersects",
        overlap_method: Literal["exact", "rasterized"] = "exact",
    ) -> None:
        """Initialize :class:`CoverageClassTargetBuilder`."""
        super().__init__(where=where, geometry_predicate=geometry_predicate)
        self.class_mapping = class_mapping
        self.class_property = class_property
        self.min_fraction = _normalize_min_fraction(min_fraction)
        self.default_label = int(default_label)
        self.overlap_method = _normalize_overlap_method(overlap_method)

    def build_target(
        self: CoverageClassTargetBuilder,
        annotations: dict[str, Annotation],
        patch_bounds: Bounds,
        output_shape: tuple[int, int],
    ) -> int:
        """Build scalar class target from per-class overlap fractions."""
        class_coverages = _per_class_coverage(
            annotations,
            patch_bounds,
            output_shape,
            class_property=self.class_property,
            class_mapping=self.class_mapping,
            overlap_method=self.overlap_method,
        )

        if not class_coverages:
            return self.default_label

        best_label, best_fraction = max(
            sorted(class_coverages.items(), key=lambda item: item[0]),
            key=lambda item: item[1],
        )
        if best_fraction < self.min_fraction:
            return self.default_label

        return int(best_label)


class MultiLabelTargetBuilder(TargetBuilderABC):
    """Build a multi-label target vector from class-wise annotation coverage."""

    def __init__(
        self: MultiLabelTargetBuilder,
        *,
        class_mapping: dict[object, int],
        class_property: str = "class",
        min_fraction: float = 0.0,
        where: Predicate | None = None,
        geometry_predicate: str = "intersects",
        overlap_method: Literal["exact", "rasterized"] = "exact",
    ) -> None:
        """Initialize :class:`MultiLabelTargetBuilder`."""
        super().__init__(where=where, geometry_predicate=geometry_predicate)
        self.class_mapping = class_mapping
        self.class_property = class_property
        self.min_fraction = _normalize_min_fraction(min_fraction)
        self.overlap_method = _normalize_overlap_method(overlap_method)

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
        class_coverages = _per_class_coverage(
            annotations,
            patch_bounds,
            output_shape,
            class_property=self.class_property,
            class_mapping=self.class_mapping,
            overlap_method=self.overlap_method,
        )

        target = np.zeros((self.num_classes,), dtype=np.int64)
        for class_label, fraction in class_coverages.items():
            if fraction >= self.min_fraction:
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
        where: Predicate | None = None,
        geometry_predicate: str = "intersects",
    ) -> None:
        """Initialize :class:`BinaryDiskTargetBuilder`."""
        super().__init__(where=where, geometry_predicate=geometry_predicate)
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
            center = (int(round(point_x)), int(round(point_y)))
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
        where: Predicate | None = None,
        geometry_predicate: str = "intersects",
    ) -> None:
        """Initialize :class:`GaussianHeatmapTargetBuilder`."""
        super().__init__(where=where, geometry_predicate=geometry_predicate)
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
            x_min = max(int(np.floor(point_x)) - radius, 0)
            x_max = min(int(np.floor(point_x)) + radius + 1, width)
            y_min = max(int(np.floor(point_y)) - radius, 0)
            y_max = min(int(np.floor(point_y)) + radius + 1, height)

            if x_min >= x_max or y_min >= y_max:
                continue

            y_coords, x_coords = np.mgrid[y_min:y_max, x_min:x_max]
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
        mask_builder = MaskTargetBuilder(
            class_mapping=class_mapping,
            class_property=class_property,
            default_label=-1,
        )
        mask = mask_builder.build_target(annotations, patch_bounds, output_shape)
        class_coverages: dict[int, float] = {}
        total_pixels = mask.size
        if total_pixels == 0:
            return class_coverages

        for mapped_label in set(class_mapping.values()):
            class_coverages[int(mapped_label)] = float(np.mean(mask == mapped_label))
        return class_coverages

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
        if overlaps
    }
