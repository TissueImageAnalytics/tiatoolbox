"""Tests for annotation-based training targets and datasets."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from shapely.geometry import Point, Polygon

from tiatoolbox.annotation import Annotation, SQLiteStore
from tiatoolbox.models.training import (
    BinaryDiskTargetBuilder,
    BoundaryTargetBuilder,
    ClassBalancedIndexSampler,
    CompositeTargetBuilder,
    CoverageClassTargetBuilder,
    GaussianHeatmapTargetBuilder,
    MaskTargetBuilder,
    MultiLabelTargetBuilder,
    PatchAnnotationDataset,
    PresenceTargetBuilder,
    SlideAnnotationPatchDataset,
    StackedSpatialTargetSpec,
    StackedTargetBuilder,
    generate_slide_patch_coordinates,
)


def _create_test_store(store_path: Path) -> SQLiteStore:
    """Create a small annotation store for synthetic target tests."""
    store = SQLiteStore(store_path)

    store.append(
        Annotation(
            Polygon([(0, 0), (5, 0), (5, 10), (0, 10)]),
            properties={"class": "tumor"},
        ),
        key="tumor",
    )
    store.append(
        Annotation(
            Polygon([(5, 0), (10, 0), (10, 5), (5, 5)]),
            properties={"class": "stroma"},
        ),
        key="stroma",
    )
    return store


def _flip_left_right(
    image: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip an image and target together for deterministic alignment tests."""
    return (
        np.ascontiguousarray(np.fliplr(image)),
        np.ascontiguousarray(np.fliplr(target)),
    )


def test_mask_target_builder(track_tmp_path: Path) -> None:
    """Mask target builder should rasterize class labels from annotations."""
    store = _create_test_store(track_tmp_path / "targets.db")

    builder = MaskTargetBuilder(
        class_mapping={"tumor": 1, "stroma": 2},
        class_property="class",
        default_label=0,
    )
    target = builder.create_target(
        store=store,
        patch_bounds=(0, 0, 10, 10),
        output_shape=(10, 10),
    )

    assert target.shape == (10, 10)
    assert target[2, 2] == 1
    assert target[2, 7] == 2
    assert target[7, 7] == 0


def test_target_builders_pass_query_options(track_tmp_path: Path) -> None:
    """Target builders should pass store query options through consistently."""
    store = _create_test_store(track_tmp_path / "query_options.db")

    area_filtered = MaskTargetBuilder(
        class_mapping={"tumor": 1, "stroma": 2},
        class_property="class",
        min_area=30,
    ).create_target(
        store=store,
        patch_bounds=(0, 0, 10, 10),
        output_shape=(10, 10),
    )

    assert area_filtered[2, 2] == 1
    assert area_filtered[2, 7] == 0

    point_store = SQLiteStore(track_tmp_path / "distance_query_options.db")
    point_store.append(Annotation(Point(5, 5), properties={}), key="near")
    point_store.append(Annotation(Point(9, 9), properties={}), key="far")

    distance_filtered = BinaryDiskTargetBuilder(
        radius=0,
        geometry_predicate="centers_within_k",
        distance=2.0,
    ).create_target(
        store=point_store,
        patch_bounds=(0, 0, 10, 10),
        output_shape=(10, 10),
    )

    assert distance_filtered[5, 5] == pytest.approx(1.0)
    assert distance_filtered[9, 9] == pytest.approx(0.0)


def test_presence_coverage_and_multilabel_builders(track_tmp_path: Path) -> None:
    """Coverage-based builders should produce expected scalar/vector targets."""
    store = _create_test_store(track_tmp_path / "targets_coverage.db")

    presence_low = PresenceTargetBuilder(min_fraction=0.2)
    assert (
        presence_low.create_target(
            store=store,
            patch_bounds=(0, 0, 10, 10),
            output_shape=(10, 10),
        )
        == 1
    )

    presence_high = PresenceTargetBuilder(min_fraction=0.8)
    assert (
        presence_high.create_target(
            store=store,
            patch_bounds=(0, 0, 10, 10),
            output_shape=(10, 10),
        )
        == 0
    )

    coverage_builder = CoverageClassTargetBuilder(
        class_mapping={"tumor": 3, "stroma": 4},
        class_property="class",
        min_fraction=0.0,
        default_label=-1,
    )
    assert (
        coverage_builder.create_target(
            store=store,
            patch_bounds=(0, 0, 10, 10),
            output_shape=(10, 10),
        )
        == 3
    )

    coverage_threshold_builder = CoverageClassTargetBuilder(
        class_mapping={"tumor": 3, "stroma": 4},
        class_property="class",
        min_fraction=0.6,
        default_label=-1,
    )
    assert (
        coverage_threshold_builder.create_target(
            store=store,
            patch_bounds=(0, 0, 10, 10),
            output_shape=(10, 10),
        )
        == -1
    )

    multilabel_builder = MultiLabelTargetBuilder(
        class_mapping={"tumor": 0, "stroma": 1, "necrosis": 2},
        class_property="class",
        min_fraction=0.2,
    )
    target = multilabel_builder.create_target(
        store=store,
        patch_bounds=(0, 0, 10, 10),
        output_shape=(10, 10),
    )
    assert np.array_equal(target, np.array([1, 1, 0]))


def test_dense_point_target_builders(track_tmp_path: Path) -> None:
    """Dense point builders should generate centered detection maps."""
    store_path = track_tmp_path / "dense_targets.db"
    store = SQLiteStore(store_path)
    store.append(
        Annotation(Point(2, 2), properties={"class": "tumor"}),
        key="point_a",
    )
    store.append(
        Annotation(Point(7, 7), properties={"class": "tumor"}),
        key="point_b",
    )

    disk_target = BinaryDiskTargetBuilder(radius=1).create_target(
        store=store,
        patch_bounds=(0, 0, 10, 10),
        output_shape=(10, 10),
    )
    heatmap_target = GaussianHeatmapTargetBuilder(sigma=1.0).create_target(
        store=store,
        patch_bounds=(0, 0, 10, 10),
        output_shape=(10, 10),
    )

    assert disk_target.shape == (10, 10)
    assert disk_target.dtype == np.float32
    assert disk_target[2, 2] == pytest.approx(1.0)
    assert disk_target[7, 7] == pytest.approx(1.0)
    assert disk_target[0, 9] == pytest.approx(0.0)

    assert heatmap_target.shape == (10, 10)
    assert heatmap_target.dtype == np.float32
    assert heatmap_target[2, 2] == pytest.approx(1.0)
    assert heatmap_target[7, 7] == pytest.approx(1.0)
    assert heatmap_target[2, 3] < heatmap_target[2, 2]
    assert heatmap_target[0, 9] == pytest.approx(0.0)


def test_boundary_target_builder(track_tmp_path: Path) -> None:
    """Boundary target builder should rasterize contours without filling interiors."""
    store = _create_test_store(track_tmp_path / "boundary_targets.db")

    target = BoundaryTargetBuilder(
        where='props["class"] == "tumor"',
        line_width=1,
    ).create_target(
        store=store,
        patch_bounds=(0, 0, 10, 10),
        output_shape=(10, 10),
    )

    assert target.shape == (10, 10)
    assert target.dtype == np.float32
    assert target[0, 0] == pytest.approx(1.0)
    assert target[4, 4] == pytest.approx(0.0)
    assert target[9, 9] == pytest.approx(0.0)


def test_stacked_target_builder_shared_parent_where_queries_once(
    track_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stacked parent queries should be shared by all child target channels."""
    store = _create_test_store(track_tmp_path / "stacked_targets.db")
    where = 'props["class"] == "tumor"'
    query_calls: list[dict[str, object]] = []
    original_query = store.query

    def counting_query(*args: object, **kwargs: object) -> dict[str, Annotation]:
        query_calls.append(dict(kwargs))
        return original_query(*args, **kwargs)

    monkeypatch.setattr(store, "query", counting_query)

    builder = StackedTargetBuilder(
        {
            "mask": MaskTargetBuilder(class_mapping=None),
            "boundary": BoundaryTargetBuilder(),
            "heatmap": GaussianHeatmapTargetBuilder(sigma=1.0),
        },
        where=where,
    )

    target = builder.create_target(
        store=store,
        patch_bounds=(0, 0, 10, 10),
        output_shape=(10, 10),
    )

    assert len(query_calls) == 1
    assert query_calls[0]["where"] == where
    assert target.shape == (10, 10, 3)
    assert target.dtype == np.float32
    assert target[2, 2, 0] == pytest.approx(1.0)
    assert target[2, 7, 0] == pytest.approx(0.0)
    assert target[4, 4, 1] == pytest.approx(0.0)
    assert 0.8 < float(target[..., 2].max()) <= 1.0


def test_stacked_target_builder_delegates_child_specific_filters(
    track_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stacked child query settings should still delegate to child builders."""
    store = _create_test_store(track_tmp_path / "stacked_child_filters.db")
    query_calls: list[dict[str, object]] = []
    original_query = store.query

    def counting_query(*args: object, **kwargs: object) -> dict[str, Annotation]:
        query_calls.append(dict(kwargs))
        return original_query(*args, **kwargs)

    monkeypatch.setattr(store, "query", counting_query)

    builder = StackedTargetBuilder(
        {
            "tumor": MaskTargetBuilder(
                where='props["class"] == "tumor"',
                class_mapping=None,
            ),
            "stroma": MaskTargetBuilder(
                where='props["class"] == "stroma"',
                class_mapping=None,
            ),
        }
    )

    target = builder.create_target(
        store=store,
        patch_bounds=(0, 0, 10, 10),
        output_shape=(10, 10),
    )

    assert len(query_calls) == 2
    assert [call["where"] for call in query_calls] == [
        'props["class"] == "tumor"',
        'props["class"] == "stroma"',
    ]
    assert target[2, 2, 0] == pytest.approx(1.0)
    assert target[2, 7, 0] == pytest.approx(0.0)
    assert target[2, 2, 1] == pytest.approx(0.0)
    assert target[2, 7, 1] == pytest.approx(1.0)


def test_stacked_target_builder_rejects_parent_and_child_query_settings() -> None:
    """Stacked builders should not silently combine parent and child queries."""
    builder = StackedTargetBuilder(
        {
            "mask": MaskTargetBuilder(
                where='props["class"] == "tumor"',
                class_mapping=None,
            ),
        },
        where='props["class"] == "stroma"',
    )

    with pytest.raises(ValueError, match="cannot combine parent query settings.*mask"):
        builder.create_target(
            store=SQLiteStore(":memory:"),
            patch_bounds=(0, 0, 10, 10),
            output_shape=(10, 10),
        )


def test_stacked_target_builder_forwards_parent_query_options(
    track_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parent stacked query options should be forwarded to the store."""
    store = _create_test_store(track_tmp_path / "stacked_parent_query_options.db")
    query_calls: list[dict[str, object]] = []
    original_query = store.query

    def counting_query(*args: object, **kwargs: object) -> dict[str, Annotation]:
        query_calls.append(dict(kwargs))
        return original_query(*args, **kwargs)

    monkeypatch.setattr(store, "query", counting_query)

    area_filtered = StackedTargetBuilder(
        {
            "mask": MaskTargetBuilder(
                class_mapping={"tumor": 1, "stroma": 2},
                class_property="class",
            ),
            "boundary": BoundaryTargetBuilder(),
        },
        min_area=30,
    ).create_target(
        store=store,
        patch_bounds=(0, 0, 10, 10),
        output_shape=(10, 10),
    )

    assert len(query_calls) == 1
    assert query_calls[0]["min_area"] == pytest.approx(30.0)
    assert query_calls[0]["geometry_predicate"] == "intersects"
    assert query_calls[0]["distance"] == pytest.approx(0.0)
    assert area_filtered[2, 2, 0] == pytest.approx(1.0)
    assert area_filtered[2, 7, 0] == pytest.approx(0.0)

    point_store = SQLiteStore(track_tmp_path / "stacked_parent_distance_options.db")
    point_store.append(Annotation(Point(5, 5), properties={}), key="near")
    point_store.append(Annotation(Point(9, 9), properties={}), key="far")
    distance_query_calls: list[dict[str, object]] = []
    original_distance_query = point_store.query

    def counting_distance_query(
        *args: object,
        **kwargs: object,
    ) -> dict[str, Annotation]:
        distance_query_calls.append(dict(kwargs))
        return original_distance_query(*args, **kwargs)

    monkeypatch.setattr(point_store, "query", counting_distance_query)

    distance_filtered = StackedTargetBuilder(
        {"disk": BinaryDiskTargetBuilder(radius=0)},
        geometry_predicate="centers_within_k",
        distance=2.0,
    ).create_target(
        store=point_store,
        patch_bounds=(0, 0, 10, 10),
        output_shape=(10, 10),
    )

    assert len(distance_query_calls) == 1
    assert distance_query_calls[0]["geometry_predicate"] == "centers_within_k"
    assert distance_query_calls[0]["distance"] == pytest.approx(2.0)
    assert distance_filtered[5, 5, 0] == pytest.approx(1.0)
    assert distance_filtered[9, 9, 0] == pytest.approx(0.0)


def test_stacked_target_builder_reports_per_channel_spatial_specs() -> None:
    """Stacked target builders should preserve mixed interpolation metadata."""
    builder = StackedTargetBuilder(
        {
            "mask": MaskTargetBuilder(class_mapping=None),
            "boundary": BoundaryTargetBuilder(),
            "heatmap": GaussianHeatmapTargetBuilder(sigma=1.0),
        }
    )

    assert builder.spatial_target_spec == StackedSpatialTargetSpec(
        (
            ("mask", "mask"),
            ("boundary", "mask"),
            ("heatmap", "image"),
        )
    )


def test_stacked_target_builder_honours_child_geometry_predicates(
    track_tmp_path: Path,
) -> None:
    """Stacked child builders should use their own geometry predicates."""
    store = SQLiteStore(track_tmp_path / "stacked_geometry_predicates.db")
    store.append(
        Annotation(
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            properties={"class": "inside"},
        ),
        key="inside",
    )
    store.append(
        Annotation(
            Polygon([(8, 1), (12, 1), (12, 3), (8, 3)]),
            properties={"class": "partial"},
        ),
        key="partial",
    )

    builder = StackedTargetBuilder(
        {
            "contained": MaskTargetBuilder(
                class_mapping=None,
                geometry_predicate="contains",
            ),
            "intersecting": MaskTargetBuilder(
                class_mapping=None,
                geometry_predicate="intersects",
            ),
        }
    )

    target = builder.create_target(
        store=store,
        patch_bounds=(0, 0, 10, 10),
        output_shape=(10, 10),
    )

    assert target[2, 2, 0] == pytest.approx(1.0)
    assert target[2, 9, 0] == pytest.approx(0.0)
    assert target[2, 2, 1] == pytest.approx(1.0)
    assert target[2, 9, 1] == pytest.approx(1.0)


def test_stacked_target_builder_honours_class_specific_child_filters(
    track_tmp_path: Path,
) -> None:
    """Class-specific stacked builders should match the KongNet target pattern."""
    store = _create_test_store(track_tmp_path / "stacked_class_filters.db")
    builder = CompositeTargetBuilder(
        {
            label: StackedTargetBuilder(
                {
                    "mask": MaskTargetBuilder(
                        where=f'props["class"] == "{label}"',
                        class_mapping=None,
                        default_label=0,
                    ),
                    "boundary": BoundaryTargetBuilder(
                        where=f'props["class"] == "{label}"',
                        line_width=1,
                    ),
                    "centroid": GaussianHeatmapTargetBuilder(
                        where=f'props["class"] == "{label}"',
                        sigma=1.0,
                    ),
                }
            )
            for label in ("tumor", "stroma")
        }
    )

    target = builder.create_target(
        store=store,
        patch_bounds=(0, 0, 10, 10),
        output_shape=(10, 10),
    )

    assert target["tumor"].shape == (10, 10, 3)
    assert target["stroma"].shape == (10, 10, 3)
    assert target["tumor"][2, 2, 0] == pytest.approx(1.0)
    assert target["tumor"][2, 7, 0] == pytest.approx(0.0)
    assert target["stroma"][2, 2, 0] == pytest.approx(0.0)
    assert target["stroma"][2, 7, 0] == pytest.approx(1.0)
    assert target["tumor"][0, 0, 1] == pytest.approx(1.0)
    assert target["stroma"][0, 0, 1] == pytest.approx(0.0)
    assert 0.8 < float(target["tumor"][..., 2].max()) <= 1.0
    assert 0.7 < float(target["stroma"][..., 2].max()) <= 1.0


def test_composite_target_builder(track_tmp_path: Path) -> None:
    """Composite builders should return structured nested target dictionaries."""
    store = _create_test_store(track_tmp_path / "composite_targets.db")

    builder = CompositeTargetBuilder(
        {
            "mask": MaskTargetBuilder(
                class_mapping={"tumor": 1, "stroma": 2},
                class_property="class",
                default_label=0,
            ),
            "signals": CompositeTargetBuilder(
                {
                    "presence": PresenceTargetBuilder(
                        where='props["class"] == "tumor"',
                        min_fraction=0.1,
                    ),
                    "disk": BinaryDiskTargetBuilder(
                        where='props["class"] == "tumor"',
                        radius=1,
                    ),
                }
            ),
        }
    )

    target = builder.create_target(
        store=store,
        patch_bounds=(0, 0, 10, 10),
        output_shape=(10, 10),
    )

    assert set(target.keys()) == {"mask", "signals"}
    assert target["mask"].shape == (10, 10)
    assert int(target["signals"]["presence"]) == 1
    assert target["signals"]["disk"].shape == (10, 10)


def test_target_builders_validate_fractional_thresholds() -> None:
    """Coverage-based builders should reject invalid min_fraction values."""
    with pytest.raises(ValueError, match="`min_fraction` must be in the interval"):
        _ = PresenceTargetBuilder(min_fraction=-0.1)

    with pytest.raises(ValueError, match="`min_fraction` must be in the interval"):
        _ = CoverageClassTargetBuilder(
            class_mapping={"tumor": 1},
            min_fraction=1.1,
        )

    with pytest.raises(ValueError, match="`min_fraction` must be in the interval"):
        _ = MultiLabelTargetBuilder(
            class_mapping={"tumor": 0},
            min_fraction=2.0,
        )


def test_target_builders_validate_other_constructor_arguments() -> None:
    """Target builders should reject invalid overlap and rasterization arguments."""
    with pytest.raises(ValueError, match="`line_width` must be a positive integer"):
        _ = MaskTargetBuilder(line_width=0)

    with pytest.raises(ValueError, match="`point_radius` must be a positive integer"):
        _ = MaskTargetBuilder(point_radius=0)

    with pytest.raises(ValueError, match="`overlap_method` must be either"):
        _ = PresenceTargetBuilder(overlap_method="unknown")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="`radius` must be a non-negative integer"):
        _ = BinaryDiskTargetBuilder(radius=-1)

    with pytest.raises(ValueError, match="`sigma` must be positive"):
        _ = GaussianHeatmapTargetBuilder(sigma=0.0)

    with pytest.raises(ValueError, match="`builders` must contain at least one"):
        _ = CompositeTargetBuilder({})

    with pytest.raises(ValueError, match="`line_width` must be a positive integer"):
        _ = BoundaryTargetBuilder(line_width=0)

    with pytest.raises(ValueError, match="`point_radius` must be a positive integer"):
        _ = BoundaryTargetBuilder(point_radius=0)

    with pytest.raises(ValueError, match="`builders` must contain at least one"):
        _ = StackedTargetBuilder({})


def test_patch_annotation_dataset_with_bounds_and_where(track_tmp_path: Path) -> None:
    """Patch+annotation dataset should generate per-patch targets from one store."""
    store_path = track_tmp_path / "dataset_targets.db"
    store = SQLiteStore(store_path)
    store.append(
        Annotation(
            Polygon([(1, 1), (5, 1), (5, 5), (1, 5)]),
            properties={"class": "tumor"},
        ),
        key="tumor_patch_0",
    )
    store.append(
        Annotation(
            Polygon([(11, 1), (15, 1), (15, 5), (11, 5)]),
            properties={"class": "stroma"},
        ),
        key="stroma_patch_1",
    )

    patch_0 = np.zeros((10, 10, 3), dtype=np.uint8)
    patch_1 = np.zeros((10, 10, 3), dtype=np.uint8)

    builder = PresenceTargetBuilder(
        where='props["class"] == "tumor"',
        min_fraction=0.1,
        positive_label=1,
        negative_label=0,
    )

    dataset = PatchAnnotationDataset(
        patch_inputs=[patch_0, patch_1],
        annotation_stores=store_path,
        target_builder=builder,
        patch_bounds=[(0, 0, 10, 10), (10, 0, 20, 10)],
    )

    sample_0 = dataset[0]
    sample_1 = dataset[1]

    assert sample_0["image"].shape == (3, 10, 10)
    assert int(sample_0["target"].item()) == 1
    assert int(sample_1["target"].item()) == 0


def test_patch_annotation_dataset_validation_errors(track_tmp_path: Path) -> None:
    """Patch annotation dataset should fail for invalid input combinations."""
    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    builder = PresenceTargetBuilder()

    with pytest.raises(ValueError, match="same length as `patch_inputs`"):
        _ = PatchAnnotationDataset(
            patch_inputs=[patch, patch],
            annotation_stores=[track_tmp_path / "a.db"],
            target_builder=builder,
        )

    with pytest.raises(ValueError, match="same length as `patch_inputs`"):
        _ = PatchAnnotationDataset(
            patch_inputs=[patch, patch],
            annotation_stores=track_tmp_path / "a.db",
            target_builder=builder,
            patch_bounds=[(0, 0, 10, 10)],
        )

    dataset = PatchAnnotationDataset(
        patch_inputs=[patch],
        annotation_stores=track_tmp_path / "missing.db",
        target_builder=builder,
    )
    with pytest.raises(ValueError, match="does not exist"):
        _ = dataset[0]


def test_patch_annotation_dataset_survives_pickle_roundtrip(
    track_tmp_path: Path,
) -> None:
    """Path-backed annotation stores should be reopened after pickling."""
    store_path = track_tmp_path / "pickle_patch_store.db"
    store = SQLiteStore(store_path)
    store.append(
        Annotation(
            Polygon([(0, 0), (8, 0), (8, 8), (0, 8)]),
            properties={"class": "tumor"},
        ),
        key="tumor",
    )

    dataset = PatchAnnotationDataset(
        patch_inputs=[np.zeros((8, 8, 3), dtype=np.uint8)],
        annotation_stores=store_path,
        target_builder=PresenceTargetBuilder(
            where='props["class"] == "tumor"',
            min_fraction=0.25,
            positive_label=1,
            negative_label=0,
        ),
    )
    _ = dataset[0]

    restored_dataset = pickle.loads(pickle.dumps(dataset))
    sample = restored_dataset[0]

    assert int(sample["target"].item()) == 1


def test_patch_annotation_dataset_pair_transform_keeps_mask_targets_aligned(
    track_tmp_path: Path,
) -> None:
    """Pair transforms should keep annotation-derived masks aligned to images."""
    store_path = track_tmp_path / "pair_patch_store.db"
    _create_test_store(store_path)

    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    patch[:, :5] = 32
    patch[:, 5:] = 224

    dataset = PatchAnnotationDataset(
        patch_inputs=[patch],
        annotation_stores=store_path,
        target_builder=MaskTargetBuilder(
            class_mapping={"tumor": 1, "stroma": 2},
            class_property="class",
            default_label=0,
        ),
        pair_transform=_flip_left_right,
    )

    sample = dataset[0]
    assert sample["image"].shape == (3, 10, 10)
    assert int(sample["target"][2, 2].item()) == 2
    assert int(sample["target"][2, 7].item()) == 1


def test_patch_annotation_dataset_supports_structured_targets(
    track_tmp_path: Path,
) -> None:
    """Patch annotation datasets should tensorize nested structured targets."""
    store_path = track_tmp_path / "structured_patch_store.db"
    _create_test_store(store_path)

    builder = CompositeTargetBuilder(
        {
            "mask": MaskTargetBuilder(
                class_mapping={"tumor": 1, "stroma": 2},
                class_property="class",
                default_label=0,
            ),
            "signals": CompositeTargetBuilder(
                {
                    "presence": PresenceTargetBuilder(
                        where='props["class"] == "tumor"',
                        min_fraction=0.1,
                    ),
                    "heatmap": GaussianHeatmapTargetBuilder(
                        where='props["class"] == "tumor"',
                        sigma=1.0,
                    ),
                }
            ),
        }
    )

    dataset = PatchAnnotationDataset(
        patch_inputs=[np.zeros((10, 10, 3), dtype=np.uint8)],
        annotation_stores=store_path,
        target_builder=builder,
    )

    sample = dataset[0]

    assert isinstance(sample["target"], dict)
    assert isinstance(sample["target"]["mask"], torch.Tensor)
    assert sample["target"]["mask"].shape == (10, 10)
    assert isinstance(sample["target"]["signals"], dict)
    assert sample["target"]["signals"]["presence"].shape == ()
    assert sample["target"]["signals"]["heatmap"].shape == (10, 10)


def test_patch_annotation_dataset_channel_first_stacked_targets(
    track_tmp_path: Path,
) -> None:
    """Stacked dense targets should be tensorized to channel-first layout."""
    store_path = track_tmp_path / "stacked_patch_store.db"
    _create_test_store(store_path)

    builder = CompositeTargetBuilder(
        {
            "tumor": StackedTargetBuilder(
                {
                    "mask": MaskTargetBuilder(
                        where='props["class"] == "tumor"',
                        class_mapping=None,
                    ),
                    "boundary": BoundaryTargetBuilder(
                        where='props["class"] == "tumor"',
                    ),
                    "heatmap": GaussianHeatmapTargetBuilder(
                        where='props["class"] == "tumor"',
                        sigma=1.0,
                    ),
                }
            )
        }
    )

    dataset = PatchAnnotationDataset(
        patch_inputs=[np.zeros((10, 10, 3), dtype=np.uint8)],
        annotation_stores=store_path,
        target_builder=builder,
    )

    sample = dataset[0]

    assert sample["target"]["tumor"].shape == (3, 10, 10)
    assert sample["target"]["tumor"].dtype == torch.float32
    assert sample["target"]["tumor"][0, 2, 2].item() == pytest.approx(1.0)


def test_generate_slide_patch_coordinates_with_mask(track_tmp_path: Path) -> None:
    """Coordinate generation should respect optional mask filtering."""
    slide = np.zeros((32, 32, 3), dtype=np.uint8)
    slide_path = track_tmp_path / "slide.npy"
    np.save(slide_path, slide)

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:, :16] = 1

    coords = generate_slide_patch_coordinates(
        input_img=slide_path,
        patch_size=(16, 16),
        stride=(16, 16),
        resolution=1.0,
        units="baseline",
        input_mask=mask,
        min_mask_ratio=0.5,
    )
    assert coords.shape == (2, 4)
    assert {
        tuple(coord.tolist()) for coord in coords
    } == {
        (0, 0, 16, 16),
        (0, 16, 16, 32),
    }


def test_slide_annotation_patch_dataset(track_tmp_path: Path) -> None:
    """Slide+annotation dataset should stream patches and build targets per slide."""
    slide_0 = np.zeros((32, 32, 3), dtype=np.uint8)
    slide_1 = np.zeros((32, 32, 3), dtype=np.uint8)
    slide_0_path = track_tmp_path / "slide_0.npy"
    slide_1_path = track_tmp_path / "slide_1.npy"
    np.save(slide_0_path, slide_0)
    np.save(slide_1_path, slide_1)

    store_0_path = track_tmp_path / "store_0.db"
    store_1_path = track_tmp_path / "store_1.db"
    store_0 = SQLiteStore(store_0_path)
    store_1 = SQLiteStore(store_1_path)

    store_0.append(
        Annotation(
            Polygon([(0, 0), (16, 0), (16, 16), (0, 16)]),
            properties={"class": "tumor"},
        ),
        key="slide_0_tumor",
    )
    store_1.append(
        Annotation(
            Polygon([(16, 0), (32, 0), (32, 16), (16, 16)]),
            properties={"class": "tumor"},
        ),
        key="slide_1_tumor",
    )

    dataset = SlideAnnotationPatchDataset(
        slide_inputs=[slide_0_path, slide_1_path],
        annotation_stores=[store_0_path, store_1_path],
        target_builder=PresenceTargetBuilder(
            where='props["class"] == "tumor"',
            min_fraction=0.25,
            positive_label=1,
            negative_label=0,
        ),
        patch_size=(16, 16),
        stride=(16, 16),
        resolution=1.0,
        units="baseline",
    )

    assert len(dataset) == 8
    targets = [int(dataset[index]["target"].item()) for index in range(len(dataset))]
    assert targets[:4].count(1) == 1
    assert targets[4:].count(1) == 1

    sample = dataset[0]
    assert sample["image"].shape == (3, 16, 16)
    assert sample["bounds"].shape == (4,)
    assert sample["slide_index"].dtype == torch.long


def test_slide_annotation_patch_dataset_resolution_conversion(
    track_tmp_path: Path,
) -> None:
    """Slide bounds should be converted from resolution-space to baseline bounds."""
    slide = np.zeros((32, 32, 3), dtype=np.uint8)
    slide_path = track_tmp_path / "slide_res.npy"
    np.save(slide_path, slide)

    store_path = track_tmp_path / "store_res.db"
    store = SQLiteStore(store_path)
    store.append(
        Annotation(
            Polygon([(0, 0), (16, 0), (16, 16), (0, 16)]),
            properties={"class": "tumor"},
        ),
        key="tumor",
    )

    dataset = SlideAnnotationPatchDataset(
        slide_inputs=[slide_path],
        annotation_stores=store_path,
        target_builder=PresenceTargetBuilder(
            where='props["class"] == "tumor"',
            min_fraction=0.5,
            positive_label=1,
            negative_label=0,
        ),
        patch_size=(8, 8),
        stride=(8, 8),
        resolution=0.5,
        units="baseline",
    )
    targets = [int(dataset[index]["target"].item()) for index in range(len(dataset))]
    assert targets == [1, 0, 0, 0]


def test_slide_annotation_patch_dataset_survives_pickle_roundtrip(
    track_tmp_path: Path,
) -> None:
    """Path-backed slide datasets should reopen readers and stores after pickling."""
    slide = np.zeros((32, 32, 3), dtype=np.uint8)
    slide_path = track_tmp_path / "slide_pickle.npy"
    np.save(slide_path, slide)

    store_path = track_tmp_path / "store_pickle.db"
    store = SQLiteStore(store_path)
    store.append(
        Annotation(
            Polygon([(0, 0), (16, 0), (16, 16), (0, 16)]),
            properties={"class": "tumor"},
        ),
        key="tumor",
    )

    dataset = SlideAnnotationPatchDataset(
        slide_inputs=[slide_path],
        annotation_stores=store_path,
        target_builder=PresenceTargetBuilder(
            where='props["class"] == "tumor"',
            min_fraction=0.25,
            positive_label=1,
            negative_label=0,
        ),
        patch_size=(16, 16),
        stride=(16, 16),
        resolution=1.0,
        units="baseline",
    )
    _ = dataset[0]

    restored_dataset = pickle.loads(pickle.dumps(dataset))
    sample = restored_dataset[0]

    assert sample["image"].shape == (3, 16, 16)
    assert int(sample["target"].item()) == 1


def test_slide_annotation_patch_dataset_pair_transform_keeps_masks_aligned(
    track_tmp_path: Path,
) -> None:
    """Pair transforms should keep slide-derived masks aligned to images."""
    slide = np.zeros((16, 16, 3), dtype=np.uint8)
    slide[:, :8] = 48
    slide[:, 8:] = 196
    slide_path = track_tmp_path / "slide_pair.npy"
    np.save(slide_path, slide)

    store_path = track_tmp_path / "store_pair.db"
    store = SQLiteStore(store_path)
    store.append(
        Annotation(
            Polygon([(0, 0), (8, 0), (8, 16), (0, 16)]),
            properties={"class": "tumor"},
        ),
        key="tumor",
    )
    store.append(
        Annotation(
            Polygon([(8, 0), (16, 0), (16, 8), (8, 8)]),
            properties={"class": "stroma"},
        ),
        key="stroma",
    )

    dataset = SlideAnnotationPatchDataset(
        slide_inputs=[slide_path],
        annotation_stores=store_path,
        target_builder=MaskTargetBuilder(
            class_mapping={"tumor": 1, "stroma": 2},
            class_property="class",
            default_label=0,
        ),
        patch_size=(16, 16),
        stride=(16, 16),
        resolution=1.0,
        units="baseline",
        pair_transform=_flip_left_right,
    )

    sample = dataset[0]
    assert sample["image"].shape == (3, 16, 16)
    assert int(sample["target"][2, 2].item()) == 2
    assert int(sample["target"][10, 12].item()) == 1


def test_slide_annotation_patch_dataset_supports_structured_targets(
    track_tmp_path: Path,
) -> None:
    """Slide annotation datasets should tensorize nested structured targets."""
    slide = np.zeros((16, 16, 3), dtype=np.uint8)
    slide_path = track_tmp_path / "slide_structured.npy"
    np.save(slide_path, slide)

    store_path = track_tmp_path / "store_structured.db"
    store = SQLiteStore(store_path)
    store.append(
        Annotation(
            Polygon([(0, 0), (8, 0), (8, 8), (0, 8)]),
            properties={"class": "tumor"},
        ),
        key="tumor",
    )

    builder = CompositeTargetBuilder(
        {
            "presence": PresenceTargetBuilder(
                where='props["class"] == "tumor"',
                min_fraction=0.1,
            ),
            "disk": BinaryDiskTargetBuilder(
                where='props["class"] == "tumor"',
                radius=1,
            ),
        }
    )

    dataset = SlideAnnotationPatchDataset(
        slide_inputs=[slide_path],
        annotation_stores=store_path,
        target_builder=builder,
        patch_size=(16, 16),
        stride=(16, 16),
        resolution=1.0,
        units="baseline",
    )

    sample = dataset[0]

    assert isinstance(sample["target"], dict)
    assert sample["target"]["presence"].shape == ()
    assert sample["target"]["disk"].shape == (16, 16)


def test_slide_annotation_patch_dataset_channel_first_stacked_targets(
    track_tmp_path: Path,
) -> None:
    """Slide annotation datasets should tensorize stacked targets to CxHxW."""
    slide = np.zeros((16, 16, 3), dtype=np.uint8)
    slide_path = track_tmp_path / "slide_stacked.npy"
    np.save(slide_path, slide)

    store_path = track_tmp_path / "store_stacked.db"
    store = SQLiteStore(store_path)
    store.append(
        Annotation(
            Polygon([(0, 0), (8, 0), (8, 8), (0, 8)]),
            properties={"class": "tumor"},
        ),
        key="tumor",
    )

    builder = CompositeTargetBuilder(
        {
            "tumor": StackedTargetBuilder(
                {
                    "mask": MaskTargetBuilder(
                        where='props["class"] == "tumor"',
                        class_mapping=None,
                    ),
                    "boundary": BoundaryTargetBuilder(
                        where='props["class"] == "tumor"',
                    ),
                    "heatmap": GaussianHeatmapTargetBuilder(
                        where='props["class"] == "tumor"',
                        sigma=1.0,
                    ),
                }
            )
        }
    )

    dataset = SlideAnnotationPatchDataset(
        slide_inputs=[slide_path],
        annotation_stores=store_path,
        target_builder=builder,
        patch_size=(16, 16),
        stride=(16, 16),
        resolution=1.0,
        units="baseline",
    )

    sample = dataset[0]

    assert sample["target"]["tumor"].shape == (3, 16, 16)
    assert sample["target"]["tumor"].dtype == torch.float32


def test_slide_annotation_patch_dataset_validation_errors(
    track_tmp_path: Path,
) -> None:
    """Slide annotation dataset should fail on invalid input combinations."""
    slide = np.zeros((32, 32, 3), dtype=np.uint8)
    slide_path = track_tmp_path / "valid_slide.npy"
    np.save(slide_path, slide)

    store_path = track_tmp_path / "valid_store.db"
    SQLiteStore(store_path)

    builder = PresenceTargetBuilder()
    with pytest.raises(ValueError, match="same length as `slide_inputs`"):
        _ = SlideAnnotationPatchDataset(
            slide_inputs=[slide_path, slide_path],
            annotation_stores=[store_path],
            target_builder=builder,
            patch_size=(16, 16),
        )

    with pytest.raises(ValueError, match="same length as `slide_inputs`"):
        _ = SlideAnnotationPatchDataset(
            slide_inputs=[slide_path, slide_path],
            annotation_stores=store_path,
            target_builder=builder,
            patch_size=(16, 16),
            input_masks=[None],
        )

    with pytest.raises(FileNotFoundError, match="Input path does not exist"):
        _ = SlideAnnotationPatchDataset(
            slide_inputs=[track_tmp_path / "missing_slide.npy"],
            annotation_stores=store_path,
            target_builder=builder,
            patch_size=(16, 16),
        )

    with pytest.raises(ValueError, match="No patch coordinates"):
        _ = SlideAnnotationPatchDataset(
            slide_inputs=[slide_path],
            annotation_stores=store_path,
            target_builder=builder,
            patch_size=(16, 16),
            input_masks=np.zeros((32, 32), dtype=np.uint8),
            min_mask_ratio=0.5,
        )


def test_class_balanced_index_sampler() -> None:
    """Class-balanced sampler should upweight minority classes."""
    labels = np.array([0] * 9 + [1], dtype=np.int64)
    sampler = ClassBalancedIndexSampler(
        labels=labels,
        num_samples=1000,
        generator=torch.Generator().manual_seed(7),
    )
    sampled_labels = labels[list(iter(sampler))]

    minority_ratio = float(np.mean(sampled_labels == 1))
    assert 0.4 < minority_ratio < 0.6
    assert len(sampler) == 1000

    ignore_sampler = ClassBalancedIndexSampler(
        labels=np.array([0, 0, 1, 2], dtype=np.int64),
        num_samples=200,
        ignore_labels={2},
        generator=torch.Generator().manual_seed(3),
    )
    ignore_labels = np.array([0, 0, 1, 2], dtype=np.int64)[list(iter(ignore_sampler))]
    assert np.all(ignore_labels != 2)

    with pytest.raises(ValueError, match="cannot exceed the number of non-ignored"):
        _ = ClassBalancedIndexSampler(
            labels=np.array([0, 0, 1], dtype=np.int64),
            num_samples=3,
            ignore_labels={1},
            replacement=False,
        )
