"""Tests for annotation-based training targets and datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from tiatoolbox.annotation import Annotation, SQLiteStore
from tiatoolbox.models.training import (
    CoverageClassTargetBuilder,
    MaskTargetBuilder,
    MultiLabelTargetBuilder,
    PatchAnnotationDataset,
    PresenceTargetBuilder,
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
