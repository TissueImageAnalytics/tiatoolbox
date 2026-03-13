"""Tests for annotation-based training targets and datasets."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from shapely.geometry import Polygon

from tiatoolbox.annotation import Annotation, SQLiteStore
from tiatoolbox.models.training import (
    ClassBalancedIndexSampler,
    CoverageClassTargetBuilder,
    MaskTargetBuilder,
    MultiLabelTargetBuilder,
    PatchAnnotationDataset,
    PresenceTargetBuilder,
    SlideAnnotationPatchDataset,
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

    with pytest.raises(ValueError, match="does not exist"):
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
