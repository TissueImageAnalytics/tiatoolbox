"""Exercise training notebook recipes without downloading data or training models."""

from __future__ import annotations

import json
import pickle
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import Subset

from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.models import training
from tiatoolbox.models.architecture.kongnet import KongNet
from tiatoolbox.models.dataset import predefined_preproc_func
from tiatoolbox.utils.misc import process_contours


def _cell_source(number: int, marker: str) -> str:
    """Select a recipe by its source text rather than a mutable cell index."""
    path = next((Path(__file__).parents[2] / "examples").glob(f"{number}-*.ipynb"))
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    )


def _run_recipe(source: str, namespace: dict) -> None:
    """Execute only selected, trusted repository cells with synthetic inputs."""
    exec(compile(source, "<training example>", "exec"), namespace)  # noqa: S102


def test_classification_example_keeps_validation_unaugmented(tmp_path: Path) -> None:
    """The example's split must not share the augmented training dataset."""
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    for label in ("a", "b"):
        folder = tmp_path / label
        folder.mkdir()
        for index in range(5):
            Image.fromarray(image).save(folder / f"{index}.png")
    namespace = vars(training) | {
        "dataset_root": tmp_path,
        "kather_class_to_idx": {"a": 0, "b": 1},
        "Subset": Subset,
        "predefined_preproc_func": predefined_preproc_func,
    }
    _run_recipe(_cell_source(13, "augmented ="), namespace)
    train, val = namespace["train_dataset"], namespace["val_dataset"]
    assert train.dataset is not val.dataset
    assert train.dataset.samples is val.dataset.samples
    assert train.dataset.pair_transform is not None
    assert val.dataset.pair_transform is None
    assert train.dataset.preproc == val.dataset.preproc
    expected = val[0]["image"]
    val.dataset.set_epoch(3, 42)
    torch.testing.assert_close(val[0]["image"], expected)


def test_tissue_example_preserves_holes_and_closes_store(tmp_path: Path) -> None:
    """Polygon conversion retains holes/islands and permits reruns on Windows."""
    mask = np.zeros((32, 32), np.uint8)
    mask[3:29, 3:29] = 1
    mask[10:22, 10:22] = 0
    mask[14:18, 14:18] = 1
    namespace = {
        "work_dir": tmp_path,
        "actual_mask": mask,
        "reader": SimpleNamespace(info=SimpleNamespace(slide_dimensions=(32, 32))),
        "np": np,
        "cv2": cv2,
        "SQLiteStore": SQLiteStore,
        "closing": closing,
        "process_contours": process_contours,
    }
    source = _cell_source(14, "store_path =")
    _run_recipe(source, namespace)
    _run_recipe(source, namespace)
    with closing(SQLiteStore(namespace["store_path"])) as store:
        target = training.MaskTargetBuilder().create_target(
            store=store, patch_bounds=(0, 0, 32, 32), output_shape=(32, 32)
        )
    assert target[5, 5] == 1
    assert target[11, 11] == 0
    assert target[16, 16] == 1


@pytest.mark.parametrize("num_samples", [2, 10, 2001])
def test_kongnet_example_splits_every_pair(
    monkeypatch: pytest.MonkeyPatch, num_samples: int
) -> None:
    """The demo splits all pairs into populated, disjoint train/val subsets."""
    monkeypatch.setattr(
        training.PatchAnnotationDataset,
        "from_dirs",
        lambda **_kwargs: range(num_samples),
    )
    namespace = vars(training) | {
        "images_dir": Path("images"),
        "stores_dir": Path("stores"),
        "KongNet": KongNet,
    }
    _run_recipe(_cell_source(15, "split_indices(len(dataset)"), namespace)
    train, val = namespace["train_indices"], namespace["val_indices"]
    assert len(train) > 0
    assert len(val) > 0
    assert not set(train) & set(val)
    assert sorted([*train, *val]) == list(range(num_samples))


def test_kongnet_example_augmentation_replays_seeds_in_workers() -> None:
    """The example's preset pair transform replays seeds after worker pickling."""
    preset = training.get_annotation_augmentation("heavy", spatial_target_spec="mask")
    assert "preproc=KongNet.preproc" in _cell_source(15, "from_dirs(")
    training_split = _cell_source(15, "with_transforms(")
    assert "with_transforms(pair_transform=train_augmentation)" in training_split
    # Round-trip only this locally constructed transform, as a worker would.
    worker_preset = pickle.loads(pickle.dumps(preset))  # noqa: S301
    image = np.random.default_rng(4).integers(0, 200, (16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.int64)
    mask[:5, :3] = 1
    changed = False
    for seed in range(8):
        preset.set_random_seed(seed)
        expected = preset(image, mask)
        worker_preset.set_random_seed(seed)
        actual = worker_preset(image, mask)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        changed |= not np.array_equal(actual[0], image)
    assert changed


def test_kongnet_example_counts_confidence_valued_peaks() -> None:
    """Counting respects the existing KongNet postproc confidence-map contract."""
    probabilities = np.zeros((32, 32, 1), np.float32)
    probabilities[8, 8, 0] = probabilities[24, 24, 0] = 0.6
    peaks = KongNet.postproc(
        SimpleNamespace(min_distance=2, threshold_abs=0.2, threshold_rel=None),
        probabilities,
    )
    namespace = {
        "np": np,
        "pd": pd,
        "Path": Path,
        "example_image_paths": [Path("patch.png")],
        "ordered_head_keys": ["nuclei"],
        "pred_peak_maps": peaks[None],
    }
    _run_recipe(_cell_source(15, "peak_count_rows ="), namespace)
    assert namespace["peak_count_rows"] == [{"patch": "patch", "nuclei": 2}]
