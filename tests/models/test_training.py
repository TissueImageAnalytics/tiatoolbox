"""Tests for TIAToolbox training scaffolding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from tiatoolbox.models.training import (
    ClassificationTask,
    PatchFolderClassificationDataset,
    PatchMaskPairDataset,
    SegmentationTask,
    TaskConfig,
    Trainer,
    TrainerConfig,
    create_task,
)


def _build_classification_dataset(root_dir: Path) -> Path:
    """Create a tiny, linearly separable folder classification dataset."""
    class_names = ["class_0", "class_1"]
    for class_index, class_name in enumerate(class_names):
        class_dir = root_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        value = 0 if class_index == 0 else 255
        for sample_index in range(6):
            image = np.full((16, 16, 3), fill_value=value, dtype=np.uint8)
            np.save(class_dir / f"sample_{sample_index}.npy", image)
    return root_dir


def _build_patch_mask_dataset(image_dir: Path, mask_dir: Path) -> tuple[Path, Path]:
    """Create a tiny patch/mask pair segmentation dataset."""
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for sample_index in range(8):
        mask_value = sample_index % 2
        image_value = 0 if mask_value == 0 else 255

        image = np.full((16, 16, 3), fill_value=image_value, dtype=np.uint8)
        mask = np.full((16, 16), fill_value=mask_value, dtype=np.uint8)

        np.save(image_dir / f"sample_{sample_index}.npy", image)
        np.save(mask_dir / f"sample_{sample_index}.npy", mask)

    return image_dir, mask_dir


def test_patch_folder_classification_dataset(track_tmp_path: Path) -> None:
    """Folder-based dataset should load images and class labels."""
    dataset_root = _build_classification_dataset(track_tmp_path / "class_dataset")
    dataset = PatchFolderClassificationDataset(dataset_root)

    assert len(dataset) == 12
    sample = dataset[0]

    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 16, 16)
    assert sample["target"].dtype == torch.long


def test_patch_mask_pair_dataset(track_tmp_path: Path) -> None:
    """Patch/mask pair dataset should return aligned image and mask tensors."""
    image_dir, mask_dir = _build_patch_mask_dataset(
        track_tmp_path / "images",
        track_tmp_path / "masks",
    )

    dataset = PatchMaskPairDataset(image_dir=image_dir, mask_dir=mask_dir)
    assert len(dataset) == 8

    sample = dataset[0]
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["target"], torch.Tensor)
    assert sample["image"].shape == (3, 16, 16)
    assert sample["target"].shape == (16, 16)
    assert sample["target"].dtype == torch.long


def test_patch_mask_pair_dataset_raises_when_no_pairs(track_tmp_path: Path) -> None:
    """Dataset should fail if no matching image/mask stems are found."""
    image_dir = track_tmp_path / "images"
    mask_dir = track_tmp_path / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    np.save(image_dir / "image_only.npy", np.zeros((8, 8, 3), dtype=np.uint8))
    np.save(mask_dir / "mask_only.npy", np.zeros((8, 8), dtype=np.uint8))

    with pytest.raises(ValueError, match="No image/mask pairs"):
        _ = PatchMaskPairDataset(image_dir=image_dir, mask_dir=mask_dir)


def test_binary_bce_classification_task_supports_vector_and_column_logits() -> None:
    """Binary BCE classification should support `(N,)` and `(N, 1)` logits."""
    targets = torch.tensor([1, 0, 1, 0], dtype=torch.long)

    vector_task = ClassificationTask(
        loss="bce_with_logits",
        target_mode="binary",
    )
    vector_logits = torch.tensor([6.0, -6.0, 6.0, -6.0], dtype=torch.float32)
    vector_loss = vector_task.compute_loss(vector_logits, targets)
    vector_metrics = vector_task.compute_metrics(vector_logits, targets)

    assert torch.isfinite(vector_loss)
    assert vector_metrics["accuracy"] == pytest.approx(1.0)
    assert vector_metrics["f1"] == pytest.approx(1.0)

    column_logits = vector_logits.unsqueeze(1)
    column_loss = vector_task.compute_loss(column_logits, targets)
    column_metrics = vector_task.compute_metrics(column_logits, targets)

    assert torch.isfinite(column_loss)
    assert column_metrics["accuracy"] == pytest.approx(1.0)
    assert column_metrics["f1"] == pytest.approx(1.0)


def test_multilabel_bce_classification_task_supports_multilabel_targets() -> None:
    """Multi-label BCE classification should compute loss and metrics end-to-end."""
    task = ClassificationTask(
        loss="bce_with_logits",
        target_mode="multi_label",
    )
    logits = torch.tensor(
        [[6.0, -6.0, 6.0], [-6.0, 6.0, -6.0]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.long)

    loss = task.compute_loss(logits, targets)
    metrics = task.compute_metrics(logits, targets)

    assert torch.isfinite(loss)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_classification_task_metrics_respect_ignore_index() -> None:
    """Classification metrics should ignore masked targets."""
    single_label_task = ClassificationTask(loss="cross_entropy", ignore_index=-100)
    single_label_logits = torch.tensor(
        [[6.0, -6.0], [6.0, -6.0], [-6.0, 6.0]],
        dtype=torch.float32,
    )
    single_label_targets = torch.tensor([0, -100, 1], dtype=torch.long)
    single_label_metrics = single_label_task.compute_metrics(
        single_label_logits,
        single_label_targets,
    )

    assert single_label_metrics["accuracy"] == pytest.approx(1.0)
    assert single_label_metrics["f1"] == pytest.approx(1.0)

    multilabel_task = ClassificationTask(
        loss="bce_with_logits",
        target_mode="multi_label",
        ignore_index=-100,
    )
    multilabel_logits = torch.tensor(
        [[6.0, -6.0], [-6.0, 6.0]],
        dtype=torch.float32,
    )
    multilabel_targets = torch.tensor(
        [[1, -100], [-100, 1]],
        dtype=torch.long,
    )
    multilabel_metrics = multilabel_task.compute_metrics(
        multilabel_logits,
        multilabel_targets,
    )

    assert multilabel_metrics["accuracy"] == pytest.approx(1.0)
    assert multilabel_metrics["f1"] == pytest.approx(1.0)


def test_task_config_validation_for_classification_target_modes() -> None:
    """Task config should reject incompatible classification settings."""
    task = create_task(
        TaskConfig(
            task_type="classification",
            loss="bce_with_logits",
            target_mode="multi_label",
        )
    )
    assert isinstance(task, ClassificationTask)

    with pytest.raises(ValueError, match="requires an explicit loss"):
        _ = TaskConfig(task_type="classification", loss="auto")

    with pytest.raises(ValueError, match="requires `loss='bce_with_logits'`"):
        _ = TaskConfig(
            task_type="classification",
            loss="cross_entropy",
            target_mode="multi_label",
        )

    with pytest.raises(
        ValueError,
        match="does not support `target_mode='single_label'`",
    ):
        _ = TaskConfig(
            task_type="classification",
            loss="bce_with_logits",
            target_mode="single_label",
        )

    with pytest.raises(ValueError, match="only supported for classification tasks"):
        _ = TaskConfig(
            task_type="segmentation",
            loss="cross_entropy",
            target_mode="binary",
        )


def test_classification_trainer_and_resume(track_tmp_path: Path) -> None:
    """Classification trainer should train, checkpoint, and resume."""
    dataset_root = _build_classification_dataset(track_tmp_path / "class_dataset")
    dataset = PatchFolderClassificationDataset(dataset_root)

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    output_dir = track_tmp_path / "classification_run"
    trainer = Trainer(
        model=model,
        task=ClassificationTask(),
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TrainerConfig(
            max_epochs=1,
            output_dir=output_dir,
            amp=False,
            seed=42,
            log_every_n_steps=0,
        ),
    )

    history = trainer.fit()
    assert history[0]["train_loss"] > 0
    assert (output_dir / "last.ckpt").exists()
    assert (output_dir / "best_model_weights.pth").exists()

    resumed_model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 2),
    )
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-2)

    resumed_trainer = Trainer(
        model=resumed_model,
        task=ClassificationTask(),
        optimizer=resumed_optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TrainerConfig(
            max_epochs=2,
            output_dir=output_dir,
            amp=False,
            seed=42,
            log_every_n_steps=0,
        ),
    )
    resumed_history = resumed_trainer.fit(resume_from=output_dir / "last.ckpt")

    assert resumed_history[-1]["epoch"] == 2.0


def test_segmentation_trainer(track_tmp_path: Path) -> None:
    """Segmentation trainer should run on patch/mask pairs and reduce loss."""
    image_dir, mask_dir = _build_patch_mask_dataset(
        track_tmp_path / "seg_images",
        track_tmp_path / "seg_masks",
    )
    dataset = PatchMaskPairDataset(image_dir=image_dir, mask_dir=mask_dir)

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 2, kernel_size=1),
    )

    trainer = Trainer(
        model=model,
        task=SegmentationTask(loss="cross_entropy"),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-2),
        train_loader=train_loader,
        val_loader=val_loader,
        config=TrainerConfig(
            max_epochs=3,
            output_dir=track_tmp_path / "segmentation_run",
            amp=False,
            seed=42,
            log_every_n_steps=0,
        ),
    )

    history = trainer.fit()
    assert history[-1]["train_loss"] <= history[0]["train_loss"]
    assert "train_dice" in history[-1]
    assert (track_tmp_path / "segmentation_run" / "last.ckpt").exists()
