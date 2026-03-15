"""Tests for TIAToolbox training scaffolding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from tiatoolbox.models.models_abc import load_torch_model
from tiatoolbox.models.training import (
    CheckpointConfig,
    ClassificationTask,
    PatchFolderClassificationDataset,
    PatchMaskPairDataset,
    SegmentationTask,
    Trainer,
    TrainerConfig,
    save_checkpoint,
    save_model_weights,
    stratified_split_indices,
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


def _build_dummy_loader() -> DataLoader:
    """Create a minimal dataloader for trainer unit tests."""
    images = torch.zeros((4, 3, 4, 4), dtype=torch.float32)
    targets = torch.zeros(4, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(images, targets)
    return DataLoader(dataset, batch_size=2, shuffle=False)


class WrappedStateDictModel(nn.Module):
    """Small model that expects weights wrapped under a `model` key."""

    def __init__(self) -> None:
        """Initialize the wrapped-state test model."""
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3 * 4 * 4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.linear(self.flatten(inputs))

    def load_state_dict(  # type: ignore[override]
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        strict: bool = True,
        assign: bool = False,
    ) -> nn.Module:
        """Load weights from a payload wrapped under `model`."""
        return super().load_state_dict(
            state_dict["model"],
            strict=strict,
            assign=assign,
        )


def test_patch_folder_classification_dataset(track_tmp_path: Path) -> None:
    """Folder-based dataset should load images and class labels."""
    dataset_root = _build_classification_dataset(track_tmp_path / "class_dataset")
    dataset = PatchFolderClassificationDataset(dataset_root)

    assert len(dataset) == 12
    sample = dataset[0]

    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 16, 16)
    assert sample["target"].dtype == torch.long


def test_patch_folder_classification_dataset_works_with_dataloader(
    track_tmp_path: Path,
) -> None:
    """Folder dataset should batch cleanly with a direct PyTorch DataLoader."""
    dataset_root = _build_classification_dataset(track_tmp_path / "class_dataset")
    dataset = PatchFolderClassificationDataset(dataset_root)

    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, drop_last=True)

    batch = next(iter(dataloader))
    assert dataloader.batch_size == 3
    assert dataloader.drop_last is True
    assert batch["image"].shape[0] == 3


def test_stratified_split_indices_preserves_class_balance() -> None:
    """Stratified index splits should preserve per-class proportions."""
    targets = [0] * 10 + [1] * 10 + [2] * 10

    train_indices, val_indices = stratified_split_indices(
        targets,
        val_fraction=0.2,
        seed=7,
    )

    train_targets = [targets[index] for index in train_indices]
    val_targets = [targets[index] for index in val_indices]

    assert len(train_indices) == 24
    assert len(val_indices) == 6
    assert np.bincount(train_targets).tolist() == [8, 8, 8]
    assert np.bincount(val_targets).tolist() == [2, 2, 2]


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


def test_patch_mask_pair_dataset_matches_nested_relative_paths(
    track_tmp_path: Path,
) -> None:
    """Dataset should pair nested files by relative path, not bare stem."""
    image_dir = track_tmp_path / "images"
    mask_dir = track_tmp_path / "masks"

    for subset, image_value, mask_value in (
        ("case_a", 32, 1),
        ("case_b", 224, 2),
    ):
        (image_dir / subset).mkdir(parents=True, exist_ok=True)
        (mask_dir / subset).mkdir(parents=True, exist_ok=True)
        np.save(
            image_dir / subset / "sample.npy",
            np.full((8, 8, 3), fill_value=image_value, dtype=np.uint8),
        )
        np.save(
            mask_dir / subset / "sample.npy",
            np.full((8, 8), fill_value=mask_value, dtype=np.uint8),
        )

    dataset = PatchMaskPairDataset(image_dir=image_dir, mask_dir=mask_dir)

    assert len(dataset) == 2
    assert int(dataset[0]["target"][0, 0].item()) == 1
    assert int(dataset[1]["target"][0, 0].item()) == 2


def test_patch_mask_pair_dataset_raises_on_unmatched_files(
    track_tmp_path: Path,
) -> None:
    """Dataset should reject partial image/mask directory mismatches."""
    image_dir = track_tmp_path / "images"
    mask_dir = track_tmp_path / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    np.save(image_dir / "paired.npy", np.zeros((8, 8, 3), dtype=np.uint8))
    np.save(mask_dir / "paired.npy", np.zeros((8, 8), dtype=np.uint8))
    np.save(image_dir / "image_only.npy", np.zeros((8, 8, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="Unmatched image/mask files"):
        _ = PatchMaskPairDataset(image_dir=image_dir, mask_dir=mask_dir)


def test_patch_mask_pair_dataset_raises_on_duplicate_relative_keys(
    track_tmp_path: Path,
) -> None:
    """Dataset should reject duplicate files that collapse to the same key."""
    image_dir = track_tmp_path / "images"
    mask_dir = track_tmp_path / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    np.save(image_dir / "sample.npy", np.zeros((8, 8, 3), dtype=np.uint8))
    (image_dir / "sample.png").write_bytes(b"")
    np.save(mask_dir / "sample.npy", np.zeros((8, 8), dtype=np.uint8))

    with pytest.raises(ValueError, match="Duplicate image files"):
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


def test_classification_task_validation_for_target_modes() -> None:
    """Classification tasks should enforce compatible loss/target-mode pairs."""
    task = ClassificationTask(
        loss="bce_with_logits",
        target_mode="multi_label",
    )
    assert isinstance(task, ClassificationTask)

    binary_ce_task = ClassificationTask(
        loss="cross_entropy",
        target_mode="binary",
    )
    assert isinstance(binary_ce_task, ClassificationTask)

    with pytest.raises(ValueError, match="Unsupported classification loss `auto`"):
        _ = ClassificationTask(loss="auto")

    with pytest.raises(ValueError, match="requires `loss='bce_with_logits'`"):
        _ = ClassificationTask(
            loss="cross_entropy",
            target_mode="multi_label",
        )

    with pytest.raises(
        ValueError,
        match="does not support single-label targets",
    ):
        _ = ClassificationTask(
            loss="bce_with_logits",
            target_mode="single_label",
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


def test_trainer_requires_validation_loader_for_val_monitor(
    track_tmp_path: Path,
) -> None:
    """Validation monitors should require a validation dataloader."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = _build_dummy_loader()

    with pytest.raises(ValueError, match="requires a validation loader"):
        _ = Trainer(
            model=model,
            task=ClassificationTask(),
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=None,
            config=TrainerConfig(
                max_epochs=1,
                monitor="val_loss",
                amp=False,
                output_dir=track_tmp_path / "missing_val_loader",
            ),
            checkpoint_config=CheckpointConfig(save_best=False, save_last=False),
        )


def test_trainer_val_monitor_updates_only_on_validation_epochs(
    track_tmp_path: Path,
) -> None:
    """Validation monitors should ignore training-only epochs."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = _build_dummy_loader()
    val_loader = _build_dummy_loader()

    trainer = Trainer(
        model=model,
        task=ClassificationTask(),
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TrainerConfig(
            max_epochs=3,
            val_interval=2,
            monitor="val_loss",
            early_stopping_patience=0,
            amp=False,
            output_dir=track_tmp_path / "val_monitor",
        ),
        checkpoint_config=CheckpointConfig(save_best=False, save_last=False),
    )

    epoch_metrics = iter(
        [
            {"loss": 3.0, "accuracy": 0.25, "f1": 0.25},
            {"loss": 2.0, "accuracy": 0.50, "f1": 0.50},
            {"loss": 5.0, "accuracy": 0.10, "f1": 0.10},
            {"loss": 1.0, "accuracy": 0.75, "f1": 0.75},
        ]
    )

    def fake_run_epoch(
        loader: DataLoader,
        *,
        training: bool,
    ) -> dict[str, float]:
        del loader, training
        return next(epoch_metrics)

    trainer._run_epoch = fake_run_epoch  # type: ignore[method-assign]
    history = trainer.fit()

    assert len(history) == 3
    assert "val_loss" not in history[0]
    assert history[1]["val_loss"] == pytest.approx(5.0)
    assert "val_loss" not in history[2]
    assert trainer.best_epoch == 2
    assert trainer.best_monitor_value == pytest.approx(5.0)


def test_reduce_on_plateau_steps_only_when_monitor_is_available(
    track_tmp_path: Path,
) -> None:
    """ReduceLROnPlateau should only step on epochs with an available monitor."""

    class TrackingReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
        """ReduceLROnPlateau that records each metric it sees."""

        def __init__(self, optimizer: torch.optim.Optimizer) -> None:
            """Initialize tracking scheduler."""
            super().__init__(optimizer=optimizer, mode="min")
            self.metrics_seen: list[float] = []

        def step(self, metrics: float, epoch: int | None = None) -> None:
            """Record the metric before stepping the scheduler."""
            self.metrics_seen.append(float(metrics))
            super().step(metrics, epoch=epoch)

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = TrackingReduceLROnPlateau(optimizer)
    train_loader = _build_dummy_loader()
    val_loader = _build_dummy_loader()

    trainer = Trainer(
        model=model,
        task=ClassificationTask(),
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TrainerConfig(
            max_epochs=3,
            val_interval=2,
            monitor="val_loss",
            amp=False,
            output_dir=track_tmp_path / "plateau_monitor",
        ),
        checkpoint_config=CheckpointConfig(save_best=False, save_last=False),
    )

    epoch_metrics = iter(
        [
            {"loss": 3.0, "accuracy": 0.25, "f1": 0.25},
            {"loss": 2.0, "accuracy": 0.50, "f1": 0.50},
            {"loss": 4.0, "accuracy": 0.40, "f1": 0.40},
            {"loss": 1.0, "accuracy": 0.75, "f1": 0.75},
        ]
    )

    def fake_run_epoch(
        loader: DataLoader,
        *,
        training: bool,
    ) -> dict[str, float]:
        del loader, training
        return next(epoch_metrics)

    trainer._run_epoch = fake_run_epoch  # type: ignore[method-assign]
    _ = trainer.fit()

    assert scheduler.metrics_seen == [4.0]


def test_save_model_weights_supports_wrapped_load_state_dict_models(
    track_tmp_path: Path,
) -> None:
    """Exported weights should load through TIAToolbox helpers for wrapped models."""
    model = WrappedStateDictModel()
    with torch.no_grad():
        model.linear.weight.fill_(1.5)
        model.linear.bias.fill_(-0.25)

    weights_path = track_tmp_path / "wrapped_weights.pth"
    save_model_weights(model, weights_path)

    reloaded_model = WrappedStateDictModel()
    load_torch_model(reloaded_model, weights_path)

    assert torch.allclose(reloaded_model.linear.weight, model.linear.weight)
    assert torch.allclose(reloaded_model.linear.bias, model.linear.bias)


def test_trainer_resume_supports_wrapped_load_state_dict_models(
    track_tmp_path: Path,
) -> None:
    """Trainer resume should restore wrapped-load models from training checkpoints."""
    model = WrappedStateDictModel()
    with torch.no_grad():
        model.linear.weight.fill_(2.0)
        model.linear.bias.fill_(0.5)

    trainer = Trainer(
        model=model,
        task=ClassificationTask(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        train_loader=_build_dummy_loader(),
        val_loader=_build_dummy_loader(),
        config=TrainerConfig(
            max_epochs=1,
            amp=False,
            output_dir=track_tmp_path / "wrapped_resume_source",
        ),
        checkpoint_config=CheckpointConfig(save_best=False, save_last=False),
    )

    checkpoint_path = track_tmp_path / "wrapped_resume.ckpt"
    save_checkpoint(trainer._build_checkpoint_state(epoch=1), checkpoint_path)

    resumed_model = WrappedStateDictModel()
    resumed_trainer = Trainer(
        model=resumed_model,
        task=ClassificationTask(),
        optimizer=torch.optim.AdamW(resumed_model.parameters(), lr=1e-3),
        train_loader=_build_dummy_loader(),
        val_loader=_build_dummy_loader(),
        config=TrainerConfig(
            max_epochs=1,
            amp=False,
            output_dir=track_tmp_path / "wrapped_resume_target",
        ),
        checkpoint_config=CheckpointConfig(save_best=False, save_last=False),
    )

    start_epoch = resumed_trainer._resume_from_checkpoint(checkpoint_path)

    assert start_epoch == 1
    assert torch.allclose(resumed_model.linear.weight, model.linear.weight)
    assert torch.allclose(resumed_model.linear.bias, model.linear.bias)


def test_binary_segmentation_task_bce_loss_respects_ignore_index() -> None:
    """Binary segmentation BCE loss should ignore masked pixels."""
    task = SegmentationTask(loss="bce_with_logits", ignore_index=-100)
    logits = torch.tensor([[[[6.0, -6.0], [4.0, -4.0]]]], dtype=torch.float32)
    targets = torch.tensor([[[1, 0], [-100, 1]]], dtype=torch.long)

    loss = task.compute_loss(logits, targets)

    expected_losses = F.binary_cross_entropy_with_logits(
        logits,
        torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32),
        reduction="none",
    )
    expected = expected_losses[torch.tensor([[[[True, True], [False, True]]]])].mean()

    assert loss.item() == pytest.approx(expected.item())


def test_binary_segmentation_metrics_ignore_masked_pixels() -> None:
    """Binary segmentation metrics should ignore pixels with ignore_index."""
    task = SegmentationTask(loss="bce_with_logits", ignore_index=-100)
    logits = torch.tensor([[[[-6.0, 6.0], [-6.0, -6.0]]]], dtype=torch.float32)
    targets = torch.tensor([[[0, -100], [0, 0]]], dtype=torch.long)

    metrics = task.compute_metrics(logits, targets)

    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["iou"] == pytest.approx(1.0)


def test_binary_segmentation_task_handles_all_ignored_pixels() -> None:
    """Binary segmentation should return zero loss and metrics when all pixels are ignored."""
    task = SegmentationTask(loss="bce_with_logits", ignore_index=-100)
    logits = torch.tensor([[[[2.0, -2.0], [1.0, -1.0]]]], dtype=torch.float32)
    targets = torch.full((1, 2, 2), fill_value=-100, dtype=torch.long)

    loss = task.compute_loss(logits, targets)
    metrics = task.compute_metrics(logits, targets)

    assert loss.item() == pytest.approx(0.0)
    assert metrics["dice"] == pytest.approx(0.0)
    assert metrics["iou"] == pytest.approx(0.0)
