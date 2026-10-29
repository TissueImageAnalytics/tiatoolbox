"""Tests for TIAToolbox training scaffolding."""

from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import tiatoolbox.models.training.augmentations as training_augmentations
from tiatoolbox.models.architecture.kongnet import KongNet
from tiatoolbox.models.architecture.vanilla import CNNModel
from tiatoolbox.models.dataset import predefined_preproc_func
from tiatoolbox.models.engine.io_config import IOPatchPredictorConfig, IOSegmentorConfig
from tiatoolbox.models.models_abc import (
    ModelABC,
    load_model_state_dict,
    load_torch_model,
)
from tiatoolbox.models.training import (
    BinaryDiskTargetBuilder,
    CheckpointConfig,
    ClassificationTask,
    CompositeTargetBuilder,
    DenseHeadSpec,
    GaussianHeatmapTargetBuilder,
    MaskTargetBuilder,
    ModelExport,
    PatchFolderClassificationDataset,
    PatchMaskPairDataset,
    PresenceTargetBuilder,
    SegmentationTask,
    StackedTargetBuilder,
    StructuredDenseTask,
    Trainer,
    TrainerConfig,
    get_annotation_augmentation,
    get_classification_augmentation,
    get_segmentation_augmentation,
    load_trained_model,
    split_indices,
)
from tiatoolbox.models.training.checkpoint import (
    extract_model_state_dict,
    save_checkpoint,
    save_model_weights,
)
from tiatoolbox.models.training.config import (
    resolve_trainer_amp,
    resolve_trainer_device,
)
from tiatoolbox.models.training.tasks import TrainingTaskABC

if TYPE_CHECKING:
    from pathlib import Path


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


class LightweightKongNet(KongNet):
    """Tiny network exercising KongNet's actual state-dictionary contract."""

    def __init__(self) -> None:
        """Initialize only a tiny classifier, avoiding the full KongNet encoder."""
        nn.Module.__init__(self)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3 * 4 * 4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.linear(self.flatten(inputs))


class DictOutputClassifier(nn.Module):
    """Small model returning logits under a mapping output."""

    def __init__(self) -> None:
        """Initialize the dict-output test model."""
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3 * 4 * 4, 2)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a forward pass and wrap the logits in a dict."""
        return {"logits": self.linear(self.flatten(inputs))}


class MinimalModelABC(ModelABC):
    """Small ModelABC subclass for direct weight-loading tests."""

    def __init__(self) -> None:
        """Initialize the minimal model."""
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3 * 4 * 4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.linear(self.flatten(inputs))

    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: np.ndarray | torch.Tensor,
        *,
        device: str,
    ) -> np.ndarray:
        """Run a minimal inference batch for abstract interface completeness."""
        del device
        with torch.no_grad():
            if isinstance(batch_data, np.ndarray):
                batch_data = torch.from_numpy(batch_data)
            return model(batch_data.float()).detach().cpu().numpy()


class StructuredClassificationTask(TrainingTaskABC):
    """Test task consuming dict outputs, dict targets, and epoch hooks."""

    def __init__(self) -> None:
        """Initialize the structured test task."""
        super().__init__(output_key="logits")
        self.loss_fn = nn.CrossEntropyLoss()
        self.reset_calls: list[bool] = []
        self.correct = 0
        self.total = 0

    def reset_epoch_state(self, *, training: bool) -> None:
        """Reset tracked prediction counts for the current epoch."""
        self.reset_calls.append(training)
        self.correct = 0
        self.total = 0

    def update_epoch_state(self, output: object, targets: object) -> None:
        """Accumulate accuracy over detached epoch batches."""
        if not isinstance(targets, dict) or not isinstance(
            targets["label"], torch.Tensor
        ):
            msg = (
                "StructuredClassificationTask expects dict targets with tensor labels."
            )
            raise ValueError(msg)  # noqa: TRY004 - preserve validation API

        logits = self.select_output(output)
        labels = targets["label"]
        predictions = torch.argmax(logits, dim=1)
        self.correct += int((predictions == labels).sum().item())
        self.total += int(labels.shape[0])

    def compute_epoch_metrics(self) -> dict[str, float]:
        """Expose epoch-level accuracy based on accumulated state."""
        if self.total == 0:
            return {"epoch_accuracy": 0.0}
        return {"epoch_accuracy": float(self.correct / self.total)}

    def loss_components(
        self, output: object, targets: object
    ) -> dict[str, torch.Tensor]:
        """Compute classification loss from dict outputs and targets."""
        if not isinstance(targets, dict) or not isinstance(
            targets["label"], torch.Tensor
        ):
            msg = (
                "StructuredClassificationTask expects dict targets with tensor labels."
            )
            raise ValueError(msg)  # noqa: TRY004 - preserve validation API

        logits = self.select_output(output)
        return {"loss": self.loss_fn(logits, targets["label"].long())}


def test_patch_folder_classification_dataset(track_tmp_path: Path) -> None:
    """Folder-based dataset should load images and class labels."""
    dataset_root = _build_classification_dataset(track_tmp_path / "class_dataset")
    dataset = PatchFolderClassificationDataset(dataset_root)

    assert len(dataset) == 12
    sample = dataset[0]

    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 16, 16)
    assert sample["target"].dtype == torch.long


def test_patch_folder_classification_dataset_keeps_transformed_target_dtype(
    track_tmp_path: Path,
) -> None:
    """A ``target_transform`` returning float labels keeps them floating."""
    dataset_root = _build_classification_dataset(track_tmp_path / "class_dataset")
    soft_labels = {
        0: np.array([0.8, 0.2], dtype=np.float32),
        1: np.array([0.2, 0.8], dtype=np.float32),
    }
    dataset = PatchFolderClassificationDataset(
        dataset_root, target_transform=lambda label: soft_labels[label]
    )

    target = dataset[0]["target"]
    assert target.dtype == torch.float32
    assert torch.equal(target, torch.tensor([0.8, 0.2]))
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    assert batch["target"].shape == (4, 2)
    loss = ClassificationTask().compute_loss(torch.zeros(4, 2), batch["target"])
    assert torch.isfinite(loss)


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


def test_split_indices_preserves_class_balance() -> None:
    """Stratified index splits should preserve per-class proportions."""
    targets = [0] * 10 + [1] * 10 + [2] * 10

    train_indices, val_indices = split_indices(
        len(targets),
        stratify=targets,
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


def test_classification_augmentation_preset_returns_image_transform() -> None:
    """Classification presets should expose an image-only dataset transform."""
    preset = get_classification_augmentation("light")
    image = np.full((16, 16, 3), fill_value=128, dtype=np.uint8)

    transformed = preset.image_transform(image)

    assert preset.image_transform is not None
    assert preset.pair_transform is None
    assert transformed.shape == image.shape


def test_segmentation_augmentation_preset_splits_pair_and_image_transforms() -> None:
    """Segmentation presets should separate paired geometry from image effects."""
    preset = get_segmentation_augmentation("light")
    image = np.full((16, 16, 3), fill_value=128, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:, :8] = 1

    transformed_image, transformed_mask = preset.pair_transform(image, mask)
    transformed_image = preset.image_transform(transformed_image)

    assert preset.pair_transform is not None
    assert preset.image_transform is not None
    assert transformed_image.shape == image.shape
    assert transformed_mask.shape == mask.shape
    assert set(np.unique(transformed_mask)).issubset({0, 1})


def test_annotation_augmentation_preset_uses_pair_transform_for_masks() -> None:
    """Annotation presets should only enable paired geometry for spatial targets."""
    spatial_preset = get_annotation_augmentation(
        "light",
        target_builder=MaskTargetBuilder(class_mapping={"tumor": 1}),
    )
    scalar_preset = get_annotation_augmentation(
        "light",
        target_builder=PresenceTargetBuilder(),
    )

    assert spatial_preset.pair_transform is not None
    assert spatial_preset.image_transform is not None
    assert scalar_preset.pair_transform is None
    assert scalar_preset.image_transform is not None


def test_annotation_augmentation_preset_supports_structured_spatial_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structured spatial leaves should share geometry while scalars stay fixed."""
    monkeypatch.setattr(
        training_augmentations,
        "_geometric_ops",
        lambda **_kwargs: [v2.RandomHorizontalFlip(p=1.0)],
    )

    preset = get_annotation_augmentation(
        "light",
        target_builder=CompositeTargetBuilder(
            {
                "mask": MaskTargetBuilder(class_mapping={"tumor": 1}),
                "signals": CompositeTargetBuilder(
                    {
                        "disk": BinaryDiskTargetBuilder(radius=1),
                        "heatmap": GaussianHeatmapTargetBuilder(sigma=1.0),
                    },
                ),
                "presence": PresenceTargetBuilder(),
            },
        ),
    )

    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    target = {
        "mask": np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            dtype=np.int64,
        ),
        "signals": {
            "disk": np.array(
                [
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "heatmap": np.array(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2],
                ],
                dtype=np.float32,
            ),
        },
        "presence": 1,
    }

    transformed_image, transformed_target = preset.pair_transform(image, target)

    assert preset.pair_transform is not None
    assert np.array_equal(transformed_image, np.fliplr(image))
    assert np.array_equal(transformed_target["mask"], np.fliplr(target["mask"]))
    assert np.array_equal(
        transformed_target["signals"]["disk"],
        np.fliplr(target["signals"]["disk"]),
    )
    assert np.array_equal(
        transformed_target["signals"]["heatmap"],
        np.fliplr(target["signals"]["heatmap"]),
    )
    assert transformed_target["presence"] == 1


def test_annotation_augmentation_preset_splits_mixed_stacked_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed stacked masks and heatmaps should use per-channel interpolation."""
    monkeypatch.setattr(
        training_augmentations,
        "_geometric_ops",
        lambda **_kwargs: [training_augmentations._TrainingAffine(degrees=(15, 15))],
    )

    target_builder = CompositeTargetBuilder(
        {
            "tumor": StackedTargetBuilder(
                {
                    "mask": MaskTargetBuilder(class_mapping=None),
                    "centroid": GaussianHeatmapTargetBuilder(sigma=1.0),
                }
            ),
            "presence": PresenceTargetBuilder(),
        }
    )
    preset = get_annotation_augmentation("medium", target_builder=target_builder)

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    square = np.zeros((32, 32), dtype=np.float32)
    square[8:24, 8:24] = 1.0
    target = {
        "tumor": np.stack([square, square], axis=-1),
        "presence": 1,
    }

    transformed_image, transformed_target = preset.pair_transform(image, target)
    transformed_stack = transformed_target["tumor"]
    mask_channel = transformed_stack[..., 0]
    heatmap_channel = transformed_stack[..., 1]

    assert preset.pair_transform is not None
    assert transformed_image.shape == image.shape
    assert transformed_stack.shape == target["tumor"].shape
    assert set(np.unique(mask_channel)).issubset({0.0, 1.0})
    assert np.any((heatmap_channel > 0.0) & (heatmap_channel < 1.0))
    assert transformed_target["presence"] == 1


def test_augmentation_presets_construct_for_all_levels() -> None:
    """All preset levels should construct without error."""
    for level in ("light", "medium", "heavy", "strong"):
        assert get_classification_augmentation(level).image_transform is not None
        assert get_segmentation_augmentation(level).pair_transform is not None
        assert (
            get_annotation_augmentation(
                level,
                target_builder=MaskTargetBuilder(class_mapping={"tumor": 1}),
            ).pair_transform
            is not None
        )


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


def test_classification_task_reports_common_shape_mismatches() -> None:
    """Single-label classification should fail early with clear messages."""
    task = ClassificationTask(loss="cross_entropy")
    logits = torch.zeros((2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="targets batch size `3` must match"):
        task.compute_loss(logits, torch.tensor([0, 1, 2], dtype=torch.long))

    with pytest.raises(ValueError, match="outside the valid range"):
        task.compute_loss(logits, torch.tensor([0, 3], dtype=torch.long))


def test_classification_task_validation_for_target_modes() -> None:
    """Classification tasks should enforce compatible loss/target-mode pairs."""
    task = ClassificationTask(
        loss="bce_with_logits",
        target_mode="multi_label",
    )
    assert isinstance(task, ClassificationTask)

    callable_binary_task = ClassificationTask(
        loss=nn.BCEWithLogitsLoss(reduction="none"),
        target_mode="binary",
    )
    assert isinstance(callable_binary_task, ClassificationTask)

    with pytest.raises(ValueError, match="Unsupported classification loss `auto`"):
        _ = ClassificationTask(loss="auto")

    for target_mode in ("binary", "multi_label"):
        with pytest.raises(ValueError, match="requires `loss='bce_with_logits'`"):
            _ = ClassificationTask(loss="cross_entropy", target_mode=target_mode)

    with pytest.raises(
        ValueError,
        match="does not support single-label targets",
    ):
        _ = ClassificationTask(
            loss="bce_with_logits",
            target_mode="single_label",
        )


def test_trainer_export_rebuilds_a_registry_style_model(
    track_tmp_path: Path,
) -> None:
    """``model_config.yaml`` rebuilds a TIAToolbox model like a pretrained one."""
    dataset_root = _build_classification_dataset(track_tmp_path / "class_dataset")
    model = CNNModel("resnet18", num_classes=2)
    model.preproc_func = predefined_preproc_func("kather100k")
    dataset = PatchFolderClassificationDataset(dataset_root, preproc=model.preproc_func)
    ioconfig = IOPatchPredictorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 0.5}],
        patch_input_shape=(16, 16),
        stride_shape=(16, 16),
    )
    class_dict = {index: name for name, index in dataset.class_to_idx.items()}
    output_dir = track_tmp_path / "export_run"
    trainer = Trainer(
        model=model,
        task=ClassificationTask(),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-2),
        train_loader=DataLoader(dataset, batch_size=4),
        val_loader=DataLoader(dataset, batch_size=4),
        config=TrainerConfig(
            max_epochs=1, output_dir=output_dir, amp=False, log_every_n_steps=0
        ),
        export=ModelExport(
            model_kwargs={"backbone": "resnet18", "num_classes": 2},
            ioconfig=ioconfig,
            class_dict=class_dict,
        ),
    )
    trainer.fit()

    config = yaml.safe_load((output_dir / "model_config.yaml").read_text())
    assert config["architecture"]["class"] == "vanilla.CNNModel"
    assert config["dataset"] == "kather100k"
    assert config["weights"] == "best_model_weights.pth"

    loaded, loaded_ioconfig = load_trained_model(output_dir)
    assert isinstance(loaded, CNNModel)
    assert loaded.class_dict == class_dict
    assert loaded.preproc_func.dataset_name == "kather100k"
    assert loaded_ioconfig.patch_input_shape == [16, 16]
    assert isinstance(loaded_ioconfig, IOPatchPredictorConfig)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], tensor.cpu())

    last, _ = load_trained_model(
        output_dir / "model_config.yaml", model=CNNModel, weights="last.ckpt"
    )
    torch.testing.assert_close(
        last.classifier.weight, model.classifier.weight.detach().cpu()
    )


def test_trainer_rejects_a_classifier_head_that_disagrees_with_the_task(
    track_tmp_path: Path,
) -> None:
    """A BCE task needs a sigmoid head, and the export must rebuild that head."""
    dataset_root = _build_classification_dataset(track_tmp_path / "class_dataset")
    loader = DataLoader(PatchFolderClassificationDataset(dataset_root), batch_size=4)
    config = TrainerConfig(
        max_epochs=1, output_dir=track_tmp_path / "run", amp=False, log_every_n_steps=0
    )

    def build(
        model: CNNModel,
        task: ClassificationTask,
        export: ModelExport | None = None,
    ) -> Trainer:
        return Trainer(
            model=model,
            task=task,
            optimizer=torch.optim.SGD(model.parameters(), lr=1e-2),
            train_loader=loader,
            val_loader=loader,
            config=config,
            export=export,
        )

    softmax_head = CNNModel("resnet18", num_classes=2)
    with pytest.raises(ValueError, match="Build the model with output_mode='multi"):
        build(
            softmax_head,
            ClassificationTask(loss="bce_with_logits", target_mode="multi_label"),
        )
    one_output = CNNModel("resnet18", num_classes=1)
    with pytest.raises(ValueError, match="Build the model with output_mode='single"):
        build(one_output, ClassificationTask(loss="cross_entropy"))

    multi_label = CNNModel("resnet18", num_classes=2, output_mode="multi_label")
    task = ClassificationTask(loss="bce_with_logits")
    build(multi_label, task)
    kwargs = {"backbone": "resnet18", "num_classes": 2}
    with pytest.raises(ValueError, match="Record the trained head"):
        build(multi_label, task, ModelExport(model_kwargs=kwargs))
    with pytest.raises(ValueError, match="cannot rebuild CNNModel"):
        build(
            multi_label,
            task,
            ModelExport(model_kwargs={**kwargs, "output_mode": "binary"}),
        )
    with pytest.raises(ValueError, match="Record the trained head"):
        ModelExport(model_kwargs={**kwargs, "num_classes": 3}).check_model(multi_label)
    build(
        multi_label,
        task,
        ModelExport(model_kwargs={**kwargs, "output_mode": "multi_label"}),
    )


def test_binary_classifier_trains_exports_and_infers_probabilities(
    track_tmp_path: Path,
) -> None:
    """A one-output head trained with BCE infers sigmoid probabilities when reloaded."""
    dataset_root = _build_classification_dataset(track_tmp_path / "class_dataset")
    model = CNNModel("resnet18", num_classes=1)
    dataset = PatchFolderClassificationDataset(dataset_root, preproc=model.preproc_func)
    output_dir = track_tmp_path / "binary_run"
    Trainer(
        model=model,
        task=ClassificationTask(loss="bce_with_logits"),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-2),
        train_loader=DataLoader(dataset, batch_size=4),
        val_loader=DataLoader(dataset, batch_size=4),
        config=TrainerConfig(
            max_epochs=1, output_dir=output_dir, amp=False, log_every_n_steps=0
        ),
        export=ModelExport(model_kwargs={"backbone": "resnet18", "num_classes": 1}),
    ).fit()

    loaded, _ = load_trained_model(output_dir)
    assert loaded.output_mode == "binary"
    batch = torch.stack([dataset[index]["image"] for index in (0, 6)])
    inputs = batch.permute(0, 2, 3, 1)
    probabilities = loaded.infer_batch(loaded, inputs, device="cpu")
    with torch.inference_mode():
        expected = torch.sigmoid(loaded(batch.float())).numpy()
    assert probabilities.shape == (2, 1)
    np.testing.assert_allclose(probabilities, expected, rtol=1e-5)
    assert loaded.postproc_func(probabilities).shape == (2,)
    # With a known logit, the activation is sigmoid, not a softmax over one output.
    torch.nn.init.zeros_(loaded.classifier.weight)
    torch.nn.init.constant_(loaded.classifier.bias, -2.0)
    probabilities = loaded.infer_batch(loaded, inputs, device="cpu")
    np.testing.assert_allclose(probabilities, 1 / (1 + np.exp(2.0)), rtol=1e-5)
    assert loaded.postproc_func(probabilities).tolist() == [0, 0]


def test_export_of_custom_models_needs_an_explicit_model(
    track_tmp_path: Path,
) -> None:
    """Classes outside TIAToolbox are never imported from a configuration."""
    model = DictOutputClassifier()
    save_model_weights(model, track_tmp_path / "weights.pth")
    segmentor_ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 1.0}],
        output_resolutions=[{"units": "mpp", "resolution": 1.0}],
        patch_input_shape=(256, 256),
        patch_output_shape=(128, 128),
        stride_shape=(64, 64),
        save_resolution={"units": "mpp", "resolution": 2.0},
    )
    ModelExport(ioconfig=segmentor_ioconfig).save(
        track_tmp_path / "model_config.yaml", model, "weights.pth"
    )
    with pytest.raises(ValueError, match="Pass `model=`"):
        load_trained_model(track_tmp_path)

    target = DictOutputClassifier()
    loaded, ioconfig = load_trained_model(track_tmp_path, model=target)
    assert loaded is target
    torch.testing.assert_close(target.linear.weight, model.linear.weight)
    assert isinstance(ioconfig, IOSegmentorConfig)
    assert ioconfig.patch_output_shape == [128, 128]
    assert ioconfig.save_resolution == {"units": "mpp", "resolution": 2.0}


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


def test_structured_dense_task_supports_weighted_multi_head_tensor_outputs() -> None:
    """Structured dense tasks should split concatenated outputs and weight losses."""
    task = StructuredDenseTask(
        heads=[
            DenseHeadSpec(
                name="segmentation",
                loss="cross_entropy",
                target_key="segmentation",
                channel_slice=slice(0, 2),
                metrics=("dice", "iou"),
            ),
            DenseHeadSpec(
                name="auxiliary",
                loss="mse",
                target_mode="regression",
                target_key="auxiliary",
                channel_slice=2,
                loss_weight=0.5,
                metrics=("mae", "mse"),
            ),
        ]
    )

    output = torch.tensor(
        [
            [
                [[6.0, -6.0], [-6.0, 6.0]],
                [[-6.0, 6.0], [6.0, -6.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    targets = {
        "segmentation": torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long),
        "auxiliary": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
    }

    loss = task.compute_loss(output, targets)
    components = task.loss_components(output, targets)
    metrics = task.compute_metrics(output, targets)

    expected_seg_loss = functional.cross_entropy(
        output[:, :2, :, :],
        targets["segmentation"],
    )
    expected_aux_loss = functional.mse_loss(
        output[:, 2:3, :, :],
        targets["auxiliary"],
    )
    expected_total = expected_seg_loss + (0.5 * expected_aux_loss)

    assert loss.item() == pytest.approx(expected_total.item())
    assert components["segmentation"].item() == pytest.approx(expected_seg_loss.item())
    assert components["auxiliary"].item() == pytest.approx(0.5)
    assert metrics["segmentation_dice"] == pytest.approx(1.0)
    assert metrics["segmentation_iou"] == pytest.approx(1.0)
    assert metrics["auxiliary_mae"] == pytest.approx(1.0)
    assert metrics["auxiliary_mse"] == pytest.approx(1.0)


def test_structured_dense_task_supports_named_mapping_outputs() -> None:
    """Structured dense tasks should resolve named heads from mapping outputs."""
    task = StructuredDenseTask(
        heads=[
            DenseHeadSpec(
                name="mask",
                loss="cross_entropy",
                metrics=("dice", "iou"),
            ),
            DenseHeadSpec(
                name="heatmap",
                loss="bce_with_logits",
                metrics=("dice", "iou"),
            ),
        ]
    )

    output = {
        "mask": torch.tensor(
            [
                [
                    [[6.0, -6.0], [-6.0, 6.0]],
                    [[-6.0, 6.0], [6.0, -6.0]],
                ]
            ],
            dtype=torch.float32,
        ),
        "heatmap": torch.tensor([[[[6.0, -6.0], [6.0, -6.0]]]], dtype=torch.float32),
    }
    targets = {
        "mask": torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long),
        "heatmap": torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32),
    }

    loss = task.compute_loss(output, targets)
    metrics = task.compute_metrics(output, targets)

    assert torch.isfinite(loss)
    assert metrics["mask_dice"] == pytest.approx(1.0)
    assert metrics["mask_iou"] == pytest.approx(1.0)
    assert metrics["heatmap_dice"] == pytest.approx(1.0)
    assert metrics["heatmap_iou"] == pytest.approx(1.0)


def test_structured_dense_task_matches_single_head_segmentation_behavior() -> None:
    """Single-head structured dense tasks should match segmentation-task outputs."""
    structured_task = StructuredDenseTask(
        heads=[
            DenseHeadSpec(
                name="segmentation",
                loss="cross_entropy",
                metrics=("dice", "iou"),
            )
        ],
        prefix_head_metrics=False,
    )
    segmentation_task = SegmentationTask(loss="cross_entropy")

    logits = torch.tensor(
        [
            [
                [[6.0, -6.0], [-6.0, 6.0]],
                [[-6.0, 6.0], [6.0, -6.0]],
            ]
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)

    structured_loss = structured_task.compute_loss(logits, targets)
    structured_metrics = structured_task.compute_metrics(logits, targets)
    segmentation_loss = segmentation_task.compute_loss(logits, targets)
    segmentation_metrics = segmentation_task.compute_metrics(logits, targets)

    assert structured_loss.item() == pytest.approx(segmentation_loss.item())
    assert structured_metrics == segmentation_metrics


def test_structured_dense_task_reports_common_shape_mismatches() -> None:
    """Dense tasks should report target/logit mismatches before loss kernels."""
    task = StructuredDenseTask(
        heads=[DenseHeadSpec(name="segmentation", loss="cross_entropy")],
    )
    logits = torch.zeros((1, 2, 4, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match=r"target spatial shape `\(3, 4\)`"):
        task.compute_loss(logits, torch.zeros((1, 3, 4), dtype=torch.long))

    with pytest.raises(ValueError, match="outside the valid range"):
        task.compute_loss(logits, torch.full((1, 4, 4), 2, dtype=torch.long))


def test_output_spec_builds_one_head_per_model_output() -> None:
    """Named output groups become dense heads with per-name overrides."""
    model = KongNet(
        num_heads=2,
        num_channels_per_head=[3, 3],
        target_channels=[2, 5],
        min_distance=5,
        threshold_abs=0.5,
        class_dict={0: "Tumour Cell", 1: "Lymphocyte"},
    )
    task = StructuredDenseTask.from_output_spec(
        model.training_output_spec,
        loss="bce_with_logits",
        target_keys={"tumour_cell": "tumour"},
        loss_weights={"lymphocyte": 2.0},
    )

    assert [head.name for head in task.heads] == ["tumour_cell", "lymphocyte"]
    assert [head.channel_slice for head in task.heads] == [slice(0, 3), slice(3, 6)]
    assert [head.target_key for head in task.heads] == ["tumour", None]
    assert [head.loss_weight for head in task.heads] == [1.0, 2.0]

    output = torch.full((1, 6, 2, 2), -6.0)
    output[:, 0:3, :, 0] = 6.0
    output[:, 3:6, :, 1] = 6.0
    targets = {
        "tumour": (output[:, 0:3] > 0).float(),
        "lymphocyte": (output[:, 3:6] > 0).float(),
    }
    metrics = task.compute_metrics(output, targets)
    assert metrics["tumour_cell_dice"] == pytest.approx(1.0)
    assert metrics["lymphocyte_dice"] == pytest.approx(1.0)
    assert torch.isfinite(task.compute_loss(output, targets))


def test_output_spec_accepts_any_named_channel_groups() -> None:
    """Any objects with ``name`` and ``channel_slice`` describe output heads."""
    head = namedtuple("Head", ["name", "channel_slice"])  # noqa: PYI024
    task = StructuredDenseTask.from_output_spec(
        [head("mask", slice(0, 2)), head("distance", 2)],
        loss="mse",
        prefix_head_metrics=False,
    )
    assert [spec.channel_slice for spec in task.heads] == [slice(0, 2), 2]
    with pytest.raises(ValueError, match=r"`loss_weights` names unknown output heads"):
        StructuredDenseTask.from_output_spec([head("mask", 0)], loss_weights={"x": 2})


def test_trainer_supports_structured_targets_and_epoch_hooks(
    track_tmp_path: Path,
) -> None:
    """Trainer should support structured targets and epoch-level task metrics."""
    dataset = [
        {
            "image": torch.zeros((3, 4, 4), dtype=torch.float32),
            "target": {
                "label": torch.tensor(0, dtype=torch.long),
                "meta": {
                    "weight": torch.tensor(1.0, dtype=torch.float32),
                },
            },
        },
        {
            "image": torch.ones((3, 4, 4), dtype=torch.float32),
            "target": {
                "label": torch.tensor(1, dtype=torch.long),
                "meta": {
                    "weight": torch.tensor(1.0, dtype=torch.float32),
                },
            },
        },
        {
            "image": torch.zeros((3, 4, 4), dtype=torch.float32),
            "target": {
                "label": torch.tensor(0, dtype=torch.long),
                "meta": {
                    "weight": torch.tensor(1.0, dtype=torch.float32),
                },
            },
        },
        {
            "image": torch.ones((3, 4, 4), dtype=torch.float32),
            "target": {
                "label": torch.tensor(1, dtype=torch.long),
                "meta": {
                    "weight": torch.tensor(1.0, dtype=torch.float32),
                },
            },
        },
    ]
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = DictOutputClassifier()
    with torch.no_grad():
        model.linear.weight[0].fill_(-0.5)
        model.linear.weight[1].fill_(0.5)
        model.linear.bias.copy_(torch.tensor([1.0, -1.0]))

    task = StructuredClassificationTask()
    trainer = Trainer(
        model=model,
        task=task,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        train_loader=train_loader,
        val_loader=val_loader,
        config=TrainerConfig(
            max_epochs=1,
            amp=False,
            output_dir=track_tmp_path / "structured_targets",
            log_every_n_steps=0,
        ),
        checkpoint_config=CheckpointConfig(save_best=False, save_last=False),
    )

    history = trainer.fit()

    assert history[-1]["train_epoch_accuracy"] == pytest.approx(1.0)
    assert history[-1]["val_epoch_accuracy"] == pytest.approx(1.0)
    assert task.reset_calls == [True, False]


def test_trainer_config_auto_device_and_amp_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trainer config helpers should resolve auto device and AMP consistently."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_trainer_device("auto") == torch.device("cpu")
    assert resolve_trainer_amp(amp="auto", device="auto") is False
    assert resolve_trainer_amp(amp=True, device="cpu") is False
    assert resolve_trainer_amp(amp=False, device="cuda") is False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_trainer_device("auto") == torch.device("cuda")
    assert resolve_trainer_amp(amp="auto", device="auto") is True
    assert resolve_trainer_amp(amp=True, device="cuda") is True


def test_trainer_uses_auto_device_and_amp_defaults(track_tmp_path: Path) -> None:
    """TrainerConfig defaults should create a CPU trainer on CPU-only systems."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = _build_dummy_loader()

    trainer = Trainer(
        model=model,
        task=ClassificationTask(),
        optimizer=optimizer,
        train_loader=loader,
        val_loader=loader,
        config=TrainerConfig(
            max_epochs=1,
            output_dir=track_tmp_path / "auto_config",
        ),
        checkpoint_config=CheckpointConfig(save_best=False, save_last=False),
    )

    assert trainer.device.type in {"cpu", "cuda"}
    assert trainer.use_amp is (trainer.device.type == "cuda")


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
            early_stopping_patience=1,
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
    model = LightweightKongNet()
    with torch.no_grad():
        model.linear.weight.fill_(1.5)
        model.linear.bias.fill_(-0.25)

    weights_path = track_tmp_path / "wrapped_weights.pth"
    save_model_weights(model, weights_path)

    reloaded_model = LightweightKongNet()
    load_torch_model(reloaded_model, weights_path)

    assert torch.allclose(reloaded_model.linear.weight, model.linear.weight)
    assert torch.allclose(reloaded_model.linear.bias, model.linear.bias)


def test_modelabc_load_weights_from_file_accepts_trainer_checkpoints(
    track_tmp_path: Path,
) -> None:
    """Direct ModelABC loading should accept trainer checkpoint payloads."""
    model = MinimalModelABC()
    with torch.no_grad():
        model.linear.weight.fill_(1.25)
        model.linear.bias.fill_(0.75)

    checkpoint_path = track_tmp_path / "modelabc_trainer_payload.ckpt"
    save_checkpoint(
        {"model_state_dict": model.state_dict(), "epoch": 1},
        checkpoint_path,
    )

    reloaded_model = MinimalModelABC()
    reloaded_model.load_weights_from_file(checkpoint_path)

    assert torch.allclose(reloaded_model.linear.weight, model.linear.weight)
    assert torch.allclose(reloaded_model.linear.bias, model.linear.bias)


def test_trainer_resume_supports_wrapped_load_state_dict_models(
    track_tmp_path: Path,
) -> None:
    """Trainer resume should restore wrapped-load models from training checkpoints."""
    model = LightweightKongNet()
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

    resumed_model = LightweightKongNet()
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

    expected_losses = functional.binary_cross_entropy_with_logits(
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
    """All-ignored binary masks should produce zero loss and metrics."""
    task = SegmentationTask(loss="bce_with_logits", ignore_index=-100)
    logits = torch.tensor([[[[2.0, -2.0], [1.0, -1.0]]]], dtype=torch.float32)
    targets = torch.full((1, 2, 2), fill_value=-100, dtype=torch.long)

    loss = task.compute_loss(logits, targets)
    metrics = task.compute_metrics(logits, targets)

    assert loss.item() == pytest.approx(0.0)
    assert metrics["dice"] == pytest.approx(0.0)
    assert metrics["iou"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("loss", "target_mode", "implied"),
    [
        ("cross_entropy", "binary", "multiclass"),
        ("cross_entropy", "regression", "multiclass"),
        ("bce_with_logits", "multiclass", "binary"),
        ("mse", "binary", "regression"),
        ("smooth_l1", "multiclass", "regression"),
    ],
)
def test_dense_tasks_reject_conflicting_named_losses(
    loss: str, target_mode: str, implied: str
) -> None:
    """Named losses fix the target mode, so conflicts fail at construction."""
    head = DenseHeadSpec("head", loss=loss, target_mode=implied)
    assert StructuredDenseTask([head]).heads == [head]
    with pytest.raises(ValueError, match=f"supervises `{implied}` targets"):
        StructuredDenseTask([DenseHeadSpec("head", loss=loss, target_mode=target_mode)])
    with pytest.raises(ValueError, match=f"supervises `{implied}` targets"):
        SegmentationTask(loss=loss, target_mode=target_mode)
    with pytest.raises(ValueError, match="Unsupported dense loss `hinge`"):
        StructuredDenseTask([DenseHeadSpec("head", loss="hinge")])


def test_wrapper_detection_uses_parallel_module_types() -> None:
    """Only DataParallel/DDP are unwrapped, not models owning a `module` attribute."""

    class OwnsModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.module = nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.module(inputs)

    owner = OwnsModule()
    assert extract_model_state_dict(owner).keys() == owner.state_dict().keys()
    load_model_state_dict(
        owner, {"module.weight": torch.ones(2, 2), "module.bias": torch.zeros(2)}
    )
    assert torch.equal(owner.module.weight, torch.ones(2, 2))

    parallel = nn.DataParallel(nn.Linear(2, 2))
    assert set(extract_model_state_dict(parallel)) == {"weight", "bias"}
    load_model_state_dict(
        parallel,
        {"model_state_dict": {"weight": torch.ones(2, 2), "bias": torch.zeros(2)}},
    )
    # DataParallel moves its module to the first GPU when one is available.
    assert torch.equal(parallel.module.weight.cpu(), torch.ones(2, 2))
