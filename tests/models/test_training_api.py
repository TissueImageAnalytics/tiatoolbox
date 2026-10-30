"""Regression tests for the training API's extension and composition contracts."""

from __future__ import annotations

import copy
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Subset, TensorDataset

from tiatoolbox.models.architecture.kongnet import KongNet
from tiatoolbox.models.dataset import predefined_preproc_func
from tiatoolbox.models.dataset.dataset_abc import PatchDataset
from tiatoolbox.models.models_abc import ModelABC, load_model_state_dict
from tiatoolbox.models.training import (
    CheckpointConfig,
    ClassificationTask,
    DenseHeadSpec,
    ModelExport,
    PatchFolderClassificationDataset,
    PatchMaskPairDataset,
    SegmentationTask,
    StructuredDenseTask,
    Trainer,
    TrainerConfig,
    get_classification_augmentation,
    get_segmentation_augmentation,
    load_trained_model,
    split_indices,
)
from tiatoolbox.models.training.checkpoint import save_model_weights

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "kind", ["classification", "multilabel", "segmentation", "binary", "regression"]
)
def test_callable_losses_preserve_accumulated_updates(
    tmp_path: Path, kind: str
) -> None:
    """Weighted and smoothed losses give the same update with uneven microbatches."""
    torch.manual_seed(17)
    dense = kind in {"segmentation", "binary", "regression"}
    images = torch.randn(5, 3, 2, 4) if dense else torch.randn(5, 3)
    model = nn.Conv2d(3, 2, 1) if dense else nn.Linear(3, 2)
    if kind in {"classification", "segmentation"}:
        target_shape = (images.shape[0], *images.shape[2:])
        targets = torch.randint(2, target_shape)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.3, 2.5]), label_smoothing=0.1, reduction="none"
        )
        task = (
            SegmentationTask(loss=criterion, target_mode="multiclass")
            if dense
            else ClassificationTask(loss=criterion, target_mode="single_label")
        )
    else:
        target_shape = (images.shape[0], 2, *images.shape[2:])
        targets = torch.rand(target_shape)
        if kind == "regression":
            # Two independently normalized heads with ordinary callable losses.
            task = StructuredDenseTask(
                [
                    DenseHeadSpec(
                        "mse",
                        loss=lambda x, y: (x - y).square(),
                        target_mode="regression",
                        channel_slice=0,
                        loss_weight=0.3,
                    ),
                    DenseHeadSpec(
                        "l1",
                        loss=lambda x, y: (x - y).abs(),
                        target_mode="regression",
                        channel_slice=1,
                    ),
                ]
            )
        else:
            pos_weight = (
                torch.tensor([1.5, 0.7]).reshape(2, 1, 1)
                if dense
                else torch.tensor([1.5, 0.7])
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
            task = (
                SegmentationTask(loss=criterion, target_mode="binary")
                if dense
                else ClassificationTask(loss=criterion, target_mode="multi_label")
            )
    targets[0] = -100
    targets[-1].reshape(-1)[0] = -100
    if kind == "regression":
        # Give the second head a different valid-element count.
        targets[1, 1] = -100
        dataset = [
            (x, {"mse": y[:1], "l1": y[1:]})
            for x, y in zip(images, targets, strict=True)
        ]
    else:
        dataset = list(zip(images, targets, strict=True))

    results = []
    for batch_size, accumulation in ((5, 1), (2, 3)):
        candidate = copy.deepcopy(model)
        trainer = Trainer(
            candidate,
            copy.deepcopy(task),
            torch.optim.SGD(candidate.parameters(), lr=0.1),
            DataLoader(dataset, batch_size=batch_size),
            config=TrainerConfig(
                device="cpu",
                amp=False,
                grad_accum_steps=accumulation,
                monitor="train_loss",
                output_dir=tmp_path / str(batch_size),
                log_every_n_steps=0,
            ),
            checkpoint_config=CheckpointConfig(save_last=False, save_best=False),
        )
        row = trainer.fit()[0]
        results.append((candidate, {k: v for k, v in row.items() if k != "epoch_time"}))
    for full, accumulated in zip(
        results[0][0].parameters(), results[1][0].parameters(), strict=True
    ):
        torch.testing.assert_close(full, accumulated)
    assert results[0][1] == pytest.approx(results[1][1], abs=1e-6)


def test_callable_weighted_mean_and_safe_targets() -> None:
    """The task owns masking and counts valid elements, not class-weight sums."""
    logits = torch.tensor([[1.0, 2.0], [0.0, 1.0], [2.0, -1.0]], requires_grad=True)
    targets = torch.tensor([1, -7, 0])

    def focal(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute unreduced focal loss after checking target indices."""
        assert targets.min() >= 0
        ce = functional.cross_entropy(logits, targets, reduction="none")
        return (1 - torch.exp(-ce)).square() * ce

    task = ClassificationTask(loss=focal, target_mode="single_label", ignore_index=-7)
    loss = task.compute_loss({"out": logits}, targets)
    expected = focal(logits[[0, 2]], targets[[0, 2]]).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[1]) == 0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.0]), reduction="none")
    task = ClassificationTask(
        loss=criterion, target_mode="single_label", ignore_index=-7
    )
    torch.testing.assert_close(
        task.compute_loss(logits, targets),
        criterion(logits[[0, 2]], targets[[0, 2]]).mean(),
    )


def test_callable_loss_rejects_ambiguous_reductions() -> None:
    """Fail before training for reduced modules or unspecified target semantics."""
    with pytest.raises(ValueError, match="reduction='none'"):
        ClassificationTask(loss=nn.CrossEntropyLoss(), target_mode="single_label")
    with pytest.raises(ValueError, match="explicit `target_mode`"):
        SegmentationTask(loss=nn.CrossEntropyLoss(reduction="none"))
    task = ClassificationTask(loss=lambda x, _y: x.mean(), target_mode="single_label")
    with pytest.raises(ValueError, match="one unreduced value per supervised"):
        task.compute_loss(torch.randn(2, 3), torch.tensor([0, 1]))


@pytest.mark.parametrize("dense", [False, True])
def test_all_ignored_callable_receives_safe_targets(*, dense: bool) -> None:
    """Empty supervision is a differentiable zero computed from zeroed targets.

    The criterion still runs, so the task never reads a device-side flag, but
    it only ever sees in-range targets and its values are masked out.
    """
    seen = []

    def criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Record the sanitized targets received by the criterion."""
        seen.append(targets)
        return functional.cross_entropy(logits, targets, reduction="none")

    task = (
        SegmentationTask(loss=criterion, target_mode="multiclass")
        if dense
        else ClassificationTask(loss=criterion, target_mode="single_label")
    )
    logits = torch.randn((2, 3, 4, 4) if dense else (2, 3), requires_grad=True)
    targets = torch.full((2, 4, 4) if dense else (2,), -100)
    loss = task.compute_loss(logits, targets)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0
    assert len(seen) == 1
    assert not seen[0].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_callable_criterion_buffers_follow_device() -> None:
    """A CPU criterion with class weights works when the trainer uses CUDA."""
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.0]), reduction="none")
    task = ClassificationTask(loss=criterion, target_mode="single_label")
    logits = torch.randn(3, 2, device="cuda", requires_grad=True)
    with torch.autocast("cuda"):
        loss = task.compute_loss(logits, torch.tensor([0, 1, -100], device="cuda"))
    assert torch.isfinite(loss)
    assert criterion.weight.device == logits.device
    loss.backward()


def test_grouped_split_is_complete_disjoint_and_repeatable() -> None:
    """Unequal patient groups never cross the validation boundary."""
    groups = ["patient-a"] * 2 + ["patient-b"] * 4 + ["patient-c"] * 3 + ["patient-d"]
    train, val = split_indices(len(groups), 0.5, groups=groups, seed=13)
    assert (train, val) == split_indices(len(groups), 0.5, groups=groups, seed=13)
    assert sorted(train + val) == list(range(len(groups)))
    assert set(train).isdisjoint(val)
    assert {groups[i] for i in train}.isdisjoint({groups[i] for i in val})
    assert len({groups[i] for i in val}) == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"groups": [1]}, "one value per sample"),
        ({"stratify": [0]}, "one value per sample"),
        ({"groups": [1, 2], "stratify": [0, 1]}, "either"),
        ({"val_fraction": 0}, "between 0 and 1"),
    ],
)
def test_split_validation(kwargs: dict, message: str) -> None:
    """Reject configurations that could silently give misleading partitions."""
    with pytest.raises(ValueError, match=message):
        split_indices(2, **kwargs)


def test_split_keeps_transforms_independent(tmp_path: Path) -> None:
    """Index assignment does not give validation the training augmentation."""
    root = tmp_path / "one_class"
    root.mkdir()
    for index in range(6):
        np.save(root / f"{index}.npy", np.zeros((4, 4, 3), dtype=np.uint8))
    calls = []

    def pair(image: np.ndarray, target: int) -> tuple[np.ndarray, int]:
        """Record paired augmentation and increment the image."""
        calls.append("pair")
        return image + 1, target

    def image_transform(image: np.ndarray) -> np.ndarray:
        """Record image augmentation and double its values."""
        calls.append("image")
        return image * 2

    def target_transform(target: int) -> int:
        """Record target augmentation and increment the label."""
        calls.append("target")
        return target + 3

    train_source = PatchFolderClassificationDataset(
        tmp_path,
        pair_transform=pair,
        image_transform=image_transform,
        target_transform=target_transform,
    )
    val_source = PatchFolderClassificationDataset(tmp_path)
    train_idx, val_idx = split_indices(len(train_source), seed=5)
    train, val = Subset(train_source, train_idx), Subset(val_source, val_idx)
    assert train[0]["image"].mean() == pytest.approx(2)
    assert calls == ["pair", "image", "target"]
    calls.clear()
    assert val[0]["image"].count_nonzero() == 0
    assert val[0]["target"] == 0
    assert calls == []


@pytest.mark.parametrize("payload_kind", ["bare", "legacy", "checkpoint"])
def test_kongnet_state_dict_formats(payload_kind: str) -> None:
    """KongNet accepts all supported formats without generic retry loading."""
    model = KongNet.__new__(KongNet)
    nn.Module.__init__(model)
    model.register_parameter("weight", nn.Parameter(torch.zeros(2)))
    state = {"weight": torch.tensor([2.0, 3.0])}
    payload = {
        "bare": state,
        "legacy": {"model": state},
        "checkpoint": {"model_state_dict": state},
    }[payload_kind]
    result = load_model_state_dict(model, payload)
    assert result.missing_keys == []
    torch.testing.assert_close(model.weight, state["weight"])
    result = model.load_state_dict({}, strict=False)
    assert result.missing_keys == ["weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict({}, strict=True)


def test_model_config_weights_resolve_from_its_own_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recorded weight paths do not depend on the working directory."""
    output = tmp_path / "run"
    model = nn.Linear(2, 2)
    save_model_weights(model, output / "best.pth")
    ModelExport().save(output / "model_config.yaml", model, "best.pth")
    monkeypatch.chdir(tmp_path)
    for path in ("run", "run/model_config.yaml"):
        loaded, ioconfig = load_trained_model(path, model=nn.Linear(2, 2))
        torch.testing.assert_close(loaded.weight, model.weight)
        assert ioconfig is None
    with pytest.raises(FileNotFoundError):
        load_trained_model(output, model=nn.Linear(2, 2), weights="missing.pth")


def test_trainer_history_records_lr_epoch_time_and_runs_callbacks(
    tmp_path: Path,
) -> None:
    """Rows carry the epoch's starting learning rate and duration for callbacks."""
    data = TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    seen: list[tuple[Trainer, dict[str, float]]] = []
    trainer = Trainer(
        model,
        ClassificationTask(),
        optimizer,
        DataLoader(data, batch_size=2),
        val_loader=DataLoader(data, batch_size=4),
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
        config=TrainerConfig(
            max_epochs=2,
            device="cpu",
            amp=False,
            output_dir=tmp_path,
            log_every_n_steps=0,
        ),
        checkpoint_config=CheckpointConfig(save_best=False, save_last=False),
        callbacks=[lambda owner, metrics: seen.append((owner, metrics))],
    )
    history = trainer.fit()
    assert [row["lr"] for row in history] == [0.1, 0.05]
    assert all(row["epoch_time"] > 0 for row in history)
    assert [metrics for _, metrics in seen] == history
    assert all(owner is trainer for owner, _ in seen)
    assert seen[0][1] is history[0]

    evaluation = trainer.evaluate()
    expected = {
        key[4:]: value for key, value in history[-1].items() if key.startswith("val_")
    }
    assert evaluation == pytest.approx(expected)
    assert set(trainer.evaluate(trainer.train_loader)) == set(evaluation)
    unvalidated = Trainer(
        model,
        ClassificationTask(),
        optimizer,
        DataLoader(data),
        config=TrainerConfig(device="cpu", monitor="train_loss", output_dir=tmp_path),
    )
    with pytest.raises(ValueError, match="requires a loader"):
        unvalidated.evaluate()


@pytest.mark.parametrize("preproc_name", ["identity", "kongnet", "kather100k"])
def test_dataset_preproc_matches_inference_engine_inputs(
    tmp_path: Path, preproc_name: str
) -> None:
    """With the model's preproc, training sees exactly what ``infer_batch`` sees."""
    preproc = {
        "identity": ModelABC.preproc,
        "kongnet": KongNet.preproc,  # HWC NumPy output
        "kather100k": predefined_preproc_func("kather100k"),  # HWC tensor output
    }[preproc_name]
    patch = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    (tmp_path / "a").mkdir()
    Image.fromarray(patch).save(tmp_path / "a" / "x.png")
    training_image = PatchFolderClassificationDataset(tmp_path, preproc=preproc)[0][
        "image"
    ]
    engine_dataset = PatchDataset(inputs=patch[None], patch_input_shape=(8, 8))
    engine_dataset.preproc_func = preproc
    engine_image = torch.as_tensor(np.asarray(engine_dataset[0]["image"]))
    # ``infer_batch`` converts collated NHWC batches to NCHW float32.
    torch.testing.assert_close(training_image, engine_image.permute(2, 0, 1).float())


def test_image_files_decode_to_rgb_and_arrays_keep_channels(tmp_path: Path) -> None:
    """Files decode to RGB like slide reads; arrays keep their channel count."""
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    (tmp_path / "a").mkdir()
    Image.fromarray(gray).save(tmp_path / "a" / "gray.png")
    image = PatchFolderClassificationDataset(tmp_path)[0]["image"]
    assert image.shape == (3, 4, 4)
    torch.testing.assert_close(image[2], torch.from_numpy(gray).float())

    dataset = PatchMaskPairDataset(images=gray[None], masks=gray[None])
    assert dataset[0]["image"].shape == (1, 4, 4)


def test_trainer_warns_when_dataset_preproc_differs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Datasets must apply the preprocessing that inference engines will apply."""

    class Normalized(ModelABC):
        """Tiny model with non-identity inference preprocessing."""

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 2)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            """Classify the spatially averaged image channels."""
            return self.linear(images.mean((2, 3)))

        @staticmethod
        def preproc(image: np.ndarray) -> np.ndarray:
            """Scale image intensities to the unit interval."""
            return image / 255

        @staticmethod
        def infer_batch(
            model: nn.Module,  # noqa: ARG004 - inference interface
            batch_data: torch.Tensor,  # noqa: ARG004 - inference interface
            *,
            device: str,  # noqa: ARG004 - inference interface
        ) -> None:
            """Leave inference unused in this training-only test model."""
            return

    (tmp_path / "a").mkdir()
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tmp_path / "a" / "x.png")
    model = Normalized()

    def build(preproc: object) -> Trainer:
        """Build a trainer with the requested dataset preprocessing."""
        dataset = PatchFolderClassificationDataset(tmp_path, preproc=preproc)
        return Trainer(
            model,
            ClassificationTask(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            DataLoader(Subset(dataset, [0])),
            config=TrainerConfig(
                device="cpu", monitor="train_loss", output_dir=tmp_path
            ),
        )

    build(None)
    assert "preproc=None" in caplog.text
    caplog.clear()
    build(model.preproc_func)
    assert "preproc_func" not in caplog.text

    # Predefined preprocessing is compared by name, so a separately created copy
    # (as restored by ``load_trained_model``) matches.
    kather = predefined_preproc_func("kather100k")
    assert kather == predefined_preproc_func("kather100k")
    assert hash(kather) == hash(predefined_preproc_func("kather100k"))
    assert kather != predefined_preproc_func("pcam")
    model.preproc_func = kather
    build(predefined_preproc_func("kather100k"))
    assert "preproc_func" not in caplog.text


def test_presets_are_seeded_pair_transforms() -> None:
    """A preset applies geometry then appearance and replays per-sample seeds."""
    preset = get_segmentation_augmentation("heavy")
    image = np.random.default_rng(1).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.int64)
    mask[:4] = 1
    outputs = []
    for _ in range(2):
        preset.set_random_seed(11)
        outputs.append(preset(image, mask))
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    preset.set_random_seed(12)
    assert not np.array_equal(preset(image, mask)[0], outputs[0][0])


def test_with_transforms_shares_the_index_but_not_runtime_state(
    tmp_path: Path,
) -> None:
    """Dataset copies keep samples and mapping, with separate transforms/state."""
    for label in ("a", "b"):
        (tmp_path / label).mkdir()
        Image.fromarray(np.ones((4, 4, 3), np.uint8)).save(tmp_path / label / "x.png")
    base = PatchFolderClassificationDataset(tmp_path, image_transform=np.negative)
    copy_ = base.with_transforms(pair_transform=get_classification_augmentation())
    assert copy_.samples is base.samples
    assert copy_.image_transform is base.image_transform  # `...` keeps a stage
    assert base.pair_transform is None
    assert copy_._augmentation_epoch is not base._augmentation_epoch
    assert base.with_transforms(image_transform=None).image_transform is None


def test_mask_pairs_accept_arrays_and_memory_mapped_stacks(tmp_path: Path) -> None:
    """Arrays and ``.npy`` stacks work, and stacks are not pickled to workers."""
    images = np.random.default_rng(2).integers(0, 255, (3, 32, 32, 3), dtype=np.uint8)
    labels = np.random.default_rng(3).integers(0, 4, (3, 32, 32), dtype=np.int64)
    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "labels.npy", labels)
    from_arrays = PatchMaskPairDataset(images=images, masks=list(labels))
    from_files = PatchMaskPairDataset(
        images=tmp_path / "images.npy", masks=str(tmp_path / "labels.npy")
    )
    assert len(from_files) == 3
    for index in range(3):
        for key in ("image", "target"):
            torch.testing.assert_close(from_arrays[index][key], from_files[index][key])
    assert len(pickle.dumps(from_files)) < images.nbytes // 4
    with pytest.raises(ValueError, match=r"`.npy` file"):
        PatchMaskPairDataset(images=tmp_path, masks=tmp_path)


def test_single_label_soft_targets_use_probability_cross_entropy(
    tmp_path: Path,
) -> None:
    """Floating (N, C) targets are class probabilities, e.g. from mixup."""
    task = ClassificationTask()
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.5]])
    soft = torch.tensor([[0.7, 0.3, 0.0], [0.1, 0.1, 0.8]])
    expected = -(soft * logits.log_softmax(1)).sum(1).mean()
    torch.testing.assert_close(task.compute_loss(logits, soft), expected)
    one_hot = functional.one_hot(torch.tensor([0, 2]), 3).float()
    torch.testing.assert_close(
        task.compute_loss(logits, one_hot),
        task.compute_loss(logits, torch.tensor([0, 2])),
    )
    assert task.loss_normalizers(soft, 2) == {"loss": 2}
    assert task.compute_metrics(logits, soft)["accuracy"] == 0.5
    with pytest.raises(ValueError, match="match the logits shape"):
        task.compute_loss(logits, soft[:, :2])

    loader = DataLoader(TensorDataset(torch.randn(4, 3), soft.repeat(2, 1)))
    model = nn.Linear(3, 3)
    Trainer(
        model,
        task,
        torch.optim.SGD(model.parameters(), lr=0.1),
        loader,
        config=TrainerConfig(device="cpu", monitor="train_loss", output_dir=tmp_path),
    ).fit()


def test_metric_classes_select_and_report_dense_scores() -> None:
    """Listed classes form the mean and are reported individually."""
    logits = torch.full((1, 3, 2, 4), -5.0)
    targets = torch.zeros((1, 2, 4), dtype=torch.long)
    targets[0, 0, :2] = 1
    logits[0, 0] = 5.0  # Background everywhere...
    logits[0, 1, 0, :1] = 10.0  # ...except one correct class-1 pixel.
    plain = SegmentationTask().compute_metrics(logits, targets)
    selected = SegmentationTask(metric_classes=(1, 2)).compute_metrics(logits, targets)
    assert selected["dice_1"] == pytest.approx(2 / 3)
    assert np.isnan(selected["dice_2"])  # Class 2 is absent from both maps.
    assert selected["dice"] == pytest.approx(2 / 3)
    assert plain["dice"] > selected["dice"]  # Background inflated the mean.
    with pytest.raises(ValueError, match="output has 3 channels"):
        SegmentationTask(metric_classes=(3,)).compute_metrics(logits, targets)
    with pytest.raises(ValueError, match="unique, non-negative"):
        SegmentationTask(metric_classes=(1, 1))
