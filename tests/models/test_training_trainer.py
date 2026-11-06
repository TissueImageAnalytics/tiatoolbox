"""Training loop edge cases and checkpoint compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    DistributedSampler,
    Subset,
    TensorDataset,
)

from tiatoolbox.models.training import (
    CheckpointConfig,
    ClassificationTask,
    ModelExport,
    Trainer,
    TrainerConfig,
)
from tiatoolbox.models.training.checkpoint import save_checkpoint
from tiatoolbox.models.training.trainer import _configure_cudnn, set_seed

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path


@pytest.fixture
def trainer(tmp_path: Path) -> Trainer:
    """Construct a small CPU trainer without automatic checkpoint writes."""
    model = nn.Linear(2, 2)
    loader = DataLoader(
        TensorDataset(torch.ones(4, 2), torch.tensor([0, 1, 0, 1])), batch_size=2
    )
    return Trainer(
        model,
        ClassificationTask(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        loader,
        config=TrainerConfig(
            device="cpu",
            amp=False,
            monitor="train_loss",
            output_dir=tmp_path,
            log_every_n_steps=0,
        ),
        checkpoint_config=CheckpointConfig(save_best=False, save_last=False),
    )


@pytest.mark.parametrize(
    "batch",
    [
        None,
        [],
        [torch.zeros(2, 3)],
        {"image": {"metadata": "only"}, "target": torch.zeros(2)},
        {"image": torch.zeros(2, 3)},
    ],
)
def test_invalid_training_batch(trainer: Trainer, batch: object) -> None:
    """Malformed batches fail before a model forward."""
    with pytest.raises(ValueError, match="Batch must provide"):
        trainer._extract_batch(batch)


def test_nhwc_batch_conversion(trainer: Trainer) -> None:
    """Channel-last convenience batches preserve pixels and labels."""
    image = torch.arange(2 * 8 * 6 * 3).reshape(2, 8, 6, 3)
    labels = torch.tensor([0, 1])
    actual, target = trainer._extract_batch({"image": image, "label": labels})
    torch.testing.assert_close(actual, image.permute(0, 3, 1, 2))
    assert actual.is_contiguous()
    assert target is labels


def test_nested_tensor_transfer(trainer: Trainer) -> None:
    """Transfers preserve container types and non-tensor metadata."""
    inputs = [
        "metadata",
        (torch.ones(2, 3, dtype=torch.int64), {"scale": torch.tensor(2.0)}),
    ]
    result = trainer._prepare_images(inputs)
    assert isinstance(result, list)
    assert isinstance(result[1], tuple)
    assert result[0] == "metadata"
    torch.testing.assert_close(result[1][0], inputs[1][0])
    torch.testing.assert_close(result[1][1]["scale"], inputs[1][1]["scale"])
    image, target = trainer._extract_batch((inputs, torch.zeros(2)))
    assert image is inputs
    torch.testing.assert_close(target, torch.zeros(2))


def test_epoch_reaches_nested_datasets_and_sampler(trainer: Trainer) -> None:
    """Subset/concat wrappers forward epochs to every epoch-aware leaf and sampler."""
    dataset = TensorDataset(torch.ones(2, 2), torch.zeros(2, dtype=torch.long))
    dataset.set_epoch = Mock()
    combined = ConcatDataset([Subset(dataset, [0]), Subset(dataset, [1])])
    sampler = DistributedSampler(combined, num_replicas=1, rank=0)
    loader = DataLoader(combined, sampler=sampler)
    trainer._set_loader_epoch(loader, 3, training=True)
    assert sampler.epoch == 3
    assert dataset.set_epoch.call_count == 2
    dataset.set_epoch.assert_called_with(3)


@pytest.mark.parametrize("scaled", [False, True])
def test_gradient_clipping_bounds_parameter_updates(
    trainer: Trainer, *, scaled: bool
) -> None:
    """Clipping applies to unscaled gradients with or without loss scaling."""
    trainer.config.grad_clip_norm = 0.01
    trainer.grad_scaler = torch.amp.GradScaler("cpu", enabled=scaled)
    before = [value.detach().clone() for value in trainer.model.parameters()]
    trainer.fit()
    update = torch.cat(
        [
            (value.detach() - initial).flatten()
            for value, initial in zip(trainer.model.parameters(), before, strict=True)
        ]
    )
    # Two optimizer steps, each bounded by learning_rate * clip_norm.
    assert 0 < update.norm().item() <= 2 * 0.1 * 0.01 + 1e-7


def test_empty_loader_is_rejected(trainer: Trainer) -> None:
    """An empty evaluation cannot produce misleading metrics."""
    with pytest.raises(ValueError, match="dataloader yielded zero samples"):
        trainer.evaluate(DataLoader([]))


@pytest.mark.parametrize("counts", [{"wrong": 2}, {"loss": 2, "extra": 1}, {}])
@pytest.mark.parametrize("training", [False, True])
def test_loss_normalizer_keys_must_match(
    trainer: Trainer, monkeypatch: pytest.MonkeyPatch, counts: dict, *, training: bool
) -> None:
    """A custom task with mismatched normalizers fails before an optimizer update."""
    monkeypatch.setattr(
        trainer.task, "loss_normalizers", lambda _targets, _size: counts
    )
    before = [value.detach().clone() for value in trainer.model.parameters()]
    with pytest.raises(ValueError, match="matching keys"):
        trainer._run_epoch(trainer.train_loader, training=training)
    for actual, expected in zip(trainer.model.parameters(), before, strict=True):
        torch.testing.assert_close(actual, expected)
        assert actual.grad is None


def test_batch_loss_logging(trainer: Trainer, caplog: pytest.LogCaptureFixture) -> None:
    """Configured logging includes both training and validation batch losses."""
    trainer.config.log_every_n_steps = 1
    with caplog.at_level("INFO", logger="tiatoolbox"):
        trainer.fit()
        trainer.evaluate(trainer.train_loader)
    assert "train step 1: loss=" in caplog.text
    assert "val step 1: loss=" in caplog.text


def test_export_requires_saved_weights(trainer: Trainer) -> None:
    """An export cannot refer to weights the run is configured never to save."""
    with pytest.raises(ValueError, match="needs saved weights"):
        Trainer(
            trainer.model,
            trainer.task,
            trainer.optimizer,
            trainer.train_loader,
            config=trainer.config,
            checkpoint_config=trainer.checkpoint_config,
            export=ModelExport(),
        )


def test_plateau_scheduler_requires_epoch_interval(trainer: Trainer) -> None:
    """Plateau scheduling requires an epoch metric, not an optimizer step."""
    trainer.config.scheduler_interval = "step"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer)
    with pytest.raises(ValueError, match="requires scheduler_interval='epoch'"):
        Trainer(
            trainer.model,
            trainer.task,
            trainer.optimizer,
            trainer.train_loader,
            config=trainer.config,
            scheduler=scheduler,
        )


def test_unsupported_cuda_bfloat16(
    trainer: Trainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsupported CUDA bfloat16 is rejected before constructing an AMP scaler."""
    trainer.device = torch.device("cuda")
    trainer.config.amp = True
    trainer.config.amp_dtype = "bfloat16"
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(ValueError, match="does not support bfloat16"):
        trainer._configure_amp()


def test_resume_restores_scheduler_and_scaler(trainer: Trainer, tmp_path: Path) -> None:
    """Resuming restores optimizer, scheduler, scaler, and epoch counters."""
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(
        trainer.optimizer, step_size=1, gamma=0.5
    )
    trainer.grad_scaler = torch.amp.GradScaler("cpu", init_scale=128)
    trainer.fit()
    state = trainer._build_checkpoint_state(1)
    path = tmp_path / "resume.ckpt"
    save_checkpoint(state, path)
    trainer.scheduler.last_epoch = 99
    trainer.grad_scaler.update(new_scale=16.0)
    trainer.optimizer.param_groups[0]["lr"] = 0.9
    assert trainer._resume_from_checkpoint(path) == 1
    assert trainer.scheduler.state_dict() == state["scheduler_state_dict"]
    assert trainer.grad_scaler.state_dict() == state["grad_scaler_state_dict"]
    assert trainer.optimizer.param_groups[0]["lr"] == 0.05


def test_random_state_ignores_absent_loader_generators(
    trainer: Trainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CPU resume tolerates saved generators for loaders no longer present."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    set_seed(19)
    state = trainer._random_state()
    state["generators"]["removed_loader"] = torch.Generator().get_state()
    expected = torch.rand(3)
    trainer._restore_random_state(state)
    torch.testing.assert_close(torch.rand(3), expected)


def test_cudnn_benchmark_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Benchmarking is enabled on request and deterministic mode overrides it."""
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    _configure_cudnn(deterministic=False, benchmark=True)
    assert torch.backends.cudnn.benchmark
    _configure_cudnn(deterministic=True, benchmark=True)
    assert not torch.backends.cudnn.benchmark
    assert torch.backends.cudnn.deterministic


def test_cuda_random_state_forwarding(
    trainer: Trainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CUDA seed and RNG state are forwarded to torch when CUDA is available."""
    state = trainer._random_state()
    cuda_state = torch.Generator().get_state()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    seed_all, restore = Mock(), Mock()
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seed_all)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restore)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state_all", Mock(return_value=[cuda_state])
    )
    set_seed(23)
    seed_all.assert_called_with(23)
    snapshot = trainer._random_state()
    assert snapshot["cuda"] == [cuda_state]
    trainer._restore_random_state(snapshot)
    restore.assert_called_once()
    torch.testing.assert_close(restore.call_args.args[0][0], cuda_state)
    # Empty CUDA state from a CPU run must not overwrite a device's RNG state.
    state["cuda"] = []
    trainer._restore_random_state(state)
    restore.assert_called_once()
