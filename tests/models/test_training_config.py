"""Validation and device policy tests for training configuration."""

from __future__ import annotations

import pytest
import torch

from tiatoolbox.models.training import CheckpointConfig, TrainerConfig
from tiatoolbox.models.training.config import (
    resolve_trainer_amp,
    resolve_trainer_device,
)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("monitor_mode", "median"),
        ("scheduler_interval", "batch"),
        ("amp_dtype", "float32"),
        ("max_epochs", 0),
        ("grad_accum_steps", 0),
        ("val_interval", 0),
        ("early_stopping_patience", 0),
        ("grad_clip_norm", 0),
    ],
)
def test_invalid_trainer_options(option: str, value: object) -> None:
    """Reject invalid settings with the offending option in the message."""
    with pytest.raises(ValueError, match=option):
        TrainerConfig(**{option: value})


def test_invalid_amp_option() -> None:
    """Only booleans and auto are accepted for mixed precision."""
    with pytest.raises(TypeError, match="`amp`"):
        TrainerConfig(amp="yes")


@pytest.mark.parametrize("interval", [0, -1])
def test_invalid_checkpoint_interval(interval: int) -> None:
    """Checkpoint intervals must be positive."""
    with pytest.raises(ValueError, match="save_last_every"):
        CheckpointConfig(save_last_every=interval)


@pytest.mark.parametrize("available", [False, True])
def test_auto_device_uses_cuda_availability(
    monkeypatch: pytest.MonkeyPatch, *, available: bool
) -> None:
    """Automatic device selection follows CUDA availability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    expected = torch.device("cuda" if available else "cpu")
    assert resolve_trainer_device() == expected
    assert resolve_trainer_device(torch.device("cpu")) == torch.device("cpu")


@pytest.mark.parametrize(
    ("device", "amp", "dtype", "expected"),
    [
        ("cuda", "auto", "float16", True),
        ("cuda", False, "float16", False),
        ("cpu", "auto", "bfloat16", False),
        ("cpu", True, "float16", False),
        ("cpu", False, "bfloat16", False),
        ("cpu", True, "bfloat16", True),
    ],
)
def test_amp_device_policy(
    device: str, dtype: str, *, amp: bool | str, expected: bool
) -> None:
    """CPU autocast requires explicit bfloat16 while CUDA enables it automatically."""
    assert resolve_trainer_amp(device=device, amp=amp, dtype=dtype) is expected
