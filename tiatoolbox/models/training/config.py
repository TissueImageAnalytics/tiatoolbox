"""Configuration objects for TIAToolbox model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint and artifact file names/save behavior."""

    save_last: bool = True
    save_best: bool = True
    save_artifact: bool = True
    last_filename: str = "last.ckpt"
    best_weights_filename: str = "best_model_weights.pth"
    artifact_filename: str = "training_artifact.json"


def resolve_trainer_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve a trainer device specification to a concrete torch device."""
    if isinstance(device, str) and device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_trainer_amp(
    *,
    amp: bool | Literal["auto"] = "auto",
    device: str | torch.device = "auto",
) -> bool:
    """Resolve an AMP setting for a concrete or automatic trainer device."""
    resolved_device = resolve_trainer_device(device)
    if amp == "auto":
        return resolved_device.type == "cuda"
    if isinstance(amp, bool):
        return amp and resolved_device.type == "cuda"
    msg = "`amp` must be a boolean or 'auto'."
    raise TypeError(msg)


@dataclass
class TrainerConfig:
    """Configuration for training loop behavior."""

    max_epochs: int = 1
    device: str | torch.device = "auto"
    amp: bool | Literal["auto"] = "auto"
    grad_accum_steps: int = 1
    grad_clip_norm: float | None = None
    val_interval: int = 1
    seed: int | None = None
    deterministic: bool = False
    monitor: str = "val_loss"
    monitor_mode: Literal["min", "max"] = "min"
    early_stopping_patience: int | None = None
    output_dir: Path | str = Path()
    log_every_n_steps: int = 20

    def __post_init__(self: TrainerConfig) -> None:
        """Validate trainer configuration values."""
        if self.max_epochs <= 0:
            msg = "`max_epochs` must be a positive integer."
            raise ValueError(msg)
        if self.grad_accum_steps <= 0:
            msg = "`grad_accum_steps` must be a positive integer."
            raise ValueError(msg)
        if self.val_interval <= 0:
            msg = "`val_interval` must be a positive integer."
            raise ValueError(msg)
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 0
        ):
            msg = "`early_stopping_patience` must be non-negative when provided."
            raise ValueError(msg)
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            msg = "`grad_clip_norm` must be positive when provided."
            raise ValueError(msg)
        if not isinstance(self.amp, bool) and self.amp != "auto":
            msg = "`amp` must be a boolean or 'auto'."
            raise TypeError(msg)
        self.output_dir = Path(self.output_dir)
