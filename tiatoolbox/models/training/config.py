"""Configuration objects for TIAToolbox model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint file names and save behavior."""

    save_last: bool = True
    save_best: bool = True
    last_filename: str = "last.ckpt"
    best_weights_filename: str = "best_model_weights.pth"


@dataclass
class TrainerConfig:
    """Configuration for training loop behavior."""

    max_epochs: int = 1
    device: str = "cpu"
    amp: bool = True
    grad_accum_steps: int = 1
    grad_clip_norm: float | None = None
    val_interval: int = 1
    seed: int | None = None
    deterministic: bool = False
    monitor: str = "val_loss"
    monitor_mode: Literal["min", "max"] = "min"
    early_stopping_patience: int | None = None
    output_dir: Path | str = Path(".")
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
        self.output_dir = Path(self.output_dir)
