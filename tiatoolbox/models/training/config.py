"""Configuration objects for TIAToolbox model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class DataLoaderConfig:
    """Configuration for a :class:`torch.utils.data.DataLoader`."""

    batch_size: int = 8
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = False
    drop_last: bool = False
    persistent_workers: bool = False

    def __post_init__(self: DataLoaderConfig) -> None:
        """Validate dataloader configuration values."""
        if self.batch_size <= 0:
            msg = "`batch_size` must be a positive integer."
            raise ValueError(msg)
        if self.num_workers < 0:
            msg = "`num_workers` must be non-negative."
            raise ValueError(msg)
        if self.persistent_workers and self.num_workers == 0:
            msg = "`persistent_workers=True` requires `num_workers > 0`."
            raise ValueError(msg)


@dataclass
class OptimizerConfig:
    """Configuration for optimizer creation."""

    name: Literal["adamw", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self: OptimizerConfig) -> None:
        """Validate optimizer configuration values."""
        if self.lr <= 0:
            msg = "`lr` must be positive."
            raise ValueError(msg)
        if self.weight_decay < 0:
            msg = "`weight_decay` must be non-negative."
            raise ValueError(msg)


@dataclass
class SchedulerConfig:
    """Configuration for learning-rate scheduler creation."""

    name: Literal["none", "cosine", "step"] = "none"
    step_size: int = 10
    gamma: float = 0.1
    t_max: int = 10
    eta_min: float = 0.0

    def __post_init__(self: SchedulerConfig) -> None:
        """Validate scheduler configuration values."""
        if self.step_size <= 0:
            msg = "`step_size` must be a positive integer."
            raise ValueError(msg)
        if self.t_max <= 0:
            msg = "`t_max` must be a positive integer."
            raise ValueError(msg)
        if not 0 < self.gamma <= 1:
            msg = "`gamma` must be in the interval (0, 1]."
            raise ValueError(msg)


@dataclass
class TaskConfig:
    """Configuration for task adapters used by the trainer."""

    task_type: Literal["classification", "segmentation"] = "classification"
    loss: Literal["auto", "cross_entropy", "bce_with_logits"] = "cross_entropy"
    output_key: str | None = None
    output_index: int | None = None
    ignore_index: int = -100


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
