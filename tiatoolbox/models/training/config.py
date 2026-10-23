"""Configuration objects for TIAToolbox model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

AmpDtype = Literal["float16", "bfloat16"]


@dataclass
class CheckpointConfig:
    """File names and save behavior for checkpoints, weights and model configs.

    ``last.ckpt`` is rewritten every ``save_last_every`` epochs, and always
    after the final epoch, including one ended by early stopping.
    ``model_config_filename`` is written when the trainer is given a
    :class:`~tiatoolbox.models.training.ModelExport`.
    """

    save_last: bool = True
    save_best: bool = True
    save_last_every: int = 1
    last_filename: str = "last.ckpt"
    best_weights_filename: str = "best_model_weights.pth"
    model_config_filename: str = "model_config.yaml"

    def __post_init__(self: CheckpointConfig) -> None:
        """Validate checkpoint configuration values."""
        if self.save_last_every <= 0:
            msg = "`save_last_every` must be a positive integer."
            raise ValueError(msg)


def resolve_trainer_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve a trainer device specification to a concrete torch device."""
    if isinstance(device, str) and device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_trainer_amp(
    *,
    amp: bool | Literal["auto"] = "auto",
    device: str | torch.device = "auto",
    dtype: AmpDtype = "float16",
) -> bool:
    """Resolve whether mixed precision is enabled on a concrete or automatic device.

    ``"auto"`` enables it on CUDA. ``True`` enables float16 on CUDA, and
    bfloat16 on any device type with autocast support, such as the CPU.
    """
    resolved_device = resolve_trainer_device(device)
    if amp == "auto" or resolved_device.type == "cuda":
        return bool(amp) and resolved_device.type == "cuda"
    return (
        amp
        and dtype == "bfloat16"
        and torch.amp.is_autocast_available(resolved_device.type)
    )


@dataclass
class TrainerConfig:
    """Configuration for training loop behavior.

    ``scheduler_interval='step'`` advances the scheduler after each successful
    optimizer update (one accumulation window). Use this for OneCycleLR and
    other update-based schedules. The default ``'epoch'`` also supports
    ReduceLROnPlateau, using the monitored metric when it is available.

    ``early_stopping_patience=N`` stops training once ``N`` consecutive
    monitored epochs fail to improve on the best score, following the Keras
    and Lightning convention: with ``N=1`` the first non-improving epoch is the
    last one trained. Epochs without a monitor value (validation not scheduled)
    do not count. Resuming a checkpoint whose patience is exhausted stops
    before training another epoch.

    ``amp`` enables mixed precision; ``'auto'`` turns it on for CUDA devices.
    With the default ``amp_dtype='float16'`` only training passes use autocast:
    validation and ``evaluate`` run in full precision, like TIAToolbox's
    inference engines, because eval-mode BatchNorm statistics can push
    activations past float16's range. ``amp_dtype='bfloat16'`` uses bfloat16
    autocast for every pass without loss scaling, on CUDA devices that support
    it or with ``amp=True`` on the CPU.
    ``channels_last`` converts the model and 4D image batches to the
    channels-last memory format, which speeds up convolutions on recent GPUs.
    ``cudnn_benchmark`` lets cuDNN pick the fastest kernels for fixed input
    shapes; it is ignored when ``deterministic`` is set. ``progress_bar`` shows
    a bar for each training and validation pass.
    """

    max_epochs: int = 1
    device: str | torch.device = "auto"
    amp: bool | Literal["auto"] = "auto"
    amp_dtype: AmpDtype = "float16"
    grad_accum_steps: int = 1
    scheduler_interval: Literal["epoch", "step"] = "epoch"
    grad_clip_norm: float | None = None
    val_interval: int = 1
    seed: int | None = None
    deterministic: bool = False
    cudnn_benchmark: bool = False
    channels_last: bool = False
    monitor: str = "val_loss"
    monitor_mode: Literal["min", "max"] = "min"
    early_stopping_patience: int | None = None
    output_dir: Path | str = Path("training_output")
    log_every_n_steps: int = 20
    progress_bar: bool = False

    def __post_init__(self: TrainerConfig) -> None:
        """Validate trainer configuration values."""
        if self.monitor_mode not in {"min", "max"}:
            msg = "`monitor_mode` must be 'min' or 'max'."
            raise ValueError(msg)
        if self.scheduler_interval not in {"epoch", "step"}:
            msg = "`scheduler_interval` must be 'epoch' or 'step'."
            raise ValueError(msg)
        if self.amp_dtype not in {"float16", "bfloat16"}:
            msg = "`amp_dtype` must be 'float16' or 'bfloat16'."
            raise ValueError(msg)
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
            and self.early_stopping_patience <= 0
        ):
            msg = "`early_stopping_patience` must be a positive integer when provided."
            raise ValueError(msg)
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            msg = "`grad_clip_norm` must be positive when provided."
            raise ValueError(msg)
        if not isinstance(self.amp, bool) and self.amp != "auto":
            msg = "`amp` must be a boolean or 'auto'."
            raise TypeError(msg)
        self.output_dir = Path(self.output_dir)
