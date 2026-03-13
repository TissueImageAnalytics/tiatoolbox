"""Training loop implementation for TIAToolbox models."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR
from torch.utils.data import DataLoader

from tiatoolbox import logger
from tiatoolbox.models.training.checkpoint import (
    extract_model_state_dict,
    load_checkpoint,
    load_model_state_dict,
    save_checkpoint,
    save_model_weights,
)
from tiatoolbox.models.training.config import (
    CheckpointConfig,
    DataLoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
    TaskConfig,
    TrainerConfig,
)
from tiatoolbox.models.training.tasks import (
    ClassificationTask,
    SegmentationTask,
    TrainingTaskABC,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from torch.utils.data import Dataset, Sampler


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Set random seeds used by Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_optimizer(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    """Create an optimizer from :class:`OptimizerConfig`."""
    if config.name == "adamw":
        return torch.optim.AdamW(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )

    if config.name == "sgd":
        return torch.optim.SGD(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )

    msg = f"Unsupported optimizer `{config.name}`."
    raise ValueError(msg)


def create_dataloader(
    dataset: Dataset,
    config: DataLoaderConfig,
    *,
    sampler: Sampler[int] | None = None,
    collate_fn: Callable | None = None,
) -> DataLoader:
    """Create a dataloader from :class:`DataLoaderConfig`."""
    if sampler is not None and config.shuffle:
        msg = "`shuffle=True` cannot be combined with an explicit sampler."
        raise ValueError(msg)

    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=config.shuffle if sampler is None else False,
        sampler=sampler,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        persistent_workers=config.persistent_workers,
        collate_fn=collate_fn,
    )


def create_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
) -> LRScheduler | None:
    """Create a scheduler from :class:`SchedulerConfig`."""
    if config.name == "none":
        return None
    if config.name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.t_max,
            eta_min=config.eta_min,
        )
    if config.name == "step":
        return StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    msg = f"Unsupported scheduler `{config.name}`."
    raise ValueError(msg)


def create_task(config: TaskConfig) -> TrainingTaskABC:
    """Create a task adapter from :class:`TaskConfig`."""
    kwargs = {
        "loss": config.loss,
        "output_key": config.output_key,
        "output_index": config.output_index,
        "ignore_index": config.ignore_index,
    }

    if config.task_type == "classification":
        kwargs["target_mode"] = config.target_mode
        return ClassificationTask(**kwargs)
    if config.task_type == "segmentation":
        return SegmentationTask(**kwargs)

    msg = f"Unsupported task type `{config.task_type}`."
    raise ValueError(msg)


class Trainer:
    """General-purpose training loop for TIAToolbox models."""

    def __init__(
        self: Trainer,
        model: nn.Module,
        task: TrainingTaskABC,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        scheduler: LRScheduler | None = None,
        config: TrainerConfig | None = None,
        checkpoint_config: CheckpointConfig | None = None,
    ) -> None:
        """Initialize :class:`Trainer`."""
        self.config = config or TrainerConfig()
        self.checkpoint_config = checkpoint_config or CheckpointConfig()

        if self.config.seed is not None:
            set_seed(self.config.seed, deterministic=self.config.deterministic)

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)

        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.use_amp = self.config.amp and self.device.type == "cuda"
        self.grad_scaler = GradScaler(self.device.type, enabled=self.use_amp)

        self._validate_monitor_configuration()

        self.history: list[dict[str, float]] = []
        self.best_monitor_value = self._initial_monitor_value()
        self.best_epoch = -1

    def _initial_monitor_value(self: Trainer) -> float:
        """Return initial sentinel based on monitor mode."""
        return float("inf") if self.config.monitor_mode == "min" else -float("inf")

    def _extract_batch(
        self: Trainer,
        batch: dict | list | tuple,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract image and target tensors from a dataloader batch."""
        images = None
        targets = None

        if isinstance(batch, dict):
            images = batch.get("image")
            targets = batch.get("target", batch.get("label"))
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images = batch[0]
            targets = batch[1]

        if not isinstance(images, torch.Tensor) or not isinstance(
            targets, torch.Tensor
        ):
            msg = "Batch must provide tensor `image` and `target`/`label` entries."
            raise ValueError(msg)

        # Accept NHWC tensors for convenience.
        if (
            images.ndim == 4
            and images.shape[1] not in {1, 3, 4}
            and images.shape[-1] in {1, 3, 4}
        ):
            images = images.permute(0, 3, 1, 2).contiguous()

        return images, targets

    def _is_improved(self: Trainer, value: float) -> bool:
        """Check whether monitored value improved."""
        if self.config.monitor_mode == "min":
            return value < self.best_monitor_value
        return value > self.best_monitor_value

    def _validate_monitor_configuration(self: Trainer) -> None:
        """Validate monitor configuration against available loaders."""
        if self.config.monitor.startswith("val_") and self.val_loader is None:
            msg = (
                f"Monitor `{self.config.monitor}` requires a validation loader, "
                "but `val_loader` is `None`."
            )
            raise ValueError(msg)

    def _resolve_monitor_value(
        self: Trainer,
        metrics: dict[str, float],
    ) -> float | None:
        """Resolve monitor value from epoch metrics, if available."""
        if self.config.monitor in metrics:
            return metrics[self.config.monitor]

        if self.config.monitor.startswith("val_"):
            return None

        msg = (
            f"Monitor metric `{self.config.monitor}` not found in epoch metrics: "
            f"{sorted(metrics.keys())}."
        )
        raise KeyError(msg)

    def _optimizer_step(self: Trainer) -> None:
        """Perform one optimizer step, including AMP and gradient clipping."""
        if self.config.grad_clip_norm is not None:
            if self.use_amp:
                self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.grad_clip_norm,
            )

        if self.use_amp:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

    def _step_scheduler(self: Trainer, monitor_value: float | None) -> None:
        """Advance scheduler state."""
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if monitor_value is None:
                return
            self.scheduler.step(monitor_value)
            return
        self.scheduler.step()

    def _run_epoch(
        self: Trainer,
        loader: DataLoader,
        *,
        training: bool,
    ) -> dict[str, float]:
        """Run one training or validation epoch and return averaged metrics."""
        self.model.train(mode=training)
        mode_name = "train" if training else "val"

        metric_totals: dict[str, float] = {"loss": 0.0}
        total_samples = 0

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        for step_index, batch in enumerate(loader, start=1):
            images, targets = self._extract_batch(batch)
            images = images.to(self.device).float()
            targets = targets.to(self.device)

            with torch.set_grad_enabled(training):
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    output = self.model(images)
                    logits = self.task.select_output(output)
                    loss = self.task.compute_loss(logits, targets)

            batch_size = int(images.shape[0])
            total_samples += batch_size

            batch_metrics = self.task.compute_metrics(logits.detach(), targets.detach())
            metric_totals["loss"] += float(loss.item()) * batch_size
            for metric_name, metric_value in batch_metrics.items():
                metric_totals[metric_name] = (
                    metric_totals.get(metric_name, 0.0) + metric_value * batch_size
                )

            if training:
                scaled_loss = loss / self.config.grad_accum_steps
                if self.use_amp:
                    self.grad_scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                last_step = step_index == len(loader)
                should_step = (
                    step_index % self.config.grad_accum_steps == 0
                ) or last_step
                if should_step:
                    self._optimizer_step()

            if (
                self.config.log_every_n_steps > 0
                and step_index % self.config.log_every_n_steps == 0
            ):
                logger.info(
                    "%s step %d/%d: loss=%.6f",
                    mode_name,
                    step_index,
                    len(loader),
                    float(loss.item()),
                )

        if total_samples == 0:
            msg = f"`{mode_name}` dataloader yielded zero samples."
            raise ValueError(msg)

        return {
            metric_name: metric_total / total_samples
            for metric_name, metric_total in metric_totals.items()
        }

    def _build_checkpoint_state(self: Trainer, epoch: int) -> dict[str, Any]:
        """Build serializable training state."""
        trainer_config = asdict(self.config)
        trainer_config["output_dir"] = str(trainer_config["output_dir"])
        return {
            "epoch": epoch,
            "model_state_dict": extract_model_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": None
            if self.scheduler is None
            else self.scheduler.state_dict(),
            "grad_scaler_state_dict": self.grad_scaler.state_dict()
            if self.use_amp
            else None,
            "best_monitor_value": self.best_monitor_value,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "trainer_config": trainer_config,
        }

    def _resume_from_checkpoint(self: Trainer, checkpoint_path: str | Path) -> int:
        """Restore trainer state from a checkpoint and return start epoch."""
        checkpoint = load_checkpoint(checkpoint_path, map_location=str(self.device))

        model_state_dict = checkpoint["model_state_dict"]
        load_model_state_dict(self.model, model_state_dict, strict=True)

        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = checkpoint.get("grad_scaler_state_dict")
        if self.use_amp and scaler_state is not None:
            self.grad_scaler.load_state_dict(scaler_state)

        self.best_monitor_value = checkpoint.get(
            "best_monitor_value", self._initial_monitor_value()
        )
        self.best_epoch = int(checkpoint.get("best_epoch", -1))
        self.history = list(checkpoint.get("history", []))

        start_epoch = int(checkpoint.get("epoch", 0))
        logger.info(
            "Resumed training from checkpoint `%s` at epoch %d.",
            checkpoint_path,
            start_epoch,
        )
        return start_epoch

    def fit(
        self: Trainer,
        resume_from: str | Path | None = None,
    ) -> list[dict[str, float]]:
        """Run model training and return a per-epoch history."""
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self._resume_from_checkpoint(resume_from)

        patience_counter = 0
        for epoch_idx in range(start_epoch, self.config.max_epochs):
            train_metrics = self._run_epoch(self.train_loader, training=True)
            epoch_metrics: dict[str, float] = {
                "epoch": float(epoch_idx + 1),
                **{f"train_{key}": value for key, value in train_metrics.items()},
            }

            if (
                self.val_loader is not None
                and ((epoch_idx + 1) % self.config.val_interval == 0)
            ):
                val_metrics = self._run_epoch(self.val_loader, training=False)
                epoch_metrics.update(
                    {f"val_{key}": value for key, value in val_metrics.items()}
                )

            monitor_value = self._resolve_monitor_value(epoch_metrics)
            improved = False
            if monitor_value is not None:
                improved = self._is_improved(monitor_value)

                if improved:
                    self.best_monitor_value = monitor_value
                    self.best_epoch = epoch_idx + 1
                    patience_counter = 0
                    if self.checkpoint_config.save_best:
                        best_path = (
                            self.output_dir
                            / self.checkpoint_config.best_weights_filename
                        )
                        save_model_weights(self.model, best_path)
                else:
                    patience_counter += 1

            self._step_scheduler(monitor_value)
            self.history.append(epoch_metrics)

            if self.checkpoint_config.save_last:
                checkpoint_path = self.output_dir / self.checkpoint_config.last_filename
                save_checkpoint(
                    self._build_checkpoint_state(epoch_idx + 1),
                    checkpoint_path,
                )

            if monitor_value is None:
                logger.info(
                    "Epoch %d/%d | monitor `%s` unavailable (validation skipped) | "
                    "best_epoch=%d",
                    epoch_idx + 1,
                    self.config.max_epochs,
                    self.config.monitor,
                    self.best_epoch,
                )
            else:
                logger.info(
                    "Epoch %d/%d | monitor `%s`=%.6f | best=%.6f",
                    epoch_idx + 1,
                    self.config.max_epochs,
                    self.config.monitor,
                    monitor_value,
                    self.best_monitor_value,
                )

            if (
                monitor_value is not None
                and not improved
                and self.config.early_stopping_patience is not None
                and patience_counter > self.config.early_stopping_patience
            ):
                logger.info(
                    "Early stopping triggered at epoch %d (patience=%d).",
                    epoch_idx + 1,
                    self.config.early_stopping_patience,
                )
                break

        return self.history
