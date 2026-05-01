"""Training loop implementation for TIAToolbox models."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from tiatoolbox import logger
from tiatoolbox.models.training.checkpoint import (
    extract_model_state_dict,
    load_checkpoint,
    load_model_state_dict,
    save_checkpoint,
    save_model_weights,
)
from tiatoolbox.models.training.artifact import TrainingArtifactManifest
from tiatoolbox.models.training.config import (
    CheckpointConfig,
    TrainerConfig,
)
from tiatoolbox.models.training.tasks import TrainingTaskABC


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
        artifact_manifest: TrainingArtifactManifest | None = None,
    ) -> None:
        """Initialize :class:`Trainer`."""
        self.config = config or TrainerConfig()
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        self.artifact_manifest = artifact_manifest

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
    ) -> tuple[torch.Tensor, object]:
        """Extract image tensor and target object from a dataloader batch."""
        images = None
        targets = None

        if isinstance(batch, dict):
            images = batch.get("image")
            targets = batch.get("target", batch.get("label"))
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images = batch[0]
            targets = batch[1]

        if not isinstance(images, torch.Tensor) or targets is None:
            msg = (
                "Batch must provide a tensor `image` and `target`/`label` entries."
            )
            raise ValueError(msg)

        # Accept NHWC tensors for convenience.
        if (
            images.ndim == 4
            and images.shape[1] not in {1, 3, 4}
            and images.shape[-1] in {1, 3, 4}
        ):
            images = images.permute(0, 3, 1, 2).contiguous()

        return images, targets

    def _move_to_device(self: Trainer, value: object) -> object:
        """Recursively transfer supported batch values to the active device."""
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._move_to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        return value

    def _detach(self: Trainer, value: object) -> object:
        """Recursively detach tensors from the autograd graph."""
        if isinstance(value, torch.Tensor):
            return value.detach()
        if isinstance(value, dict):
            return {key: self._detach(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._detach(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._detach(item) for item in value)
        return value

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

        self.task.reset_epoch_state(training=training)

        for step_index, batch in enumerate(loader, start=1):
            images, targets = self._extract_batch(batch)
            images = images.to(self.device).float()
            targets = self._move_to_device(targets)

            with torch.set_grad_enabled(training):
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    output = self.model(images)
                    loss = self.task.compute_loss(output, targets)

            batch_size = int(images.shape[0])
            total_samples += batch_size

            detached_output = self._detach(output)
            detached_targets = self._detach(targets)
            batch_metrics = self.task.compute_metrics(detached_output, detached_targets)
            self.task.update_epoch_state(detached_output, detached_targets)
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

        epoch_metrics = {
            metric_name: metric_total / total_samples
            for metric_name, metric_total in metric_totals.items()
        }
        epoch_metrics.update(self.task.compute_epoch_metrics())
        return epoch_metrics

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

    def _save_artifact_manifest(self: Trainer) -> None:
        """Persist lightweight training artifact metadata, when configured."""
        if (
            self.artifact_manifest is None
            or not self.checkpoint_config.save_artifact
        ):
            return

        self.artifact_manifest.record_training_state(
            best_epoch=self.best_epoch,
            best_monitor_value=self.best_monitor_value,
            monitor=self.config.monitor,
            monitor_mode=self.config.monitor_mode,
            history=self.history,
        )
        artifact_path = self.output_dir / self.checkpoint_config.artifact_filename
        self.artifact_manifest.save(artifact_path)

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
                        if self.artifact_manifest is not None:
                            self.artifact_manifest.record_weight(
                                "best",
                                best_path,
                                relative_to=self.output_dir,
                            )
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
                if self.artifact_manifest is not None:
                    self.artifact_manifest.record_checkpoint(
                        "last",
                        checkpoint_path,
                        relative_to=self.output_dir,
                    )

            self._save_artifact_manifest()

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
