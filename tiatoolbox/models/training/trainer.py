"""Training loop implementation for TIAToolbox models."""

from __future__ import annotations

import random
import shutil
import time
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from tiatoolbox import logger
from tiatoolbox.models.models_abc import ModelABC, load_model_state_dict, unwrap_model
from tiatoolbox.models.training.checkpoint import (
    extract_model_state_dict,
    load_checkpoint,
    save_checkpoint,
    save_model_weights,
)
from tiatoolbox.models.training.config import (
    CheckpointConfig,
    TrainerConfig,
    resolve_trainer_amp,
    resolve_trainer_device,
)
from tiatoolbox.models.training.datasets import _TrainingDataset
from tiatoolbox.models.training.tasks import ClassificationTask

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader, Dataset

    from tiatoolbox.models.training.export import ModelExport
    from tiatoolbox.models.training.tasks import TrainingTaskABC

    EpochCallback = Callable[["Trainer", dict[str, float]], None]


_BATCH_IMAGE_DIMS = 4
_MIN_BATCH_ITEMS = 2


def _inner_datasets(dataset: Dataset) -> Iterator[Dataset]:
    """Yield the datasets inside ``Subset``- and ``ConcatDataset``-like wrappers.

    Training datasets and other epoch-aware datasets are leaves, as are
    datasets without a ``dataset``/``datasets`` attribute.
    """
    if isinstance(dataset, _TrainingDataset) or hasattr(dataset, "set_epoch"):
        yield dataset
    elif hasattr(dataset, "dataset"):
        yield from _inner_datasets(dataset.dataset)
    elif hasattr(dataset, "datasets"):
        for child in dataset.datasets:
            yield from _inner_datasets(child)
    else:
        yield dataset


def _map_tensors(value: object, function: Callable) -> object:
    """Apply ``function`` to every tensor in nested dicts, lists and tuples."""
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tensors(item, function) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, function) for item in value)
    return value


def _first_tensor(value: object) -> torch.Tensor | None:
    """Return the first tensor in a nested batch structure."""
    if isinstance(value, torch.Tensor):
        return value
    items = value.values() if isinstance(value, dict) else value
    if isinstance(value, (dict, list, tuple)):
        for item in items:
            if (tensor := _first_tensor(item)) is not None:
                return tensor
    return None


def _effective_preproc(preproc: Callable | None) -> Callable | None:
    """Treat the base model's identity preprocessing as no preprocessing."""
    return None if preproc is None or preproc is ModelABC.preproc else preproc


def _callable_name(function: Callable | None) -> str:
    """Name a preprocessing callable for messages."""
    if function is None:
        return "None"
    return getattr(function, "__qualname__", type(function).__qualname__)


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Set random seeds used by Python, NumPy, and PyTorch."""
    random.seed(seed)
    # Seed the global NumPy stream used by caller transforms and DataLoader workers.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _configure_cudnn(deterministic=deterministic)


def _configure_cudnn(*, deterministic: bool, benchmark: bool = False) -> None:
    """Select deterministic or benchmarked cuDNN kernels; otherwise leave as is."""
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif benchmark:
        torch.backends.cudnn.benchmark = True


class Trainer:
    """General-purpose training loop for TIAToolbox models.

    Each history row records ``epoch``, the first optimizer group's ``lr`` at
    the start of the epoch, ``train_``/``val_`` metrics and ``epoch_time``, the
    wall-clock seconds spent in the training and validation passes.
    ``callbacks`` are called as ``callback(trainer, epoch_metrics)`` after each
    epoch's history row and checkpoint are written. With ``export``, ``fit``
    writes ``model_config.yaml`` for :func:`load_trained_model`, pointing at the
    best weights (or at the last checkpoint when best weights are not saved).
    """

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
        export: ModelExport | None = None,
        callbacks: Sequence[EpochCallback] | None = None,
    ) -> None:
        """Initialize :class:`Trainer`."""
        self.config = config or TrainerConfig()
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        self.export = export
        if export is not None and not (
            self.checkpoint_config.save_best or self.checkpoint_config.save_last
        ):
            msg = "`export` needs saved weights: enable `save_best` or `save_last`."
            raise ValueError(msg)
        self.callbacks = list(callbacks or [])

        if self.config.seed is not None:
            set_seed(self.config.seed)
        _configure_cudnn(
            deterministic=self.config.deterministic,
            benchmark=self.config.cudnn_benchmark,
        )

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = resolve_trainer_device(self.config.device)
        self.model = model.to(self.device)
        if self.config.channels_last:
            # ``ModelABC.to`` only moves devices, so convert with ``nn.Module.to``.
            torch.nn.Module.to(self.model, memory_format=torch.channels_last)

        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler
        if (
            isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            and self.config.scheduler_interval != "epoch"
        ):
            msg = "ReduceLROnPlateau requires scheduler_interval='epoch'."
            raise ValueError(msg)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self._data_seed = (
            self.config.seed if self.config.seed is not None else random.getrandbits(32)
        )
        self._seed_loaders()
        self._configure_amp()
        self._validate_monitor_configuration()
        self._check_preprocessing()
        self._check_classifier_head()

        self.history: list[dict[str, float]] = []
        self.best_monitor_value = self._initial_monitor_value()
        self.best_epoch = -1
        self.patience_counter = 0

    def _seed_loaders(self: Trainer) -> None:
        """Give loaders and samplers their own generators, seeded from the run."""
        for index, loader in enumerate((self.train_loader, self.val_loader)):
            if loader is None:
                continue
            # Isolate worker-startup draws from sampling and model randomness.
            worker_generator = torch.Generator()
            if loader.generator is not None:
                worker_generator.set_state(loader.generator.get_state())
            else:
                worker_generator.manual_seed(self._data_seed + index)
            loader.generator = worker_generator
            if (
                hasattr(loader.sampler, "generator")
                and loader.sampler.generator is None
            ):
                loader.sampler.generator = torch.Generator().manual_seed(
                    self._data_seed + index
                )

    def _configure_amp(self: Trainer) -> None:
        """Resolve autocast and create a loss scaler for float16 only."""
        self.use_amp = resolve_trainer_amp(
            amp=self.config.amp, device=self.device, dtype=self.config.amp_dtype
        )
        self._amp_dtype = getattr(torch, self.config.amp_dtype)
        if (
            self.use_amp
            and self._amp_dtype == torch.bfloat16
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            msg = "This GPU does not support bfloat16; use amp_dtype='float16'."
            raise ValueError(msg)
        # bfloat16 shares float32's exponent range, so needs no loss scaling.
        self.grad_scaler = GradScaler(
            self.device.type,
            enabled=self.use_amp and self._amp_dtype == torch.float16,
        )

    def _set_loader_epoch(
        self, loader: DataLoader, epoch: int, *, training: bool
    ) -> None:
        """Update epoch-aware samplers and datasets, including subset wrappers."""
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        for dataset in _inner_datasets(loader.dataset):
            if isinstance(dataset, _TrainingDataset):
                dataset.set_epoch(
                    epoch * 2 + int(not training), self._data_seed, training=training
                )
            elif hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

    def _check_classifier_head(self: Trainer) -> None:
        """Raise when a classifier's inference head disagrees with the task.

        ``CNNModel`` and ``TimmModel`` turn logits into probabilities with
        softmax or sigmoid according to their ``output_mode``, so a head
        trained with BCE must say so, or engines return the wrong
        probabilities. ``ModelExport`` must also rebuild the same head.
        """
        model = unwrap_model(self.model)
        output_mode = getattr(model, "output_mode", None)
        num_classes = getattr(model, "num_classes", None)
        if (
            output_mode is None
            or num_classes is None
            or not isinstance(self.task, ClassificationTask)
        ):
            return
        expected = self.task.required_output_mode(num_classes)
        if expected != output_mode:
            msg = (
                f"{type(model).__name__} has output_mode='{output_mode}', but the "
                f"task trains a '{expected}' head, so inference engines would "
                "apply the wrong activation to its logits. Build the model with "
                f"output_mode='{expected}'."
            )
            raise ValueError(msg)
        if self.export is not None:
            self.export.check_model(model)

    def _check_preprocessing(self: Trainer) -> None:
        """Warn when datasets preprocess differently from inference engines.

        Engines apply a TIAToolbox model's ``preproc_func`` to every patch, so
        training datasets should apply the same function as their ``preproc``.
        """
        model = unwrap_model(self.model)
        if not isinstance(model, ModelABC):
            return
        try:
            expected = _effective_preproc(model.preproc_func)
        except AttributeError:  # Subclass skipped ModelABC.__init__.
            expected = _effective_preproc(model.preproc)
        for loader in (self.train_loader, self.val_loader):
            if loader is None:
                continue
            for dataset in _inner_datasets(loader.dataset):
                if not isinstance(dataset, _TrainingDataset):
                    continue
                actual = _effective_preproc(dataset.preproc)
                if actual != expected:
                    logger.warning(
                        "Inference engines preprocess patches with %s.preproc_func "
                        "(%s), but %s uses preproc=%s. Pass "
                        "`preproc=model.preproc_func` to training datasets so the "
                        "model sees the same inputs in training and inference.",
                        type(model).__name__,
                        _callable_name(expected),
                        type(dataset).__name__,
                        _callable_name(actual),
                    )
                    return

    def _loader_generators(self) -> dict[str, torch.Generator]:
        """Find standard loader and sampler generators for checkpointing."""
        generators = {}
        for name, loader in (("train", self.train_loader), ("val", self.val_loader)):
            if loader is not None:
                for part, owner in (("loader", loader), ("sampler", loader.sampler)):
                    generator = getattr(owner, "generator", None)
                    if generator is not None:
                        generators[f"{name}_{part}"] = generator
        return generators

    def _random_state(self) -> dict:
        """Capture epoch-boundary random state using weights-only-safe values."""
        numpy_state = np.random.get_state()  # noqa: NPY002 - capture the global RNG
        return {
            "python": random.getstate(),
            "numpy": (
                numpy_state[0],
                torch.from_numpy(numpy_state[1].astype(np.int64)),
                *numpy_state[2:],
            ),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "generators": {
                key: generator.get_state()
                for key, generator in self._loader_generators().items()
            },
        }

    def _restore_random_state(self, state: dict) -> None:
        """Restore stochastic continuation after all checkpoint loading work."""
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        np.random.set_state(  # noqa: NPY002 - restore the global RNG
            (
                numpy_state[0],
                numpy_state[1].cpu().numpy().astype(np.uint32),
                *numpy_state[2:],
            )
        )
        torch.set_rng_state(state["torch"].cpu())
        if torch.cuda.is_available() and state["cuda"]:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
        generators = self._loader_generators()
        for key, value in state.get("generators", {}).items():
            if key in generators:
                generators[key].set_state(value.cpu())

    def _initial_monitor_value(self: Trainer) -> float:
        """Return initial sentinel based on monitor mode."""
        return float("inf") if self.config.monitor_mode == "min" else -float("inf")

    @staticmethod
    def _extract_batch(
        batch: dict | list | tuple,
    ) -> tuple[torch.Tensor, object]:
        """Extract image tensor and target object from a dataloader batch."""
        images = None
        targets = None

        if isinstance(batch, dict):
            images = batch.get("image")
            targets = batch.get("target", batch.get("label"))
        elif isinstance(batch, (list, tuple)) and len(batch) >= _MIN_BATCH_ITEMS:
            images = batch[0]
            targets = batch[1]

        if _first_tensor(images) is None or targets is None:
            msg = (
                "Batch must provide `image` tensors (one tensor, or a dict, list or "
                "tuple of tensors for multi-input models) and `target`/`label` entries."
            )
            raise ValueError(msg)

        # Accept NHWC tensors for convenience.
        if (
            isinstance(images, torch.Tensor)
            and images.ndim == _BATCH_IMAGE_DIMS
            and images.shape[1] not in {1, 3, 4}
            and images.shape[-1] in {1, 3, 4}
        ):
            images = images.permute(0, 3, 1, 2).contiguous()

        return images, targets

    def _move_to_device(self: Trainer, value: object) -> object:
        """Transfer nested batch tensors, overlapping copies from pinned memory."""
        return _map_tensors(
            value, lambda tensor: tensor.to(self.device, non_blocking=True)
        )

    def _prepare_images(self: Trainer, images: object) -> object:
        """Move model inputs to the device; a single image tensor becomes float.

        Structured inputs keep their dtypes, as they may hold non-image values.
        """
        if not isinstance(images, torch.Tensor):
            return self._move_to_device(images)
        images = images.to(self.device, non_blocking=True).float()
        if self.config.channels_last and images.ndim == _BATCH_IMAGE_DIMS:
            images = images.contiguous(memory_format=torch.channels_last)
        return images

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

        if self.config.monitor.startswith("val_") and not any(
            key.startswith("val_") for key in metrics
        ):
            return None

        msg = (
            f"Monitor metric `{self.config.monitor}` not found in epoch metrics: "
            f"{sorted(metrics.keys())}."
        )
        raise KeyError(msg)

    def _optimizer_step(self: Trainer) -> None:
        """Perform one optimizer step, including AMP and gradient clipping."""
        if self.config.grad_clip_norm is not None:
            if self.grad_scaler.is_enabled():
                self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.grad_clip_norm,
            )

        if self.grad_scaler.is_enabled():
            previous_scale = self.grad_scaler.get_scale()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            stepped = self.grad_scaler.get_scale() >= previous_scale
        else:
            self.optimizer.step()
            stepped = True

        self.optimizer.zero_grad(set_to_none=True)
        if stepped and self.config.scheduler_interval == "step":
            self._step_scheduler(None)

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

    def _forward(
        self: Trainer,
        batch_images: object,
        batch_targets: object,
        loss_scales: dict[str, float] | None,
        *,
        training: bool,
    ) -> tuple[object, object, dict[str, torch.Tensor]]:
        """Run one batch; with ``loss_scales``, backpropagate its window share."""
        images = self._prepare_images(batch_images)
        targets = self._move_to_device(batch_targets)
        # In eval mode BatchNorm's running statistics can push activations far
        # past float16's range even when training passes stay finite, and one
        # overflowing batch makes the epoch's validation loss non-finite.
        # Inference engines run in float32 too, so validation does the same.
        # bfloat16 has float32's range and keeps autocast in both modes.
        use_autocast = self.use_amp and (training or self._amp_dtype != torch.float16)
        with torch.set_grad_enabled(training):
            with autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=use_autocast,
            ):
                output = self.model(images)
                components = self.task.loss_components(output, targets)
            if loss_scales is not None:
                scaled_loss = sum(
                    value * loss_scales[key]
                    for key, value in components.items()
                    if key in loss_scales
                )
                if self.grad_scaler.is_enabled():
                    scaled_loss = self.grad_scaler.scale(scaled_loss)
                scaled_loss.backward()
        return output, targets, components

    def _log_batch_loss(
        self,
        components: dict[str, torch.Tensor],
        step_index: int,
        *,
        training: bool,
    ) -> None:
        """Log the detached batch loss at the configured step interval."""
        if (
            self.config.log_every_n_steps > 0
            and step_index % self.config.log_every_n_steps == 0
        ):
            logger.info(
                "%s step %d: loss=%.6f",
                "train" if training else "val",
                step_index,
                float(sum(components.values()).detach()),
            )

    def _run_window(
        self,
        window: list[dict | list | tuple],
        loss_totals: dict[str, torch.Tensor],
        loss_counts: dict[str, int],
        step_index: int,
        *,
        training: bool,
    ) -> tuple[int, int]:
        """Accumulate one window and return its sample count and final step index."""
        total_samples = 0
        # Keep only input batches, never a window of autograd graphs. Counts
        # are available on the CPU before any forward/backward passes.
        prepared = [self._extract_batch(batch) for batch in window]
        batch_sizes = [int(_first_tensor(images).shape[0]) for images, _ in prepared]
        normalizers = [
            self.task.loss_normalizers(targets, batch_size)
            for (_, targets), batch_size in zip(prepared, batch_sizes, strict=True)
        ]
        window_counts = {
            key: sum(counts[key] for counts in normalizers) for key in normalizers[0]
        }
        supervised = any(window_counts.values())
        for (batch_images, batch_targets), counts, batch_size in zip(
            prepared, normalizers, batch_sizes, strict=True
        ):
            step_index += 1
            total_samples += batch_size
            loss_scales = None
            if training and supervised:
                loss_scales = {
                    key: count / window_counts[key]
                    for key, count in counts.items()
                    if window_counts[key]
                }
            output, targets, components = self._forward(
                batch_images, batch_targets, loss_scales, training=training
            )
            if components.keys() != counts.keys():
                msg = "Loss components and normalizers must have matching keys."
                raise ValueError(msg)
            self.task.update_epoch_state(
                _map_tensors(output, torch.Tensor.detach),
                _map_tensors(targets, torch.Tensor.detach),
            )
            for key, value in components.items():
                loss_totals[key] = (
                    loss_totals.get(key, 0.0) + value.detach().double() * counts[key]
                )
                loss_counts[key] = loss_counts.get(key, 0) + counts[key]

            self._log_batch_loss(components, step_index, training=training)
        if training and supervised:
            self._optimizer_step()
        return total_samples, step_index

    def _run_epoch(
        self: Trainer,
        loader: DataLoader,
        *,
        training: bool,
    ) -> dict[str, float]:
        """Run an epoch, reducing losses by supervision and metrics by counts."""
        self.model.train(mode=training)
        mode_name = "train" if training else "val"
        loss_totals: dict[str, torch.Tensor] = {}
        loss_counts: dict[str, int] = {}
        total_samples = 0
        step_index = 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        self.task.reset_epoch_state(training=training)

        batches = iter(
            tqdm(
                loader,
                desc=mode_name,
                leave=False,
                disable=not self.config.progress_bar,
            )
        )
        window_size = self.config.grad_accum_steps if training else 1
        while window := list(islice(batches, window_size)):
            window_samples, step_index = self._run_window(
                window, loss_totals, loss_counts, step_index, training=training
            )
            total_samples += window_samples

        if total_samples == 0:
            msg = f"`{mode_name}` dataloader yielded zero samples."
            raise ValueError(msg)
        component_means = {
            key: float(total / max(loss_counts[key], 1))
            for key, total in loss_totals.items()
        }
        epoch_metrics = dict(self.task.compute_epoch_metrics())
        if len(component_means) > 1:
            epoch_metrics.update(
                {f"{key}_loss": value for key, value in component_means.items()}
            )
        epoch_metrics["loss"] = sum(component_means.values())
        if not np.isfinite(epoch_metrics["loss"]):
            logger.warning(
                "The %s loss is not finite; one non-finite batch loss is enough. "
                "Check the data, the learning rate and the precision settings.",
                mode_name,
            )
        # Plain floats keep history and checkpoints loadable with weights_only,
        # whatever numeric types (NumPy, 0-d tensors) a task reports.
        return {key: float(value) for key, value in epoch_metrics.items()}

    def _build_checkpoint_state(self: Trainer, epoch: int) -> dict[str, Any]:
        """Build serializable training state."""
        trainer_config = asdict(self.config)
        trainer_config["device"] = str(trainer_config["device"])
        trainer_config["output_dir"] = str(trainer_config["output_dir"])
        return {
            "epoch": epoch,
            "model_state_dict": extract_model_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": None
            if self.scheduler is None
            else self.scheduler.state_dict(),
            "grad_scaler_state_dict": self.grad_scaler.state_dict()
            if self.grad_scaler.is_enabled()
            else None,
            "best_monitor_value": self.best_monitor_value,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "trainer_config": trainer_config,
            "patience_counter": self.patience_counter,
            "random_state": self._random_state(),
            "data_seed": self._data_seed,
        }

    def _save_model_config(self: Trainer) -> None:
        """Write the inference configuration next to the weights it names."""
        config = self.checkpoint_config
        weights = (
            config.best_weights_filename if config.save_best else config.last_filename
        )
        self.export.save(
            self.output_dir / config.model_config_filename, self.model, weights
        )

    def _carry_over_best_weights(self: Trainer, previous_dir: Path) -> None:
        """Copy the resumed run's best weights when resuming into a new directory.

        The best score is restored with the checkpoint, so the matching weights
        must stay beside it even if training never improves after resuming.
        """
        if not self.checkpoint_config.save_best:
            return
        name = self.checkpoint_config.best_weights_filename
        previous, current = previous_dir / name, self.output_dir / name
        if previous.is_file() and not current.exists():
            shutil.copy2(previous, current)

    def _resume_from_checkpoint(self: Trainer, checkpoint_path: str | Path) -> int:
        """Restore trainer state from a checkpoint and return start epoch."""
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        saved_config = checkpoint.get("trainer_config", {})
        for key in ("monitor", "monitor_mode", "scheduler_interval"):
            if key in saved_config and saved_config[key] != getattr(self.config, key):
                msg = (
                    f"Cannot resume with a different `{key}`; "
                    "load model weights to start a new run."
                )
                raise ValueError(msg)

        model_state_dict = checkpoint["model_state_dict"]
        load_model_state_dict(self.model, model_state_dict, strict=True)

        if checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = checkpoint.get("grad_scaler_state_dict")
        if self.grad_scaler.is_enabled() and scaler_state is not None:
            self.grad_scaler.load_state_dict(scaler_state)

        self.best_monitor_value = checkpoint.get(
            "best_monitor_value", self._initial_monitor_value()
        )
        self.best_epoch = int(checkpoint.get("best_epoch", -1))
        self.history = list(checkpoint.get("history", []))
        self.patience_counter = int(checkpoint.get("patience_counter", 0))
        if "patience_counter" not in checkpoint:
            self.patience_counter = sum(
                row.get("epoch", 0) > self.best_epoch and self.config.monitor in row
                for row in self.history
            )
        self._data_seed = int(checkpoint.get("data_seed", self._data_seed))
        self._carry_over_best_weights(Path(checkpoint_path).parent)

        if "random_state" in checkpoint:
            self._restore_random_state(checkpoint["random_state"])

        start_epoch = int(checkpoint.get("epoch", 0))
        logger.info(
            "Resumed training from checkpoint `%s` at epoch %d.",
            checkpoint_path,
            start_epoch,
        )
        return start_epoch

    def evaluate(self: Trainer, loader: DataLoader | None = None) -> dict[str, float]:
        """Run one evaluation pass and return unprefixed epoch metrics.

        ``loader`` defaults to the validation loader. Epoch-aware datasets and
        samplers are keyed by the number of completed epochs, so repeated calls
        on the same loader see the same deterministic preprocessing.
        """
        loader = self.val_loader if loader is None else loader
        if loader is None:
            msg = "`evaluate` requires a loader when the trainer has no `val_loader`."
            raise ValueError(msg)
        self._set_loader_epoch(loader, len(self.history), training=False)
        return self._run_epoch(loader, training=False)

    def _update_best_checkpoint(self, epoch: int, monitor_value: float | None) -> None:
        """Update early-stopping state and save weights when the monitor improves."""
        if monitor_value is None:
            return
        if self._is_improved(monitor_value):
            self.best_monitor_value = monitor_value
            self.best_epoch = epoch
            self.patience_counter = 0
            if self.checkpoint_config.save_best:
                best_path = (
                    self.output_dir / self.checkpoint_config.best_weights_filename
                )
                save_model_weights(self.model, best_path)
        else:
            self.patience_counter += 1

    def _save_last_checkpoint(self, epoch: int, *, final: bool) -> None:
        """Save on configured intervals and always at the final epoch."""
        if self.checkpoint_config.save_last and (
            final or epoch % self.checkpoint_config.save_last_every == 0
        ):
            save_checkpoint(
                self._build_checkpoint_state(epoch),
                self.output_dir / self.checkpoint_config.last_filename,
            )

    # Keep validation, best-weight tracking, checkpointing, and stopping in epoch order.
    def fit(
        self: Trainer,
        resume_from: str | Path | None = None,
    ) -> list[dict[str, float]]:
        """Train or resume at an epoch boundary and return per-epoch history.

        Preset augmentations and standard sampler generators resume reproducibly
        with unchanged data and loader settings. Custom datasets/transforms with
        private mutable state must manage that state themselves.
        """
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self._resume_from_checkpoint(resume_from)
        if self.export is not None:
            self._save_model_config()

        patience = self.config.early_stopping_patience
        for epoch_idx in range(start_epoch, self.config.max_epochs):
            if patience is not None and self.patience_counter >= patience:
                logger.info(
                    "Early stopping after epoch %d: `%s` did not improve for %d "
                    "epochs (best epoch %d).",
                    epoch_idx,
                    self.config.monitor,
                    patience,
                    self.best_epoch,
                )
                break
            epoch_start = time.perf_counter()
            epoch_metrics: dict[str, float] = {
                "epoch": float(epoch_idx + 1),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            self._set_loader_epoch(self.train_loader, epoch_idx, training=True)
            train_metrics = self._run_epoch(self.train_loader, training=True)
            epoch_metrics.update(
                {f"train_{key}": value for key, value in train_metrics.items()}
            )

            if self.val_loader is not None and (
                (epoch_idx + 1) % self.config.val_interval == 0
            ):
                self._set_loader_epoch(self.val_loader, epoch_idx, training=False)
                val_metrics = self._run_epoch(self.val_loader, training=False)
                epoch_metrics.update(
                    {f"val_{key}": value for key, value in val_metrics.items()}
                )
            epoch_metrics["epoch_time"] = time.perf_counter() - epoch_start

            monitor_value = self._resolve_monitor_value(epoch_metrics)
            self._update_best_checkpoint(epoch_idx + 1, monitor_value)

            if self.config.scheduler_interval == "epoch":
                self._step_scheduler(monitor_value)
            self.history.append(epoch_metrics)

            final = epoch_idx + 1 == self.config.max_epochs or (
                patience is not None and self.patience_counter >= patience
            )
            self._save_last_checkpoint(epoch_idx + 1, final=final)

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

            for callback in self.callbacks:
                callback(self, epoch_metrics)

        return self.history
