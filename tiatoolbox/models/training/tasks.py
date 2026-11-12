"""Task adapters for model training."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional

_CLASSIFICATION_DIMS = 2
_DENSE_LABEL_DIMS = 3
_DENSE_LOGIT_DIMS = 4
_MIN_MULTICLASS_CHANNELS = 2
_BINARY_THRESHOLD = 0.5

ClassificationTargetMode = Literal["single_label", "binary", "multi_label"]
DenseLossMode = Literal["cross_entropy", "bce_with_logits", "mse", "l1", "smooth_l1"]
DenseTargetMode = Literal["multiclass", "binary", "regression"]
DenseMetricName = Literal["dice", "iou", "mae", "mse"]
NestedKey = str | int | tuple[str | int, ...]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Named dense losses fix the target semantics; callable losses declare theirs.
_DENSE_LOSS_TARGET_MODES: dict[str, DenseTargetMode] = {
    "cross_entropy": "multiclass",
    "bce_with_logits": "binary",
    "mse": "regression",
    "l1": "regression",
    "smooth_l1": "regression",
}
_REGRESSION_LOSSES = {
    "mse": functional.mse_loss,
    "l1": functional.l1_loss,
    "smooth_l1": functional.smooth_l1_loss,
}


def _validate_loss_callable(loss: LossFunction, target_mode: str | None) -> None:
    """Require explicit target semantics and an unreduced criterion."""
    if target_mode is None:
        msg = "Callable losses require an explicit `target_mode`."
        raise ValueError(msg)
    if getattr(loss, "reduction", "none") != "none":
        msg = "Callable losses must return unreduced values; use reduction='none'."
        raise ValueError(msg)


def _validate_class_indices(
    targets: torch.Tensor,
    ignore_index: int,
    num_classes: int,
    *,
    kind: str,
) -> None:
    """Raise when supervised targets fall outside ``[0, num_classes)``.

    Reading the result synchronises accelerator tensors, so the trainer passes
    targets that are still on the CPU.
    """
    invalid = (targets < 0) | (targets >= num_classes)
    if bool((invalid & (targets != ignore_index)).any()):
        msg = (
            f"{kind} targets contain class indices outside the valid range "
            f"[0, {num_classes - 1}] and not equal to ignore_index `{ignore_index}`."
        )
        raise ValueError(msg)


def _call_loss(
    loss: LossFunction,
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a criterion on zero-filled targets and average supervised values."""
    if isinstance(loss, nn.Module):
        loss.to(device=logits.device)
    # The task owns ignore_index; criteria only ever see in-range targets. The
    # criterion also runs on fully ignored batches, which avoids a device sync.
    # Row masks of soft-label targets broadcast over their class columns.
    row_valid = valid.reshape(*valid.shape, *[1] * (targets.ndim - valid.ndim))
    safe_targets = torch.where(row_valid, targets, torch.zeros_like(targets))
    values = loss(logits, safe_targets)
    if not isinstance(values, torch.Tensor) or values.shape != valid.shape:
        msg = (
            "Callable losses must return one unreduced value per supervised "
            "target (per element, or per sample for soft labels); "
            "use reduction='none'."
        )
        raise ValueError(msg)
    return _masked_mean(values, valid)


def _is_soft_label(targets: torch.Tensor) -> bool:
    """Return whether single-label targets are ``(N, C)`` class probabilities."""
    return (
        targets.is_floating_point()
        and targets.ndim == _CLASSIFICATION_DIMS
        and targets.shape[1] >= _MIN_MULTICLASS_CHANNELS
    )


def _select_nested_value(value: object, key: NestedKey, *, kind: str) -> object:
    """Follow a key, an index or a tuple path of them through nested containers."""
    key_path = key if isinstance(key, tuple) else (key,)
    current = value
    for part in key_path:
        if isinstance(current, dict):
            found = part in current
        elif isinstance(current, (list, tuple)):
            found = isinstance(part, int) and -len(current) <= part < len(current)
        else:
            msg = (
                f"{kind} path `{key_path}` expects nested dictionaries or sequences, "
                f"but encountered `{type(current).__name__}`."
            )
            raise ValueError(msg)  # noqa: TRY004 - preserve validation API
        if not found:
            msg = f"{kind} path `{key_path}` could not be resolved."
            raise ValueError(msg)
        current = current[part]
    return current


def _select_output(output: object, key: NestedKey | None) -> torch.Tensor:
    """Select a tensor from a model's output; a bare tensor is used as it is."""
    if isinstance(output, torch.Tensor):
        return output
    if key is None:
        msg = (
            f"The model returned a `{type(output).__name__}`, so the task cannot tell "
            "which output to use. Set `output_key` to a dictionary key, a sequence "
            "index, or a tuple path of them."
        )
        raise ValueError(msg)
    selected = _select_nested_value(output, key, kind="Output")
    if not isinstance(selected, torch.Tensor):
        msg = f"`output_key` {key!r} does not select a tensor output."
        raise ValueError(msg)  # noqa: TRY004 - preserve validation API
    return selected


class TrainingTaskABC(ABC):
    """Define the objective and metrics independently of model/data loading.

    Implement ``loss_components`` to return the objective as one or more named,
    weighted mean losses. Override ``loss_normalizers`` with matching keys when
    a mean is not over samples, for example over valid pixels, so the trainer
    can weight uneven microbatches. Override ``validate_targets`` to check each
    batch's targets against the model's output. Metrics are optional: accumulate
    detached, bounded statistics in ``update_epoch_state`` and reduce them in
    ``compute_epoch_metrics``. ``compute_loss`` and ``compute_metrics`` are
    single-batch conveniences built on these hooks; the trainer does not call
    them. Task state is not saved in checkpoints; keep criteria stateless or
    manage custom state explicitly.
    """

    def __init__(
        self: TrainingTaskABC,
        *,
        output_key: NestedKey | None = None,
        ignore_index: int = -100,
    ) -> None:
        """Initialize :class:`TrainingTaskABC`."""
        self.output_key = output_key
        self.ignore_index = ignore_index

    def select_output(self: TrainingTaskABC, output: object) -> torch.Tensor:
        """Select a tensor output for loss and metric computation.

        A tensor is used as it is, which also suits models that return several
        outputs only in training mode. Other outputs need ``output_key``: a
        dictionary key, a sequence index, or a tuple path of them.
        """
        return _select_output(output, self.output_key)

    # This default hook is overridden by tasks that keep per-epoch state.
    def reset_epoch_state(  # skipcq: PYL-R0201
        self: TrainingTaskABC,
        *,
        training: bool,  # noqa: ARG002 - hook signature
    ) -> None:
        """Do nothing by default; override to reset task-specific epoch state."""
        return

    # Keep the instance hook signature for stateful task implementations.
    def update_epoch_state(  # skipcq: PYL-R0201
        self: TrainingTaskABC,
        output: object,  # noqa: ARG002 - hook signature
        targets: object,  # noqa: ARG002 - hook signature
    ) -> None:
        """Do nothing by default; override to accumulate detached batch state."""
        return

    # Stateful tasks override this instance hook to reduce their statistics.
    def compute_epoch_metrics(  # skipcq: PYL-R0201
        self: TrainingTaskABC,
    ) -> dict[str, float]:
        """Return any task-specific metrics accumulated across the epoch."""
        return {}

    @abstractmethod
    def loss_components(
        self, output: object, targets: object
    ) -> dict[str, torch.Tensor]:
        """Return named, weighted mean losses whose sum is the training objective.

        With several components, the trainer also reports each epoch mean as
        ``<name>_loss`` next to the total ``loss``.
        """

    # Subclasses use task-specific state to normalize their loss components.
    def loss_normalizers(  # skipcq: PYL-R0201
        self,
        targets: object,  # noqa: ARG002 - hook signature
        batch_size: int,
    ) -> dict[str, int]:
        """Return each loss component's denominator, computed before the forward.

        The default suits a single ``"loss"`` component averaged over samples.
        Tasks with masked elements or independently averaged components must
        override this with the same keys as ``loss_components``. This lets the
        trainer weight uneven microbatches without retaining graphs.
        """
        return {"loss": batch_size}

    # Tasks whose targets depend on the output's shape override this hook.
    def validate_targets(  # skipcq: PYL-R0201
        self,
        output: object,  # noqa: ARG002 - hook signature
        targets: object,  # noqa: ARG002 - hook signature
    ) -> None:
        """Raise when a batch's targets cannot supervise the model's output.

        The trainer calls this once per batch with targets still on the CPU, so
        read only the output's shape: reading its values would synchronise with
        the device. Shape errors are left to ``loss_components``.
        """
        return

    def compute_loss(self, output: object, targets: object) -> torch.Tensor:
        """Return the objective for one batch: the sum of its loss components."""
        self.validate_targets(output, targets)
        return sum(self.loss_components(output, targets).values())

    def compute_metrics(self, output: object, targets: object) -> dict[str, float]:
        """Return epoch-style metrics for a single batch.

        The epoch hooks run on a shallow copy of the task, so a running epoch's
        statistics are left untouched provided ``reset_epoch_state`` assigns
        fresh state rather than clearing shared containers in place.
        """
        self.validate_targets(output, targets)
        task = copy.copy(self)
        task.reset_epoch_state(training=False)
        task.update_epoch_state(output, targets)
        return task.compute_epoch_metrics()


@dataclass(frozen=True)
class DenseHeadSpec:
    """Select and supervise one dense output.

    ``loss`` is a named loss or a callable returning unreduced losses with the
    prepared target shape. A named loss fixes the target semantics:
    ``cross_entropy`` is multiclass, ``bce_with_logits`` is binary and the rest
    are regression. A callable declares them with ``target_mode``, which is only
    for callables. Losses are averaged over valid target elements, then
    multiplied by ``loss_weight``. Class weights multiply elements, not the
    denominator.
    ``output_key`` selects the head's output from a model returning several: a
    dictionary key, a sequence index, or a tuple path of them. Without one, a
    head reads the dictionary entry named after it, and other containers raise; a
    tensor output is used as it is. ``target_key`` selects the head's target in
    the same way and also defaults to the head name for dictionaries.
    ``channel_slice`` selects output channels while preserving the channel
    dimension.
    ``metrics=None`` chooses Dice/IoU for classification and MAE/MSE for regression;
    an empty tuple disables metrics. ``metric_classes`` restricts the mean
    Dice/IoU to the listed classes (or binary channels), for example ``(1, 2)``
    to leave out background class 0, and also reports each as
    ``<metric>_<class>``. A listed class that the usual rules leave out of the
    mean (a multiclass class absent from prediction and target, or a wholly
    ignored binary channel) scores NaN.
    """

    name: str
    loss: DenseLossMode | LossFunction
    target_mode: DenseTargetMode | None = None
    target_key: NestedKey | None = None
    output_key: NestedKey | None = None
    channel_slice: int | slice | tuple[int, ...] | list[int] | None = None
    loss_weight: float = 1.0
    metrics: tuple[DenseMetricName, ...] | None = None
    metric_classes: tuple[int, ...] | None = None


class StructuredDenseTask(TrainingTaskABC):
    """Train dense heads using independent valid-element loss normalization.

    ``heads`` must have unique names and positive weights. ``ignore_index``
    applies to every head. ``prefix_head_metrics`` controls
    score prefixes for a single head; multiple heads always retain prefixes.
    Each head is one loss component, so with several heads the history also
    reports each head's weighted mean loss as ``<head>_loss``.
    """

    def __init__(
        self: StructuredDenseTask,
        heads: list[DenseHeadSpec],
        *,
        ignore_index: int = -100,
        prefix_head_metrics: bool = True,
    ) -> None:
        """Initialize :class:`StructuredDenseTask`."""
        super().__init__(ignore_index=ignore_index)
        if not heads:
            msg = "`heads` must contain at least one dense head specification."
            raise ValueError(msg)
        if len({head.name for head in heads}) != len(heads):
            msg_0 = "Dense head names must be unique."
            raise ValueError(msg_0)
        for head in heads:
            if callable(head.loss):
                _validate_loss_callable(head.loss, head.target_mode)
                if head.target_mode not in {"multiclass", "binary", "regression"}:
                    msg = f"Unsupported dense target mode `{head.target_mode}`."
                    raise ValueError(msg)
            elif head.loss not in _DENSE_LOSS_TARGET_MODES:
                msg = f"Unsupported dense loss `{head.loss}` for head `{head.name}`."
                raise ValueError(msg)
            elif head.target_mode is not None:
                msg = (
                    f"Dense head `{head.name}` uses `loss='{head.loss}'`, which "
                    f"supervises `{_DENSE_LOSS_TARGET_MODES[head.loss]}` targets. "
                    "`target_mode` is only for callable losses."
                )
                raise ValueError(msg)
            if head.loss_weight <= 0:
                msg = (
                    f"Dense head `{head.name}` must have a positive `loss_weight`, "
                    f"got {head.loss_weight}."
                )
                raise ValueError(msg)
            classes = head.metric_classes
            if classes is not None and (
                not classes or len(set(classes)) != len(classes) or min(classes) < 0
            ):
                msg = (
                    f"Dense head `{head.name}` needs `metric_classes` to be unique, "
                    "non-negative class indices."
                )
                raise ValueError(msg)
        self.heads = heads
        self.prefix_head_metrics = prefix_head_metrics
        self._epoch_states: dict[str, _MetricState] = {}

    @staticmethod
    def _select_head_output(output: object, head: DenseHeadSpec) -> torch.Tensor:
        """Resolve one head tensor from raw model output."""
        key = head.output_key
        if key is None and isinstance(output, dict):
            if head.name not in output:
                msg = (
                    f"Dense head `{head.name}` reads the output named after it, but "
                    f"the outputs are named {list(output)}. Rename the head or set "
                    "its `output_key`."
                )
                raise ValueError(msg)
            key = head.name
        head_output = _select_output(output, key)

        if head.channel_slice is None:
            return head_output

        channel_slice = head.channel_slice
        if isinstance(channel_slice, int):
            return head_output[:, channel_slice, ...].unsqueeze(1)
        return head_output[:, channel_slice, ...]

    @staticmethod
    def _select_head_target(
        targets: object,
        head: DenseHeadSpec,
    ) -> torch.Tensor:
        """Resolve one head target tensor from raw task targets."""
        head_target: object
        if head.target_key is not None:
            head_target = _select_nested_value(targets, head.target_key, kind="Target")
        elif isinstance(targets, dict):
            if head.name not in targets:
                msg = (
                    f"Dense head `{head.name}` reads the target named after it, but "
                    f"the targets are named {list(targets)}. Rename the head or "
                    "set its `target_key`."
                )
                raise ValueError(msg)
            head_target = targets[head.name]
        else:
            head_target = targets

        if not isinstance(head_target, torch.Tensor):
            msg = f"Dense head `{head.name}` requires tensor targets."
            raise ValueError(msg)  # noqa: TRY004 - preserve validation API
        return head_target

    @staticmethod
    def _head_target_mode(head: DenseHeadSpec) -> DenseTargetMode:
        """Return the target mode a callable head declares or a named loss fixes."""
        if callable(head.loss):
            return head.target_mode
        return _DENSE_LOSS_TARGET_MODES[head.loss]

    @staticmethod
    def _resolve_head_metrics(
        head: DenseHeadSpec,
        mode: DenseTargetMode,
    ) -> tuple[DenseMetricName, ...]:
        """Resolve the effective metric set for one head."""
        if head.metrics is not None:
            return head.metrics
        return ("mae", "mse") if mode == "regression" else ("dice", "iou")

    @staticmethod
    def _validate_dense_batch_and_spatial_shapes(
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        mode: str,
    ) -> None:
        """Validate common dense prediction batch and spatial dimensions."""
        if targets.shape[0] != logits.shape[0]:
            msg = (
                f"Dense {mode} targets batch size `{targets.shape[0]}` must "
                f"match logits batch size `{logits.shape[0]}`."
            )
            raise ValueError(msg)
        if tuple(targets.shape[-2:]) != tuple(logits.shape[-2:]):
            msg = (
                f"Dense {mode} target spatial shape `{tuple(targets.shape[-2:])}` "
                f"must match logits spatial shape `{tuple(logits.shape[-2:])}`."
            )
            raise ValueError(msg)

    def _prepare_multiclass_dense_tensors(
        self: StructuredDenseTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize multiclass dense logits and targets."""
        if (
            logits.ndim != _DENSE_LOGIT_DIMS
            or logits.shape[1] < _MIN_MULTICLASS_CHANNELS
        ):
            msg = (
                "Dense multiclass heads expect logits with shape `(N, C, H, W)` "
                "where `C >= 2`."
            )
            raise ValueError(msg)
        if targets.ndim == _DENSE_LOGIT_DIMS and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if targets.ndim != _DENSE_LABEL_DIMS:
            msg = (
                "Dense multiclass targets must have shape `(N, H, W)` "
                "or `(N, 1, H, W)`."
            )
            raise ValueError(msg)
        self._validate_dense_batch_and_spatial_shapes(
            logits,
            targets,
            mode="multiclass",
        )
        targets = targets.long()
        return logits, targets, targets != self.ignore_index

    def _prepare_binary_dense_tensors(
        self: StructuredDenseTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize dense binary BCE tensors and derive a valid-element mask."""
        if logits.ndim != _DENSE_LOGIT_DIMS:
            msg = "Dense binary heads expect logits with shape `(N, C, H, W)`."
            raise ValueError(msg)
        if targets.ndim == _DENSE_LABEL_DIMS and logits.shape[1] == 1:
            targets = targets.unsqueeze(1)
        if targets.ndim != _DENSE_LOGIT_DIMS:
            msg = "Dense binary targets must have shape `(N, H, W)` or `(N, C, H, W)`."
            raise ValueError(msg)
        if targets.shape != logits.shape:
            self._validate_dense_batch_and_spatial_shapes(
                logits,
                targets,
                mode="binary",
            )
            msg = (
                f"Dense binary target shape `{tuple(targets.shape)}` must match "
                f"logits shape `{tuple(logits.shape)}`."
            )
            raise ValueError(msg)
        valid_mask = targets != self.ignore_index
        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        return logits, safe_targets.float(), valid_mask

    def _prepare_regression_dense_tensors(
        self: StructuredDenseTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize dense regression tensors and derive a valid-element mask."""
        if logits.ndim != _DENSE_LOGIT_DIMS:
            msg = "Dense regression heads expect logits with shape `(N, C, H, W)`."
            raise ValueError(msg)
        if targets.ndim == _DENSE_LABEL_DIMS and logits.shape[1] == 1:
            targets = targets.unsqueeze(1)
        if targets.ndim != _DENSE_LOGIT_DIMS:
            msg = (
                "Dense regression targets must have shape `(N, H, W)` "
                "or `(N, C, H, W)`."
            )
            raise ValueError(msg)
        if targets.shape != logits.shape:
            self._validate_dense_batch_and_spatial_shapes(
                logits,
                targets,
                mode="regression",
            )
            msg = (
                f"Dense regression target shape `{tuple(targets.shape)}` must match "
                f"logits shape `{tuple(logits.shape)}`."
            )
            raise ValueError(msg)
        valid_mask = targets != self.ignore_index
        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        return logits.float(), safe_targets.float(), valid_mask

    def _prepare_head_tensors(
        self: StructuredDenseTask,
        head: DenseHeadSpec,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate and normalize one head's tensors for its target mode."""
        mode = self._head_target_mode(head)
        if mode == "multiclass":
            return self._prepare_multiclass_dense_tensors(logits, targets)
        if mode == "binary":
            return self._prepare_binary_dense_tensors(logits, targets)
        return self._prepare_regression_dense_tensors(logits, targets)

    def _compute_head_loss(
        self: StructuredDenseTask,
        head: DenseHeadSpec,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the detached or trainable loss tensor for one dense head."""
        logits, targets, valid = self._prepare_head_tensors(head, logits, targets)
        if callable(head.loss):
            return _call_loss(head.loss, logits, targets, valid)
        if head.loss == "cross_entropy":
            losses = functional.cross_entropy(
                logits, targets, ignore_index=self.ignore_index, reduction="none"
            )
        elif head.loss == "bce_with_logits":
            losses = functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
        else:
            losses = _REGRESSION_LOSSES[head.loss](logits, targets, reduction="none")
        return _masked_mean(losses, valid)

    def _head_metric_state(
        self,
        head: DenseHeadSpec,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> _MetricState:
        """Build additive statistics for one dense head."""
        mode = self._head_target_mode(head)
        names = self._resolve_head_metrics(head, mode)
        logits, targets, valid = self._prepare_head_tensors(head, logits, targets)
        classes = head.metric_classes
        if classes is not None and max(classes) >= logits.shape[1]:
            msg = (
                f"Dense head `{head.name}` lists `metric_classes` up to "
                f"{max(classes)}, but its output has {logits.shape[1]} channels."
            )
            raise ValueError(msg)
        if mode == "multiclass":
            return _MetricState.multiclass(
                logits.argmax(1), targets, valid, logits.shape[1], names, classes
            )
        if mode == "binary":
            return _MetricState.binary(
                logits > 0, targets > _BINARY_THRESHOLD, valid, names, classes
            )
        return _MetricState.regression(logits - targets, valid, names)

    def reset_epoch_state(self, *, training: bool) -> None:  # noqa: ARG002
        """Reset bounded-size per-head statistics."""
        self._epoch_states = {}

    def update_epoch_state(self, output: object, targets: object) -> None:
        """Accumulate counts and error sums, without storing predictions."""
        for head in self.heads:
            if head.metrics == ():
                continue
            logits = self._select_head_output(output, head)
            target = self._select_head_target(targets, head)
            state = self._head_metric_state(head, logits, target)
            if head.name in self._epoch_states:
                self._epoch_states[head.name].add(state)
            else:
                self._epoch_states[head.name] = state

    def compute_epoch_metrics(self) -> dict[str, float]:
        """Compute dataset-level metrics from the accumulated statistics."""
        return {
            self._metric_key(head, name): value
            for head in self.heads
            if head.name in self._epoch_states
            for name, value in self._epoch_states[head.name].compute().items()
        }

    def loss_normalizers(
        self,
        targets: object,
        batch_size: int,  # noqa: ARG002 - hook signature
    ) -> dict[str, int]:
        """Count supervised elements per head."""
        return {
            head.name: int(
                (self._select_head_target(targets, head) != self.ignore_index).sum()
            )
            for head in self.heads
        }

    def validate_targets(self, output: object, targets: object) -> None:
        """Raise when multiclass targets fall outside a head's output channels."""
        for head in self.heads:
            if self._head_target_mode(head) != "multiclass":
                continue
            logits = self._select_head_output(output, head)
            if logits.ndim == _DENSE_LOGIT_DIMS:
                _validate_class_indices(
                    self._select_head_target(targets, head),
                    self.ignore_index,
                    logits.shape[1],
                    kind="Dense multiclass",
                )

    def loss_components(
        self, output: object, targets: object
    ) -> dict[str, torch.Tensor]:
        """Compute each weighted head loss once."""
        return {
            head.name: self._compute_head_loss(
                head,
                self._select_head_output(output, head),
                self._select_head_target(targets, head),
            )
            * head.loss_weight
            for head in self.heads
        }

    def _metric_key(
        self: StructuredDenseTask,
        head: DenseHeadSpec,
        metric_name: str,
    ) -> str:
        """Return the public metric key for a head-specific metric."""
        if not self.prefix_head_metrics and len(self.heads) == 1:
            return metric_name
        return f"{head.name}_{metric_name}"


class ClassificationTask(TrainingTaskABC):
    """Train from raw classification logits and tensor labels.

    Named losses fix the target semantics: ``cross_entropy`` is single-label,
    and ``bce_with_logits`` is binary for one output and multi-label for several.
    Their metrics differ: binary F1 averages the two classes, and multi-label F1
    the outputs. A callable loss requires an explicit ``target_mode``, which is
    only for callables, and must return one unreduced value per target element
    (e.g. ``nn.CrossEntropyLoss(reduction='none')``). So a one-output head scored
    as multi-label needs a callable BCE loss. The task replaces ignored targets
    with zero before calling a callable and averages only valid elements.
    Criterion modules' buffers follow the logits' device; trainable criteria and
    batch-coupled objectives require a custom :class:`TrainingTaskABC`.

    Single-label logits/targets have shapes ``(N, C)``/``(N,)`` with ``C >= 2``;
    floating ``(N, C)`` targets are class probabilities (soft labels, as from
    mixup or distillation), always supervised and scored by their argmax.
    Binary BCE accepts ``(N,)`` or ``(N, 1)``; multi-label BCE accepts ``(N, C)``.
    Use ``single_label`` for a callable cross-entropy loss, including two classes.
    Metrics are epoch accuracy and macro-F1, excluding ``ignore_index``.
    """

    def __init__(
        self: ClassificationTask,
        *,
        loss: str | LossFunction = "cross_entropy",
        target_mode: ClassificationTargetMode | None = None,
        output_key: NestedKey | None = None,
        ignore_index: int = -100,
    ) -> None:
        """Initialize :class:`ClassificationTask`."""
        super().__init__(output_key=output_key, ignore_index=ignore_index)
        self.loss = loss
        self.target_mode = target_mode
        self._epoch_state: _MetricState | None = None

        if callable(loss):
            _validate_loss_callable(loss, target_mode)
            if target_mode not in {"single_label", "binary", "multi_label"}:
                msg = f"Unsupported classification target mode `{target_mode}`."
                raise ValueError(msg)
            self.loss_fn = loss
        elif loss not in {"cross_entropy", "bce_with_logits"}:
            msg = f"Unsupported classification loss `{loss}`."
            raise ValueError(msg)
        elif target_mode is not None:
            msg = (
                "`target_mode` is only for callable losses: `cross_entropy` is "
                "single-label, and `bce_with_logits` is binary for one output and "
                "multi-label for several."
            )
            raise ValueError(msg)
        elif loss == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss(
                ignore_index=self.ignore_index, reduction="none"
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def required_output_mode(
        self: ClassificationTask, num_classes: int
    ) -> ClassificationTargetMode:
        """Return the ``output_mode`` a classifier with ``num_classes`` outputs needs.

        TIAToolbox's ``CNNModel`` and ``TimmModel`` apply softmax or sigmoid
        in ``infer_batch`` according to their ``output_mode``, which must
        match the objective they were trained with.
        """
        if self.target_mode is not None:
            return self.target_mode
        if self.loss == "cross_entropy":
            return "single_label"
        return "binary" if num_classes == 1 else "multi_label"

    @property
    def _single_label(self: ClassificationTask) -> bool:
        """Return whether targets are class indices or probabilities over outputs."""
        return self.target_mode == "single_label" or self.loss == "cross_entropy"

    def _resolve_target_mode(
        self: ClassificationTask,
        logits: torch.Tensor,
    ) -> ClassificationTargetMode:
        """Return the target mode for logits; shape errors are reported later."""
        return self.required_output_mode(
            logits.shape[1] if logits.ndim >= _CLASSIFICATION_DIMS else 1
        )

    @staticmethod
    def _prepare_single_label_logits(logits: torch.Tensor) -> torch.Tensor:
        """Validate logits for single-label classification."""
        if (
            logits.ndim != _CLASSIFICATION_DIMS
            or logits.shape[1] < _MIN_MULTICLASS_CHANNELS
        ):
            msg = (
                "Single-label classification expects logits with shape `(N, C)` "
                "where `C >= 2`."
            )
            raise ValueError(msg)
        return logits

    @staticmethod
    def _prepare_single_label_targets(targets: torch.Tensor) -> torch.Tensor:
        """Normalize single-label class indices to shape `(N,)`."""
        if targets.ndim == _CLASSIFICATION_DIMS and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if targets.ndim != 1:
            msg = (
                "Single-label classification targets must have shape `(N,)`, or "
                "`(N, C)` floating class probabilities."
            )
            raise ValueError(msg)
        return targets.long()

    def _prepare_single_label_tensors(
        self: ClassificationTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate paired single-label logits and targets; return a valid mask."""
        logits = self._prepare_single_label_logits(logits)
        if _is_soft_label(targets):
            if targets.shape != logits.shape:
                msg = "Soft-label targets must match the logits shape `(N, C)`."
                raise ValueError(msg)
            valid = torch.ones(
                targets.shape[0], dtype=torch.bool, device=targets.device
            )
            return logits, targets.float(), valid
        targets = self._prepare_single_label_targets(targets)
        if targets.shape[0] != logits.shape[0]:
            msg = (
                f"Single-label classification targets batch size `{targets.shape[0]}` "
                f"must match logits batch size `{logits.shape[0]}`."
            )
            raise ValueError(msg)

        return logits, targets, targets != self.ignore_index

    def _prepare_binary_tensors(
        self: ClassificationTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize binary BCE inputs and derive a valid-element mask."""
        if logits.ndim == _CLASSIFICATION_DIMS and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if logits.ndim != 1:
            msg = (
                "Binary BCE classification expects logits with shape `(N,)` "
                "or `(N, 1)`."
            )
            raise ValueError(msg)

        if targets.ndim == _CLASSIFICATION_DIMS and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if targets.ndim != 1:
            msg = (
                "Binary BCE classification targets must have shape `(N,)` or `(N, 1)`."
            )
            raise ValueError(msg)
        if targets.shape != logits.shape:
            msg = "Binary BCE classification targets must match the logits shape."
            raise ValueError(msg)

        valid_mask = targets != self.ignore_index
        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        return logits, safe_targets.float(), valid_mask

    def _prepare_multilabel_tensors(
        self: ClassificationTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize multi-label BCE inputs and derive a valid-element mask."""
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)

        if logits.ndim != _CLASSIFICATION_DIMS:
            msg = "Multi-label classification expects logits with shape `(N, C)`."
            raise ValueError(msg)
        if targets.ndim != _CLASSIFICATION_DIMS:
            msg = "Multi-label classification targets must have shape `(N, C)`."
            raise ValueError(msg)
        if logits.shape != targets.shape:
            msg = "Multi-label classification targets must match the logits shape."
            raise ValueError(msg)

        valid_mask = targets != self.ignore_index
        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        return logits, safe_targets.float(), valid_mask

    def loss_components(
        self: ClassificationTask,
        output: object,
        targets: object,
    ) -> dict[str, torch.Tensor]:
        """Return the mean classification loss over supervised targets."""
        logits = self.select_output(output)
        if not isinstance(targets, torch.Tensor):
            msg = "Classification tasks require tensor targets."
            raise ValueError(msg)  # noqa: TRY004 - preserve validation API
        target_mode = self._resolve_target_mode(logits)

        if target_mode == "single_label":
            logits, targets, valid_mask = self._prepare_single_label_tensors(
                logits, targets
            )
        elif target_mode == "binary":
            logits, targets, valid_mask = self._prepare_binary_tensors(logits, targets)
        else:
            logits, targets, valid_mask = self._prepare_multilabel_tensors(
                logits, targets
            )
        return {"loss": _call_loss(self.loss_fn, logits, targets, valid_mask)}

    def loss_normalizers(
        self,
        targets: object,
        batch_size: int,  # noqa: ARG002 - hook signature
    ) -> dict[str, int]:
        """Count supervised samples or labels."""
        if not isinstance(targets, torch.Tensor):
            msg = "Classification tasks require tensor targets."
            raise ValueError(msg)  # noqa: TRY004 - preserve validation API
        if self._single_label and _is_soft_label(targets):
            return {"loss": int(targets.shape[0])}
        return {"loss": int((targets != self.ignore_index).sum())}

    def validate_targets(self, output: object, targets: object) -> None:
        """Raise when single-label targets fall outside the model's output classes."""
        logits = self.select_output(output)
        if (
            self._single_label
            and logits.ndim == _CLASSIFICATION_DIMS
            and isinstance(targets, torch.Tensor)
            and not _is_soft_label(targets)
        ):
            _validate_class_indices(
                targets,
                self.ignore_index,
                logits.shape[1],
                kind="Single-label classification",
            )

    def _metric_state(self, output: object, targets: object) -> _MetricState:
        """Build additive classification statistics."""
        logits = self.select_output(output)
        if not isinstance(targets, torch.Tensor):
            msg = "Classification tasks require tensor targets."
            raise ValueError(msg)  # noqa: TRY004 - preserve validation API
        mode = self._resolve_target_mode(logits)
        if mode == "single_label":
            logits, targets, valid = self._prepare_single_label_tensors(logits, targets)
            truth = targets.argmax(1) if _is_soft_label(targets) else targets
            return _MetricState.multiclass(
                logits.argmax(1), truth, valid, logits.shape[1], ("accuracy", "f1")
            )
        if mode == "binary":
            logits, targets, valid = self._prepare_binary_tensors(logits, targets)
            return _MetricState.multiclass(
                (logits > 0).long(), targets.long(), valid, 2, ("accuracy", "f1")
            )
        logits, targets, valid = self._prepare_multilabel_tensors(logits, targets)
        return _MetricState.binary(
            logits > 0, targets > _BINARY_THRESHOLD, valid, ("accuracy", "f1")
        )

    def reset_epoch_state(self, *, training: bool) -> None:  # noqa: ARG002
        """Reset classification counts."""
        self._epoch_state = None

    def update_epoch_state(self, output: object, targets: object) -> None:
        """Accumulate counts without retaining predictions."""
        state = self._metric_state(output, targets)
        if self._epoch_state is None:
            self._epoch_state = state
        else:
            self._epoch_state.add(state)

    def compute_epoch_metrics(self) -> dict[str, float]:
        """Compute accuracy and macro-F1 across the whole epoch."""
        return {} if self._epoch_state is None else self._epoch_state.compute()


class SegmentationTask(StructuredDenseTask):
    """One-head dense task with unprefixed Dice and IoU metrics.

    Accepts the same named or unreduced callable losses as :class:`DenseHeadSpec`.
    Set ``target_mode`` explicitly for a callable: ``multiclass`` uses integer
    labels, ``binary`` uses independent foreground channels. Outputs must be raw
    logits of shape ``(N, C, H, W)``; targets must have matching spatial dimensions.
    ``output_key`` selects the output of a model returning several.
    ``metric_classes`` selects the classes averaged into Dice/IoU, as for
    :class:`DenseHeadSpec`.
    Use :class:`StructuredDenseTask` for regression or multiple outputs.
    """

    def __init__(
        self,
        *,
        loss: DenseLossMode | LossFunction = "cross_entropy",
        target_mode: DenseTargetMode | None = None,
        output_key: NestedKey | None = None,
        ignore_index: int = -100,
        metric_classes: tuple[int, ...] | None = None,
    ) -> None:
        """Configure one segmentation head using the shared dense implementation."""
        super().__init__(
            heads=[
                DenseHeadSpec(
                    name="segmentation",
                    loss=loss,
                    target_mode=target_mode,
                    output_key=output_key,
                    metrics=("dice", "iou"),
                    metric_classes=metric_classes,
                )
            ],
            ignore_index=ignore_index,
            prefix_head_metrics=False,
        )
        self.loss = loss


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Average supervised values without synchronising with the device.

    The result is a differentiable zero when nothing is valid. ``torch.where``
    keeps non-finite values at ignored positions out of the sum, and reduced
    precision sums are widened so large masks do not overflow.
    """
    if values.dtype in {torch.float16, torch.bfloat16}:
        values = values.float()
    masked = torch.where(valid, values, torch.zeros_like(values))
    return masked.sum() / valid.sum().clamp_min(1)


class _MetricState:
    """Additive statistics with memory proportional to the number of classes."""

    def __init__(
        self,
        mode: str,
        names: tuple[str, ...],
        totals: torch.Tensor,
        subset: torch.Tensor | None = None,
        classes: tuple[int, ...] | None = None,
    ) -> None:
        """Store small detached count or error tensors."""
        self.mode = mode
        self.names = names
        self.totals = totals
        self.subset = subset
        self.classes = classes

    @classmethod
    def multiclass(
        cls,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        num_classes: int,
        names: tuple[str, ...],
        classes: tuple[int, ...] | None = None,
    ) -> _MetricState:
        """Count intersections, predictions, targets, and valid elements.

        Fixed-size ``index_add`` counts give ignored elements zero weight, so
        no output shape depends on device-side values.
        """
        predictions = predictions.reshape(-1)
        weight = valid.reshape(-1).to(torch.long)
        # Ignored targets may hold out-of-range sentinels such as -100.
        targets = torch.where(valid.reshape(-1), targets.reshape(-1), 0)
        zeros = torch.zeros(num_classes, dtype=torch.long, device=targets.device)
        tp = zeros.index_add(0, targets, weight * (predictions == targets))
        pred = zeros.index_add(0, predictions, weight)
        truth = zeros.index_add(0, targets, weight)
        count = weight.sum().expand(num_classes)
        totals = torch.stack((tp, pred, truth, count))
        return cls("multiclass", names, totals, classes=classes)

    @classmethod
    def regression(
        cls,
        errors: torch.Tensor,
        valid: torch.Tensor,
        names: tuple[str, ...],
    ) -> _MetricState:
        """Sum absolute and squared errors over valid elements."""
        errors = torch.where(valid, errors, torch.zeros_like(errors)).double()
        return cls(
            "regression",
            names,
            torch.stack(
                (errors.abs().sum(), errors.square().sum(), valid.sum().double())
            ),
        )

    @classmethod
    def binary(
        cls,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        names: tuple[str, ...],
        classes: tuple[int, ...] | None = None,
    ) -> _MetricState:
        """Count each binary channel and valid-row subset matches."""
        axes = (0, *range(2, predictions.ndim))
        totals = torch.stack(
            (
                (predictions & targets & valid).sum(axes),
                (predictions & valid).sum(axes),
                (targets & valid).sum(axes),
                valid.sum(axes),
            )
        )
        subset = None
        if "accuracy" in names:
            rows = valid.any(1)
            subset = torch.stack(
                ((((predictions == targets) | ~valid).all(1) & rows).sum(), rows.sum())
            )
        return cls("binary", names, totals, subset, classes)

    def add(self, other: _MetricState) -> None:
        """Merge another batch of matching statistics."""
        self.totals += other.totals
        if self.subset is not None:
            self.subset += other.subset

    def compute(self) -> dict[str, float]:
        """Reduce accumulated statistics, preserving the task's empty-class rules."""
        values = self.totals.double()
        if self.mode == "regression":
            return {
                name: float(values[index] / values[2].clamp_min(1))
                for name, index in (("mae", 0), ("mse", 1))
                if name in self.names
            }
        result = {}
        if "accuracy" in self.names:
            if self.subset is not None:
                correct, count = self.subset.double()
            else:
                correct, count = values[0].sum(), values[3, 0]
            result["accuracy"] = float(correct / count.clamp_min(1))
        result.update(self._compute_overlap_metrics(values))
        return result

    def _compute_overlap_metrics(self, values: torch.Tensor) -> dict[str, float]:
        """Reduce Dice, IoU and F1 with the configured empty-class rules."""
        tp, pred, truth, valid = values
        denominator = pred + truth
        result = {}
        for name in self.names:
            if name not in {"dice", "iou", "f1"}:
                continue
            denom = denominator - tp if name == "iou" else denominator
            numerator = tp if name == "iou" else 2 * tp
            scores = numerator / denom.clamp_min(1)
            if self.mode == "binary":
                included = valid > 0
                if name in {"dice", "iou"}:
                    scores = torch.where(denom == 0, 1.0, scores)
            elif name == "f1":
                included = torch.ones_like(denom, dtype=torch.bool)
            else:
                included = denom > 0
            if self.classes is not None:
                for index in self.classes:
                    score = float(scores[index]) if included[index] else float("nan")
                    result[f"{name}_{index}"] = score
                selected = torch.zeros_like(included)
                selected[list(self.classes)] = True
                included = included & selected
            result[name] = float(scores[included].sum() / included.sum().clamp_min(1))
        return result
