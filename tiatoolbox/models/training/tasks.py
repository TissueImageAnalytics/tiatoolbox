"""Task adapters for model training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

ClassificationTargetMode = Literal[
    "auto",
    "single_label",
    "binary",
    "multi_label",
]
DenseLossMode = Literal[
    "auto",
    "cross_entropy",
    "bce_with_logits",
    "mse",
    "l1",
    "smooth_l1",
]
DenseTargetMode = Literal["auto", "multiclass", "binary", "regression"]
DenseMetricName = Literal["dice", "iou", "mae", "mse"]
NestedKey = str | tuple[str, ...]


def _first_tensor_from_mapping(output: Mapping) -> torch.Tensor:
    """Return the first tensor found in a mapping output."""
    for value in output.values():
        if isinstance(value, torch.Tensor):
            return value
    msg = "No tensor value found in model output mapping."
    raise ValueError(msg)


def _first_tensor_from_sequence(output: Sequence) -> torch.Tensor:
    """Return the first tensor found in a sequence output."""
    for value in output:
        if isinstance(value, torch.Tensor):
            return value
    msg = "No tensor value found in model output sequence."
    raise ValueError(msg)


class TrainingTaskABC(ABC):
    """Abstract task adapter used by :class:`Trainer`."""

    def __init__(
        self: TrainingTaskABC,
        *,
        loss: str = "cross_entropy",
        output_key: str | None = None,
        output_index: int | None = None,
        ignore_index: int = -100,
    ) -> None:
        """Initialize :class:`TrainingTaskABC`."""
        self.loss = loss
        self.output_key = output_key
        self.output_index = output_index
        self.ignore_index = ignore_index

    def select_output(self: TrainingTaskABC, output: object) -> torch.Tensor:
        """Select a tensor output for loss and metric computation."""
        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, dict):
            if self.output_key is not None:
                selected = output.get(self.output_key)
                if not isinstance(selected, torch.Tensor):
                    msg = f"Output key `{self.output_key}` does not contain a tensor."
                    raise ValueError(msg)
                return selected
            return _first_tensor_from_mapping(output)

        if isinstance(output, (list, tuple)):
            if self.output_index is not None:
                selected = output[self.output_index]
                if not isinstance(selected, torch.Tensor):
                    msg = (
                        f"Output index `{self.output_index}` does not contain a tensor."
                    )
                    raise ValueError(msg)
                return selected
            return _first_tensor_from_sequence(output)

        msg = f"Unsupported model output type: `{type(output).__name__}`."
        raise TypeError(msg)

    def reset_epoch_state(
        self: TrainingTaskABC,
        *,
        training: bool,
    ) -> None:
        """Reset any task-specific state tracked across an epoch."""
        del training

    def update_epoch_state(
        self: TrainingTaskABC,
        output: object,
        targets: object,
    ) -> None:
        """Update task-specific epoch state from one detached batch."""
        del output, targets

    def compute_epoch_metrics(self: TrainingTaskABC) -> dict[str, float]:
        """Return any task-specific metrics accumulated across the epoch."""
        return {}

    @abstractmethod
    def compute_loss(
        self: TrainingTaskABC,
        output: object,
        targets: object,
    ) -> torch.Tensor:
        """Compute training loss."""

    @abstractmethod
    def compute_metrics(
        self: TrainingTaskABC,
        output: object,
        targets: object,
    ) -> dict[str, float]:
        """Compute batch-level metrics."""


@dataclass(frozen=True)
class DenseHeadSpec:
    """Configuration for one dense prediction head inside a structured task."""

    name: str
    loss: DenseLossMode = "auto"
    target_mode: DenseTargetMode = "auto"
    target_key: NestedKey | None = None
    output_key: NestedKey | None = None
    output_index: int | None = None
    channel_slice: int | slice | tuple[int, ...] | list[int] | None = None
    loss_weight: float = 1.0
    metrics: tuple[DenseMetricName, ...] | None = None


class StructuredDenseTask(TrainingTaskABC):
    """Unified task adapter for one-head and multi-head dense prediction models."""

    def __init__(
        self: StructuredDenseTask,
        heads: list[DenseHeadSpec],
        *,
        output_key: str | None = None,
        output_index: int | None = None,
        ignore_index: int = -100,
        prefix_head_metrics: bool = True,
        include_head_loss_metrics: bool = True,
    ) -> None:
        """Initialize :class:`StructuredDenseTask`."""
        super().__init__(
            loss="structured_dense",
            output_key=output_key,
            output_index=output_index,
            ignore_index=ignore_index,
        )
        if not heads:
            msg = "`heads` must contain at least one dense head specification."
            raise ValueError(msg)
        for head in heads:
            if head.loss_weight <= 0:
                msg = (
                    f"Dense head `{head.name}` must have a positive `loss_weight`, "
                    f"got {head.loss_weight}."
                )
                raise ValueError(msg)
        self.heads = heads
        self.prefix_head_metrics = prefix_head_metrics
        self.include_head_loss_metrics = include_head_loss_metrics

    def _select_nested_value(
        self: StructuredDenseTask,
        value: object,
        key: NestedKey,
        *,
        kind: str,
    ) -> object:
        """Resolve a nested mapping value from a string or tuple key path."""
        key_path = (key,) if isinstance(key, str) else key
        current = value
        for part in key_path:
            if not isinstance(current, dict):
                msg = (
                    f"{kind} path `{key_path}` expects nested mappings, "
                    f"but encountered `{type(current).__name__}`."
                )
                raise ValueError(msg)
            if part not in current:
                msg = f"{kind} path `{key_path}` could not be resolved."
                raise ValueError(msg)
            current = current[part]
        return current

    def _select_head_output(
        self: StructuredDenseTask,
        output: object,
        head: DenseHeadSpec,
    ) -> torch.Tensor:
        """Resolve one head tensor from raw model output."""
        head_output: object
        if head.output_key is not None:
            head_output = self._select_nested_value(output, head.output_key, kind="Output")
        elif head.output_index is not None:
            if not isinstance(output, (list, tuple)):
                msg = (
                    f"Dense head `{head.name}` requested `output_index`, "
                    f"but model output is `{type(output).__name__}`."
                )
                raise ValueError(msg)
            head_output = output[head.output_index]
        elif isinstance(output, dict) and head.name in output:
            head_output = output[head.name]
        else:
            head_output = self.select_output(output)

        if not isinstance(head_output, torch.Tensor):
            msg = f"Dense head `{head.name}` did not resolve to a tensor output."
            raise ValueError(msg)

        if head.channel_slice is None:
            return head_output

        channel_slice = head.channel_slice
        if isinstance(channel_slice, int):
            return head_output[:, channel_slice : channel_slice + 1, ...]
        return head_output[:, channel_slice, ...]

    def _select_head_target(
        self: StructuredDenseTask,
        targets: object,
        head: DenseHeadSpec,
    ) -> torch.Tensor:
        """Resolve one head target tensor from raw task targets."""
        head_target: object
        if head.target_key is not None:
            head_target = self._select_nested_value(targets, head.target_key, kind="Target")
        elif isinstance(targets, dict) and head.name in targets:
            head_target = targets[head.name]
        else:
            head_target = targets

        if not isinstance(head_target, torch.Tensor):
            msg = f"Dense head `{head.name}` requires tensor targets."
            raise ValueError(msg)
        return head_target

    def _resolve_head_loss(
        self: StructuredDenseTask,
        head: DenseHeadSpec,
        logits: torch.Tensor,
    ) -> DenseLossMode:
        """Resolve the effective loss mode for one head."""
        if head.loss != "auto":
            return head.loss
        if head.target_mode == "regression":
            return "mse"
        if head.target_mode == "binary":
            return "bce_with_logits"
        if head.target_mode == "multiclass":
            return "cross_entropy"
        if logits.ndim == 4 and logits.shape[1] == 1:
            return "bce_with_logits"
        return "cross_entropy"

    def _resolve_head_metrics(
        self: StructuredDenseTask,
        head: DenseHeadSpec,
        resolved_loss: DenseLossMode,
    ) -> tuple[DenseMetricName, ...]:
        """Resolve the effective metric set for one head."""
        if head.metrics is not None:
            return head.metrics
        if resolved_loss in {"mse", "l1", "smooth_l1"}:
            return ("mae", "mse")
        return ("dice", "iou")

    def _validate_dense_batch_and_spatial_shapes(
        self: StructuredDenseTask,
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
        if logits.ndim != 4 or logits.shape[1] < 2:
            msg = (
                "Dense multiclass heads expect logits with shape `(N, C, H, W)` "
                "where `C >= 2`."
            )
            raise ValueError(msg)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if targets.ndim != 3:
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
        valid_mask = targets != self.ignore_index
        valid_targets = targets[valid_mask]
        if valid_targets.numel() > 0 and (
            torch.any(valid_targets < 0) or torch.any(valid_targets >= logits.shape[1])
        ):
            msg = (
                "Dense multiclass targets contain class indices outside the "
                f"valid range [0, {logits.shape[1] - 1}] and not equal to "
                f"ignore_index `{self.ignore_index}`."
            )
            raise ValueError(msg)
        return logits, targets, valid_mask

    def _prepare_binary_dense_tensors(
        self: StructuredDenseTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize dense binary BCE tensors and derive a valid-element mask."""
        if logits.ndim != 4:
            msg = "Dense binary heads expect logits with shape `(N, C, H, W)`."
            raise ValueError(msg)
        if targets.ndim == 3 and logits.shape[1] == 1:
            targets = targets.unsqueeze(1)
        if targets.ndim != 4:
            msg = (
                "Dense binary targets must have shape `(N, H, W)` "
                "or `(N, C, H, W)`."
            )
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
        if logits.ndim != 4:
            msg = "Dense regression heads expect logits with shape `(N, C, H, W)`."
            raise ValueError(msg)
        if targets.ndim == 3 and logits.shape[1] == 1:
            targets = targets.unsqueeze(1)
        if targets.ndim != 4:
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

    def _compute_binary_metrics(
        self: StructuredDenseTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
        metrics: tuple[DenseMetricName, ...],
    ) -> dict[str, float]:
        """Compute channel-averaged dense binary segmentation metrics."""
        if not torch.any(valid_mask):
            return {
                metric_name: 0.0
                for metric_name in metrics
                if metric_name in {"dice", "iou"}
            }

        pred_mask = torch.sigmoid(logits) > 0.5
        target_mask = targets > 0.5

        dice_values = []
        iou_values = []
        for channel_index in range(logits.shape[1]):
            channel_valid = valid_mask[:, channel_index, :, :]
            if not torch.any(channel_valid):
                continue

            pred_channel = pred_mask[:, channel_index, :, :]
            target_channel = target_mask[:, channel_index, :, :]

            intersection = (pred_channel & target_channel & channel_valid).sum().float()
            pred_area = (pred_channel & channel_valid).sum().float()
            target_area = (target_channel & channel_valid).sum().float()

            if float(pred_area.item()) == 0 and float(target_area.item()) == 0:
                dice_values.append(1.0)
                iou_values.append(1.0)
                continue

            dice_values.append(_safe_ratio(2.0 * intersection, pred_area + target_area))
            iou_values.append(
                _safe_ratio(intersection, pred_area + target_area - intersection)
            )

        computed_metrics: dict[str, float] = {}
        if "dice" in metrics:
            computed_metrics["dice"] = (
                float(sum(dice_values) / len(dice_values)) if dice_values else 0.0
            )
        if "iou" in metrics:
            computed_metrics["iou"] = (
                float(sum(iou_values) / len(iou_values)) if iou_values else 0.0
            )
        return computed_metrics

    def _compute_multiclass_metrics(
        self: StructuredDenseTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
        metrics: tuple[DenseMetricName, ...],
    ) -> dict[str, float]:
        """Compute dense multiclass segmentation metrics."""
        predictions = torch.argmax(logits, dim=1)
        num_classes = logits.shape[1]

        dice_values = []
        iou_values = []
        for class_index in range(num_classes):
            pred_class = (predictions == class_index) & valid_mask
            target_class = (targets == class_index) & valid_mask

            intersection = (pred_class & target_class).sum().float()
            pred_area = pred_class.sum().float()
            target_area = target_class.sum().float()

            dice_denom = pred_area + target_area
            iou_denom = pred_area + target_area - intersection

            if float(dice_denom.item()) > 0:
                dice_values.append(_safe_ratio(2.0 * intersection, dice_denom))
            if float(iou_denom.item()) > 0:
                iou_values.append(_safe_ratio(intersection, iou_denom))

        computed_metrics: dict[str, float] = {}
        if "dice" in metrics:
            computed_metrics["dice"] = (
                float(sum(dice_values) / len(dice_values)) if dice_values else 0.0
            )
        if "iou" in metrics:
            computed_metrics["iou"] = (
                float(sum(iou_values) / len(iou_values)) if iou_values else 0.0
            )
        return computed_metrics

    def _compute_regression_metrics(
        self: StructuredDenseTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
        metrics: tuple[DenseMetricName, ...],
    ) -> dict[str, float]:
        """Compute masked dense regression metrics."""
        if not torch.any(valid_mask):
            return {
                metric_name: 0.0 for metric_name in metrics if metric_name in {"mae", "mse"}
            }

        errors = logits - targets
        computed_metrics: dict[str, float] = {}
        if "mae" in metrics:
            computed_metrics["mae"] = float(errors.abs()[valid_mask].mean().item())
        if "mse" in metrics:
            computed_metrics["mse"] = float((errors.square())[valid_mask].mean().item())
        return computed_metrics

    def _compute_head_loss(
        self: StructuredDenseTask,
        head: DenseHeadSpec,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, DenseLossMode]:
        """Compute the detached or trainable loss tensor for one dense head."""
        resolved_loss = self._resolve_head_loss(head, logits)

        if resolved_loss == "cross_entropy":
            logits, targets, _ = self._prepare_multiclass_dense_tensors(logits, targets)
            loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
            return loss, resolved_loss

        if resolved_loss == "bce_with_logits":
            logits, targets, valid_mask = self._prepare_binary_dense_tensors(
                logits,
                targets,
            )
            losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            return _masked_mean(losses, valid_mask, logits), resolved_loss

        logits, targets, valid_mask = self._prepare_regression_dense_tensors(
            logits,
            targets,
        )
        if resolved_loss == "mse":
            losses = F.mse_loss(logits, targets, reduction="none")
        elif resolved_loss == "l1":
            losses = F.l1_loss(logits, targets, reduction="none")
        elif resolved_loss == "smooth_l1":
            losses = F.smooth_l1_loss(logits, targets, reduction="none")
        else:
            msg = f"Unsupported dense loss `{resolved_loss}`."
            raise ValueError(msg)
        return _masked_mean(losses, valid_mask, logits), resolved_loss

    def _compute_head_metrics(
        self: StructuredDenseTask,
        head: DenseHeadSpec,
        logits: torch.Tensor,
        targets: torch.Tensor,
        resolved_loss: DenseLossMode,
    ) -> dict[str, float]:
        """Compute metric values for one dense head."""
        metrics = self._resolve_head_metrics(head, resolved_loss)
        if not metrics:
            return {}

        if resolved_loss == "cross_entropy":
            logits, targets, valid_mask = self._prepare_multiclass_dense_tensors(
                logits,
                targets,
            )
            return self._compute_multiclass_metrics(logits, targets, valid_mask, metrics)

        if resolved_loss == "bce_with_logits":
            logits, targets, valid_mask = self._prepare_binary_dense_tensors(
                logits,
                targets,
            )
            return self._compute_binary_metrics(logits, targets, valid_mask, metrics)

        logits, targets, valid_mask = self._prepare_regression_dense_tensors(
            logits,
            targets,
        )
        return self._compute_regression_metrics(logits, targets, valid_mask, metrics)

    def _metric_key(
        self: StructuredDenseTask,
        head: DenseHeadSpec,
        metric_name: str,
    ) -> str:
        """Return the public metric key for a head-specific metric."""
        if (
            metric_name not in {"loss", "weighted_loss"}
            and not self.prefix_head_metrics
            and len(self.heads) == 1
        ):
            return metric_name
        return f"{head.name}_{metric_name}"

    def compute_loss(
        self: StructuredDenseTask,
        output: object,
        targets: object,
    ) -> torch.Tensor:
        """Compute the weighted aggregate loss across all dense heads."""
        total_loss: torch.Tensor | None = None

        for head in self.heads:
            head_output = self._select_head_output(output, head)
            head_target = self._select_head_target(targets, head)
            head_loss, _ = self._compute_head_loss(head, head_output, head_target)
            weighted_loss = head_loss * head.loss_weight
            total_loss = (
                weighted_loss if total_loss is None else total_loss + weighted_loss
            )

        if total_loss is None:
            msg = "StructuredDenseTask could not compute a loss for any configured head."
            raise RuntimeError(msg)
        return total_loss

    def compute_metrics(
        self: StructuredDenseTask,
        output: object,
        targets: object,
    ) -> dict[str, float]:
        """Compute detached per-head metrics for all configured dense heads."""
        metrics: dict[str, float] = {}
        for head in self.heads:
            head_output = self._select_head_output(output, head)
            head_target = self._select_head_target(targets, head)
            head_loss, resolved_loss = self._compute_head_loss(head, head_output, head_target)

            if self.include_head_loss_metrics:
                metrics[self._metric_key(head, "loss")] = float(head_loss.item())
                if head.loss_weight != 1.0:
                    metrics[self._metric_key(head, "weighted_loss")] = float(
                        (head_loss * head.loss_weight).item()
                    )

            for metric_name, metric_value in self._compute_head_metrics(
                head,
                head_output,
                head_target,
                resolved_loss,
            ).items():
                metrics[self._metric_key(head, metric_name)] = metric_value

        return metrics


class ClassificationTask(TrainingTaskABC):
    """Task adapter for single-label, binary, and multi-label classification."""

    def __init__(
        self: ClassificationTask,
        *,
        loss: str = "cross_entropy",
        target_mode: ClassificationTargetMode = "auto",
        output_key: str | None = None,
        output_index: int | None = None,
        ignore_index: int = -100,
    ) -> None:
        """Initialize :class:`ClassificationTask`."""
        super().__init__(
            loss=loss,
            output_key=output_key,
            output_index=output_index,
            ignore_index=ignore_index,
        )
        self.target_mode = target_mode

        if self.loss == "cross_entropy":
            if self.target_mode == "multi_label":
                msg = (
                    "`target_mode='multi_label'` requires "
                    "`loss='bce_with_logits'`."
                )
                raise ValueError(msg)
            self.loss_fn: nn.Module = nn.CrossEntropyLoss(
                ignore_index=self.ignore_index
            )
        elif self.loss == "bce_with_logits":
            if self.target_mode == "single_label":
                msg = (
                    "`loss='bce_with_logits'` does not support "
                    "single-label targets."
                )
                raise ValueError(msg)
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        else:
            msg = f"Unsupported classification loss `{self.loss}`."
            raise ValueError(msg)

    def _resolve_target_mode(
        self: ClassificationTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Literal["single_label", "binary", "multi_label"]:
        """Resolve the effective target mode for the current batch."""
        if self.target_mode == "single_label":
            return "single_label"
        if self.target_mode == "binary":
            if self.loss == "cross_entropy":
                return "single_label"
            return "binary"
        if self.target_mode == "multi_label":
            return "multi_label"

        if self.loss == "cross_entropy":
            return "single_label"

        if logits.ndim == 1:
            return "binary"
        if logits.ndim == 2 and logits.shape[1] == 1:
            return "binary"
        if logits.ndim == 2 and logits.shape[1] > 1:
            if targets.ndim == 2 and targets.shape == logits.shape:
                return "multi_label"
            if (
                logits.shape[0] == 1
                and targets.ndim == 1
                and targets.shape[0] == logits.shape[1]
            ):
                return "multi_label"

        msg = (
            "Unable to infer classification target mode from `logits` and "
            "`targets`. Provide `target_mode` explicitly."
        )
        raise ValueError(msg)

    @staticmethod
    def _prepare_single_label_logits(logits: torch.Tensor) -> torch.Tensor:
        """Validate logits for single-label classification."""
        if logits.ndim != 2 or logits.shape[1] < 2:
            msg = (
                "Single-label classification expects logits with shape `(N, C)` "
                "where `C >= 2`."
            )
            raise ValueError(msg)
        return logits

    @staticmethod
    def _prepare_single_label_targets(targets: torch.Tensor) -> torch.Tensor:
        """Normalize single-label targets to shape `(N,)`."""
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if targets.ndim != 1:
            msg = "Single-label classification targets must have shape `(N,)`."
            raise ValueError(msg)
        return targets.long()

    def _prepare_single_label_tensors(
        self: ClassificationTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate paired single-label logits and targets."""
        logits = self._prepare_single_label_logits(logits)
        targets = self._prepare_single_label_targets(targets)
        if targets.shape[0] != logits.shape[0]:
            msg = (
                f"Single-label classification targets batch size `{targets.shape[0]}` "
                f"must match logits batch size `{logits.shape[0]}`."
            )
            raise ValueError(msg)

        valid_targets = targets[targets != self.ignore_index]
        if valid_targets.numel() > 0 and (
            torch.any(valid_targets < 0) or torch.any(valid_targets >= logits.shape[1])
        ):
            msg = (
                "Single-label classification targets contain class indices outside "
                f"the valid range [0, {logits.shape[1] - 1}] and not equal to "
                f"ignore_index `{self.ignore_index}`."
            )
            raise ValueError(msg)
        return logits, targets

    def _prepare_binary_tensors(
        self: ClassificationTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize binary BCE inputs and derive a valid-element mask."""
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if logits.ndim != 1:
            msg = (
                "Binary BCE classification expects logits with shape `(N,)` "
                "or `(N, 1)`."
            )
            raise ValueError(msg)

        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if targets.ndim != 1:
            msg = (
                "Binary BCE classification targets must have shape `(N,)` "
                "or `(N, 1)`."
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

        if logits.ndim != 2:
            msg = "Multi-label classification expects logits with shape `(N, C)`."
            raise ValueError(msg)
        if targets.ndim != 2:
            msg = "Multi-label classification targets must have shape `(N, C)`."
            raise ValueError(msg)
        if logits.shape != targets.shape:
            msg = "Multi-label classification targets must match the logits shape."
            raise ValueError(msg)

        valid_mask = targets != self.ignore_index
        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        return logits, safe_targets.float(), valid_mask

    def compute_loss(
        self: ClassificationTask,
        output: object,
        targets: object,
    ) -> torch.Tensor:
        """Compute classification loss."""
        logits = self.select_output(output)
        if not isinstance(targets, torch.Tensor):
            msg = "Classification tasks require tensor targets."
            raise ValueError(msg)
        target_mode = self._resolve_target_mode(logits, targets)

        if target_mode == "single_label":
            logits, targets = self._prepare_single_label_tensors(logits, targets)
            return self.loss_fn(logits, targets)

        if target_mode == "binary":
            logits, targets, valid_mask = self._prepare_binary_tensors(logits, targets)
            losses = self.loss_fn(logits, targets)
            return _masked_mean(losses, valid_mask, logits)

        logits, targets, valid_mask = self._prepare_multilabel_tensors(logits, targets)
        losses = self.loss_fn(logits, targets)
        return _masked_mean(losses, valid_mask, logits)

    def compute_metrics(
        self: ClassificationTask,
        output: object,
        targets: object,
    ) -> dict[str, float]:
        """Compute classification accuracy and macro-F1."""
        logits = self.select_output(output)
        if not isinstance(targets, torch.Tensor):
            msg = "Classification tasks require tensor targets."
            raise ValueError(msg)
        target_mode = self._resolve_target_mode(logits, targets)

        if target_mode == "single_label":
            logits, targets = self._prepare_single_label_tensors(logits, targets)
            valid_mask = targets != self.ignore_index
            if not torch.any(valid_mask):
                return {"accuracy": 0.0, "f1": 0.0}

            logits = logits[valid_mask]
            targets = targets[valid_mask]
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == targets).float().mean().item()
            f1_score = _macro_f1_score(predictions, targets, logits.shape[-1])
            return {
                "accuracy": float(accuracy),
                "f1": f1_score,
            }

        if target_mode == "binary":
            logits, targets, valid_mask = self._prepare_binary_tensors(logits, targets)
            if not torch.any(valid_mask):
                return {"accuracy": 0.0, "f1": 0.0}

            logits = logits[valid_mask]
            targets = targets[valid_mask].long()
            predictions = (torch.sigmoid(logits) > 0.5).long()
            accuracy = (predictions == targets).float().mean().item()
            f1_score = _macro_f1_score(predictions, targets, 2)
            return {
                "accuracy": float(accuracy),
                "f1": f1_score,
            }

        logits, targets, valid_mask = self._prepare_multilabel_tensors(
            logits,
            targets,
        )
        if not torch.any(valid_mask):
            return {"accuracy": 0.0, "f1": 0.0}

        predictions = torch.sigmoid(logits) > 0.5
        targets_bool = targets > 0.5
        accuracy = _multilabel_subset_accuracy(predictions, targets_bool, valid_mask)
        f1_score = _multilabel_macro_f1_score(
            predictions,
            targets_bool,
            valid_mask,
        )

        return {
            "accuracy": float(accuracy),
            "f1": f1_score,
        }


class SegmentationTask(TrainingTaskABC):
    """Task adapter for semantic segmentation models."""

    def __init__(
        self: SegmentationTask,
        *,
        loss: str = "cross_entropy",
        output_key: str | None = None,
        output_index: int | None = None,
        ignore_index: int = -100,
    ) -> None:
        """Initialize :class:`SegmentationTask`."""
        head = DenseHeadSpec(
            name="segmentation",
            loss=loss,
            target_mode="auto",
            metrics=("dice", "iou"),
        )
        self._dense_task = StructuredDenseTask(
            heads=[head],
            output_key=output_key,
            output_index=output_index,
            ignore_index=ignore_index,
            prefix_head_metrics=False,
            include_head_loss_metrics=False,
        )
        super().__init__(
            loss=loss,
            output_key=output_key,
            output_index=output_index,
            ignore_index=ignore_index,
        )

    def compute_loss(
        self: SegmentationTask,
        output: object,
        targets: object,
    ) -> torch.Tensor:
        """Compute segmentation loss using the unified dense-task implementation."""
        return self._dense_task.compute_loss(output, targets)

    def compute_metrics(
        self: SegmentationTask,
        output: object,
        targets: object,
    ) -> dict[str, float]:
        """Compute segmentation metrics using the unified dense-task implementation."""
        return self._dense_task.compute_metrics(output, targets)

    def reset_epoch_state(
        self: SegmentationTask,
        *,
        training: bool,
    ) -> None:
        """Reset epoch state on the delegated dense-task implementation."""
        self._dense_task.reset_epoch_state(training=training)

    def update_epoch_state(
        self: SegmentationTask,
        output: object,
        targets: object,
    ) -> None:
        """Update epoch state on the delegated dense-task implementation."""
        self._dense_task.update_epoch_state(output, targets)

    def compute_epoch_metrics(self: SegmentationTask) -> dict[str, float]:
        """Return epoch metrics from the delegated dense-task implementation."""
        return self._dense_task.compute_epoch_metrics()


def _safe_ratio(
    numerator: torch.Tensor | float,
    denominator: torch.Tensor | float,
) -> float:
    """Compute a numerically safe scalar ratio."""
    denominator_value = (
        float(denominator.item())
        if isinstance(denominator, torch.Tensor)
        else float(denominator)
    )
    if denominator_value <= 0:
        return 0.0
    numerator_value = (
        float(numerator.item())
        if isinstance(numerator, torch.Tensor)
        else float(numerator)
    )
    return numerator_value / denominator_value


def _masked_mean(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return the mean over valid values, or zero if none are valid."""
    valid_values = values[valid_mask]
    if valid_values.numel() == 0:
        return reference.sum() * 0.0
    return valid_values.mean()


def _macro_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    """Compute a macro-averaged F1 score."""
    f1_values = []
    for class_index in range(num_classes):
        pred_class = predictions == class_index
        target_class = targets == class_index

        true_positive = (pred_class & target_class).sum().float()
        false_positive = (pred_class & ~target_class).sum().float()
        false_negative = (~pred_class & target_class).sum().float()

        denominator = (2 * true_positive) + false_positive + false_negative
        if float(denominator.item()) <= 0:
            f1_values.append(0.0)
        else:
            f1_values.append(_safe_ratio(2 * true_positive, denominator))

    return float(sum(f1_values) / len(f1_values)) if f1_values else 0.0


def _multilabel_subset_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """Compute subset accuracy over valid labels for each sample."""
    sample_accuracies = []
    for sample_index in range(predictions.shape[0]):
        sample_valid = valid_mask[sample_index]
        if not torch.any(sample_valid):
            continue
        matches = predictions[sample_index][sample_valid] == targets[sample_index][sample_valid]
        sample_accuracies.append(float(torch.all(matches).item()))

    return (
        float(sum(sample_accuracies) / len(sample_accuracies))
        if sample_accuracies
        else 0.0
    )


def _multilabel_macro_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """Compute a macro F1 score for multi-label predictions."""
    f1_values = []
    for class_index in range(predictions.shape[1]):
        class_valid = valid_mask[:, class_index]
        if not torch.any(class_valid):
            continue

        pred_class = predictions[class_valid, class_index]
        target_class = targets[class_valid, class_index]

        true_positive = (pred_class & target_class).sum().float()
        false_positive = (pred_class & ~target_class).sum().float()
        false_negative = (~pred_class & target_class).sum().float()

        denominator = (2 * true_positive) + false_positive + false_negative
        if float(denominator.item()) <= 0:
            f1_values.append(0.0)
        else:
            f1_values.append(_safe_ratio(2 * true_positive, denominator))

    return float(sum(f1_values) / len(f1_values)) if f1_values else 0.0
