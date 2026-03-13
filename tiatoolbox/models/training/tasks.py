"""Task adapters for model training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

ClassificationTargetMode = Literal[
    "auto",
    "single_label",
    "binary",
    "multi_label",
]


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

    @abstractmethod
    def compute_loss(
        self: TrainingTaskABC,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training loss."""

    @abstractmethod
    def compute_metrics(
        self: TrainingTaskABC,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """Compute batch-level metrics."""


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
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification loss."""
        target_mode = self._resolve_target_mode(logits, targets)

        if target_mode == "single_label":
            logits = self._prepare_single_label_logits(logits)
            targets = self._prepare_single_label_targets(targets)
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
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """Compute classification accuracy and macro-F1."""
        target_mode = self._resolve_target_mode(logits, targets)

        if target_mode == "single_label":
            logits = self._prepare_single_label_logits(logits)
            targets = self._prepare_single_label_targets(targets)
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
        super().__init__(
            loss=loss,
            output_key=output_key,
            output_index=output_index,
            ignore_index=ignore_index,
        )

        if self.loss in {"auto", "cross_entropy"}:
            self.cross_entropy_loss = nn.CrossEntropyLoss(
                ignore_index=self.ignore_index
            )
        else:
            self.cross_entropy_loss = None

        if self.loss in {"auto", "bce_with_logits"}:
            self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.bce_loss = None

        if self.loss not in {"auto", "cross_entropy", "bce_with_logits"}:
            msg = f"Unsupported segmentation loss `{self.loss}`."
            raise ValueError(msg)

    def _resolve_binary_mode(self: SegmentationTask, logits: torch.Tensor) -> bool:
        """Return whether the current output should use binary segmentation loss."""
        if self.loss == "bce_with_logits":
            return True
        if self.loss == "cross_entropy":
            return False
        return logits.shape[1] == 1

    def _prepare_binary_tensors(
        self: SegmentationTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize binary segmentation tensors and derive a valid-element mask."""
        if logits.ndim != 4 or logits.shape[1] != 1:
            msg = (
                "Binary segmentation expects logits with shape `(N, 1, H, W)`."
            )
            raise ValueError(msg)

        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if targets.ndim != 4 or targets.shape[1] != 1:
            msg = (
                "Binary segmentation targets must have shape `(N, H, W)` "
                "or `(N, 1, H, W)`."
            )
            raise ValueError(msg)
        if targets.shape != logits.shape:
            msg = "Binary segmentation targets must match the logits shape."
            raise ValueError(msg)

        valid_mask = targets != self.ignore_index
        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        return logits, safe_targets.float(), valid_mask

    def compute_loss(
        self: SegmentationTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute segmentation loss."""
        binary_mode = self._resolve_binary_mode(logits)

        if binary_mode:
            if self.bce_loss is None:
                msg = "Binary segmentation requested but BCE loss is not initialized."
                raise RuntimeError(msg)

            logits, targets, valid_mask = self._prepare_binary_tensors(logits, targets)
            losses = self.bce_loss(logits, targets)
            return _masked_mean(losses, valid_mask, logits)

        if self.cross_entropy_loss is None:
            msg = "Cross-entropy segmentation requested but CE loss is not initialized."
            raise RuntimeError(msg)

        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        targets = targets.long()
        return self.cross_entropy_loss(logits, targets)

    def compute_metrics(
        self: SegmentationTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """Compute segmentation Dice and IoU metrics."""
        binary_mode = self._resolve_binary_mode(logits)

        if binary_mode:
            logits, targets, valid_mask = self._prepare_binary_tensors(logits, targets)
            if not torch.any(valid_mask):
                return {"dice": 0.0, "iou": 0.0}

            target_mask = targets > 0.5
            pred_mask = torch.sigmoid(logits) > 0.5

            intersection = (pred_mask & target_mask & valid_mask).sum().float()
            pred_area = (pred_mask & valid_mask).sum().float()
            target_area = (target_mask & valid_mask).sum().float()

            if float(pred_area.item()) == 0 and float(target_area.item()) == 0:
                return {"dice": 1.0, "iou": 1.0}

            dice_value = _safe_ratio(2.0 * intersection, pred_area + target_area)
            iou_value = _safe_ratio(
                intersection,
                pred_area + target_area - intersection,
            )
            return {"dice": dice_value, "iou": iou_value}

        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        targets = targets.long()

        valid_mask = targets != self.ignore_index

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

        dice_score = float(sum(dice_values) / len(dice_values)) if dice_values else 0.0
        iou_score = float(sum(iou_values) / len(iou_values)) if iou_values else 0.0
        return {"dice": dice_score, "iou": iou_score}


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
