"""Task adapters for model training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence


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
    """Task adapter for classification models."""

    def __init__(
        self: ClassificationTask,
        *,
        loss: str = "cross_entropy",
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

        if self.loss == "cross_entropy":
            self.loss_fn: nn.Module = nn.CrossEntropyLoss(
                ignore_index=self.ignore_index
            )
        elif self.loss == "bce_with_logits":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            msg = f"Unsupported classification loss `{self.loss}`."
            raise ValueError(msg)

    def compute_loss(
        self: ClassificationTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification loss."""
        if self.loss == "cross_entropy":
            targets = targets.long().view(-1)
            return self.loss_fn(logits, targets)

        targets = targets.float()
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)
        return self.loss_fn(logits, targets)

    def compute_metrics(
        self: ClassificationTask,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """Compute classification accuracy and macro-F1."""
        targets = targets.long().view(-1)

        if logits.ndim == 1 or logits.shape[-1] == 1:
            probs = torch.sigmoid(logits.view(-1))
            predictions = (probs > 0.5).long()
            num_classes = 2
        else:
            predictions = torch.argmax(logits, dim=1)
            num_classes = logits.shape[-1]

        accuracy = (predictions == targets).float().mean().item()
        f1_score = _macro_f1_score(predictions, targets, num_classes)

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
            self.bce_loss = nn.BCEWithLogitsLoss()
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

            targets = targets.float()
            if targets.ndim == 3:
                targets = targets.unsqueeze(1)
            return self.bce_loss(logits, targets)

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
            if targets.ndim == 3:
                targets = targets.unsqueeze(1)

            valid_mask = targets != self.ignore_index

            target_mask = targets > 0
            pred_mask = torch.sigmoid(logits) > 0.5

            intersection = (pred_mask & target_mask & valid_mask).sum().float()
            pred_area = (pred_mask & valid_mask).sum().float()
            target_area = (target_mask & valid_mask).sum().float()

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
