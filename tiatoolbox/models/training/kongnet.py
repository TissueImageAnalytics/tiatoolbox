"""KongNet-specific helpers for TIAToolbox training tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

from tiatoolbox.models.architecture.kongnet import KongNet, KongNetOutputHeadSpec
from tiatoolbox.models.training.tasks import DenseHeadSpec, StructuredDenseTask

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    from tiatoolbox.models.training.tasks import (
        DenseLossMode,
        DenseMetricName,
        DenseTargetMode,
        NestedKey,
    )

KongNetTargetSelection = Literal["full_head", "inference_channels"]

_T = TypeVar("_T")


def _resolve_per_head_value(
    values: Mapping[str, _T] | Sequence[_T] | None,
    head: KongNetOutputHeadSpec,
    *,
    position: int,
    default: _T,
    argument_name: str,
) -> _T:
    """Resolve a per-head override from either a mapping or ordered sequence."""
    if values is None:
        return default
    if isinstance(values, dict):
        return values.get(head.name, default)

    if position >= len(values):
        msg = (
            f"`{argument_name}` must provide a value for every selected KongNet head. "
            f"Missing value for `{head.name}` at position {position}."
        )
        raise ValueError(msg)
    return values[position]


def _resolve_head_channel_slice(
    head: KongNetOutputHeadSpec,
    *,
    target_selection: KongNetTargetSelection,
) -> int | slice | tuple[int, ...] | list[int] | None:
    """Resolve the output channel selection for one KongNet head."""
    if target_selection == "full_head":
        return head.channel_slice
    if target_selection != "inference_channels":
        msg = f"Unsupported KongNet target selection `{target_selection}`."
        raise ValueError(msg)
    if not head.target_channels:
        return None
    if len(head.target_channels) == 1:
        return head.target_channels[0]
    return list(head.target_channels)


def build_kongnet_dense_heads(
    model: KongNet,
    *,
    loss: DenseLossMode = "bce_with_logits",
    target_mode: DenseTargetMode = "binary",
    metrics: tuple[DenseMetricName, ...] | None = ("dice", "iou"),
    target_selection: KongNetTargetSelection = "full_head",
    target_keys: Mapping[str, NestedKey] | Sequence[NestedKey] | None = None,
    loss_weights: Mapping[str, float] | Sequence[float] | None = None,
) -> list[DenseHeadSpec]:
    """Build dense head specs from KongNet's named output metadata."""
    dense_heads: list[DenseHeadSpec] = []
    selected_head_specs = [
        head
        for head in model.training_output_spec
        if _resolve_head_channel_slice(head, target_selection=target_selection) is not None
    ]
    if not selected_head_specs:
        msg = (
            "KongNet did not expose any heads for the requested target selection "
            f"`{target_selection}`."
        )
        raise ValueError(msg)

    if target_keys is not None and not isinstance(target_keys, (dict, list, tuple)):
        msg = "`target_keys` must be provided as a mapping or ordered sequence."
        raise TypeError(msg)
    if isinstance(target_keys, (list, tuple)) and len(target_keys) != len(selected_head_specs):
        msg = (
            "`target_keys` must either be a mapping keyed by KongNet head name or "
            "a sequence with one entry per selected head."
        )
        raise ValueError(msg)
    if loss_weights is not None and not isinstance(loss_weights, (dict, list, tuple)):
        msg = "`loss_weights` must be provided as a mapping or ordered sequence."
        raise TypeError(msg)
    if isinstance(loss_weights, (list, tuple)) and len(loss_weights) != len(
        selected_head_specs,
    ):
        msg = (
            "`loss_weights` must either be a mapping keyed by KongNet head name or "
            "a sequence with one entry per selected head."
        )
        raise ValueError(msg)

    for position, head in enumerate(selected_head_specs):
        channel_slice = _resolve_head_channel_slice(
            head,
            target_selection=target_selection,
        )
        if channel_slice is None:
            continue

        dense_heads.append(
            DenseHeadSpec(
                name=head.name,
                loss=loss,
                target_mode=target_mode,
                target_key=_resolve_per_head_value(
                    target_keys,
                    head,
                    position=position,
                    default=head.name,
                    argument_name="target_keys",
                ),
                channel_slice=channel_slice,
                loss_weight=float(
                    _resolve_per_head_value(
                        loss_weights,
                        head,
                        position=position,
                        default=1.0,
                        argument_name="loss_weights",
                    ),
                ),
                metrics=metrics,
            ),
        )

    return dense_heads


def build_kongnet_training_task(
    model: KongNet,
    *,
    loss: DenseLossMode = "bce_with_logits",
    target_mode: DenseTargetMode = "binary",
    metrics: tuple[DenseMetricName, ...] | None = ("dice", "iou"),
    target_selection: KongNetTargetSelection = "full_head",
    target_keys: Mapping[str, NestedKey] | Sequence[NestedKey] | None = None,
    loss_weights: Mapping[str, float] | Sequence[float] | None = None,
    ignore_index: int = -100,
    prefix_head_metrics: bool = True,
    include_head_loss_metrics: bool = True,
) -> StructuredDenseTask:
    """Construct a :class:`StructuredDenseTask` directly from a KongNet model."""
    return StructuredDenseTask(
        build_kongnet_dense_heads(
            model,
            loss=loss,
            target_mode=target_mode,
            metrics=metrics,
            target_selection=target_selection,
            target_keys=target_keys,
            loss_weights=loss_weights,
        ),
        ignore_index=ignore_index,
        prefix_head_metrics=prefix_head_metrics,
        include_head_loss_metrics=include_head_loss_metrics,
    )


__all__ = [
    "KongNetTargetSelection",
    "build_kongnet_dense_heads",
    "build_kongnet_training_task",
]
