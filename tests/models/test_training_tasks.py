"""Task selector contracts and validation of malformed training batches."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional

from tiatoolbox.models.training import (
    ClassificationTask,
    DenseHeadSpec,
    SegmentationTask,
    StructuredDenseTask,
)


@pytest.mark.parametrize(
    ("output", "kwargs"),
    [
        ({"metadata": None, "scores": torch.ones(2, 3)}, {}),
        ([None, torch.ones(2, 3)], {}),
        ((None, torch.ones(2, 3)), {"output_index": 1}),
        ({"scores": torch.ones(2, 3)}, {"output_key": "scores"}),
    ],
)
def test_classification_output_selectors(output: object, kwargs: dict) -> None:
    """Selectors skip metadata and return the requested tensor unchanged."""
    task = ClassificationTask(**kwargs)
    torch.testing.assert_close(task.select_output(output), torch.ones(2, 3))
    torch.testing.assert_close(
        task.compute_loss(output, torch.tensor([0, 1])), torch.tensor(math.log(3))
    )


@pytest.mark.parametrize(
    ("output", "kwargs", "error", "message"),
    [
        ({"metadata": None}, {}, ValueError, "No tensor value.*mapping"),
        ([None], {}, ValueError, "No tensor value.*sequence"),
        ({"scores": None}, {"output_key": "scores"}, ValueError, "Output key"),
        ([None], {"output_index": 0}, ValueError, "Output index"),
        (None, {}, TypeError, "Unsupported model output type"),
    ],
)
def test_invalid_classification_outputs(
    output: object, kwargs: dict, error: type[Exception], message: str
) -> None:
    """Invalid selections produce actionable errors before evaluating a loss."""
    with pytest.raises(error, match=message):
        ClassificationTask(**kwargs).compute_loss(output, torch.tensor([0]))


@pytest.mark.parametrize(
    ("heads", "message"),
    [
        ([], "at least one"),
        ([DenseHeadSpec("x"), DenseHeadSpec("x")], "unique"),
        ([DenseHeadSpec("x", target_mode="invalid")], "Unsupported dense target"),
        ([DenseHeadSpec("x", loss_weight=0)], "positive `loss_weight`"),
    ],
)
def test_invalid_dense_head_configuration(heads: list, message: str) -> None:
    """Reject empty, duplicate, unsupported, or unweighted heads."""
    with pytest.raises(ValueError, match=message):
        StructuredDenseTask(heads)


@pytest.mark.parametrize("selector", ["nested", "index", "fallback"])
def test_dense_output_selection(selector: str) -> None:
    """Nested keys, head indices, and task indices select the same prediction."""
    logits = torch.tensor([[[[2.0, -1.0]]]], requires_grad=True)
    targets = torch.tensor([[[[1.0, 0.0]]]])
    outputs = {"nested": {"aux": {"out": logits}}, "index": [None, logits]}
    outputs["fallback"] = outputs["index"]
    head_kwargs = {
        "nested": {"output_key": ("aux", "out")},
        "index": {"output_index": 1},
        "fallback": {},
    }
    task = StructuredDenseTask(
        [DenseHeadSpec("mask", target_key=("labels", "mask"), **head_kwargs[selector])],
        output_index=1,
    )
    loss = task.compute_loss(outputs[selector], {"labels": {"mask": targets}})
    torch.testing.assert_close(
        loss, functional.binary_cross_entropy_with_logits(logits, targets)
    )
    loss.backward()
    assert logits.grad is not None
    assert task.compute_metrics(outputs[selector], {"labels": {"mask": targets}}) == {
        "mask_dice": 1.0,
        "mask_iou": 1.0,
    }


@pytest.mark.parametrize(
    ("head_kwargs", "output", "target", "message"),
    [
        (
            {"output_key": ("outer", "inner")},
            {"outer": None},
            torch.zeros(1),
            "expects nested mappings",
        ),
        ({"output_key": "missing"}, {}, torch.zeros(1), "could not be resolved"),
        ({"output_key": "out"}, {"out": None}, torch.zeros(1), "tensor output"),
        ({"output_index": 0}, {}, torch.zeros(1), "requested `output_index`"),
        ({}, {"mask": None}, torch.zeros(1), "tensor output"),
        ({}, torch.zeros(1, 1, 2, 2), {"mask": None}, "tensor targets"),
        ({"target_key": "mask"}, torch.zeros(1, 1, 2, 2), {}, "could not be resolved"),
    ],
)
def test_invalid_dense_selections(
    head_kwargs: dict, output: object, target: object, message: str
) -> None:
    """Malformed output and target structures cannot silently select another head."""
    task = StructuredDenseTask([DenseHeadSpec("mask", **head_kwargs)])
    with pytest.raises(ValueError, match=message):
        task.compute_loss(output, target)


@pytest.mark.parametrize("mode", ["multiclass", "binary", "regression"])
@pytest.mark.parametrize(
    "mismatch", ["logits_rank", "targets_rank", "batch", "spatial", "channels"]
)
def test_invalid_dense_shapes(mode: str, mismatch: str) -> None:
    """Each dense objective validates ranks, batch size, spatial size, and channels."""
    logits = torch.zeros(2, 2, 3, 4)
    targets = torch.zeros(2, 2, 3, 4) if mode != "multiclass" else torch.zeros(2, 3, 4)
    if mismatch == "logits_rank":
        logits = logits.flatten(2)
        message = "expect logits with shape"
    elif mismatch == "targets_rank":
        targets = targets.flatten(1)
        message = "targets must have shape"
    elif mismatch == "batch":
        targets = targets[:1]
        message = "batch size"
    elif mismatch == "spatial":
        targets = targets[..., :2]
        message = "spatial shape"
    elif mode == "multiclass":
        logits = logits[:, :1]
        message = "C >= 2"
    else:
        targets = targets[:, :1]
        message = "target shape"
    with pytest.raises(ValueError, match=message):
        SegmentationTask(loss="auto", target_mode=mode).compute_loss(logits, targets)


@pytest.mark.parametrize("mode", ["multiclass", "binary", "regression"])
def test_dense_singleton_target_channel(mode: str) -> None:
    """Dense targets may omit the binary/regression channel or include a label axis."""
    channels = 2 if mode == "multiclass" else 1
    logits = torch.zeros(2, channels, 3, 4)
    targets = torch.zeros(2, 3, 4)
    task = SegmentationTask(loss="auto", target_mode=mode)
    torch.testing.assert_close(
        task.compute_loss(logits, targets),
        task.compute_loss(logits, targets.unsqueeze(1)),
    )


@pytest.mark.parametrize("channels", [1, 2])
def test_dense_auto_loss(channels: int) -> None:
    """Automatic heads choose binary or multiclass loss from the channel count."""
    loss = SegmentationTask(loss="auto").compute_loss(
        torch.zeros(2, channels, 3, 4), torch.zeros(2, 3, 4)
    )
    assert loss.item() == pytest.approx(math.log(2))


def test_invalid_classification_mode() -> None:
    """Unknown target modes fail at construction."""
    with pytest.raises(ValueError, match="Unsupported classification target mode"):
        ClassificationTask(target_mode="invalid")


@pytest.mark.parametrize(
    ("mode", "logit_shape", "target_shape", "message"),
    [
        ("auto", (2, 3), (2,), "Unable to infer"),
        ("single_label", (2,), (2,), "expects logits"),
        ("single_label", (2, 1), (2,), "C >= 2"),
        ("single_label", (2, 3), (2, 2, 1), "targets must have shape"),
        ("binary", (2, 3), (2,), "expects logits"),
        ("binary", (2,), (2, 2), "targets must have shape"),
        ("binary", (2,), (3,), "match the logits shape"),
        ("multi_label", (2, 3, 1), (2, 3), "expects logits"),
        ("multi_label", (2, 3), (2, 3, 1), "targets must have shape"),
        ("multi_label", (2, 3), (2, 4), "match the logits shape"),
    ],
)
def test_invalid_classification_shapes(
    mode: str, logit_shape: tuple, target_shape: tuple, message: str
) -> None:
    """Invalid classifier tensor shapes fail with task-specific errors."""
    loss = "cross_entropy" if mode == "single_label" else "bce_with_logits"
    task = ClassificationTask(target_mode=mode, loss=loss)
    with pytest.raises(ValueError, match=message):
        task.compute_loss(torch.zeros(logit_shape), torch.zeros(target_shape))


@pytest.mark.parametrize("mode", ["single_label", "binary", "multi_label"])
def test_classification_singleton_dimensions(mode: str) -> None:
    """Column labels and unbatched multilabel vectors retain their loss and metrics."""
    loss = "cross_entropy" if mode == "single_label" else "bce_with_logits"
    task = ClassificationTask(target_mode=mode, loss=loss)
    if mode == "multi_label":
        logits, targets = torch.tensor([[2.0, 1.0]]), torch.tensor([[1.0, 1.0]])
        actual_logits, actual_targets = logits[0], targets[0]
    else:
        logits = (
            torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
            if mode == "single_label"
            else torch.tensor([-1.0, 2.0])
        )
        targets = torch.tensor([0, 1])
        actual_logits, actual_targets = logits, targets[:, None]
    torch.testing.assert_close(
        task.compute_loss(actual_logits, actual_targets),
        task.compute_loss(logits, targets),
    )
    assert task.compute_metrics(actual_logits, actual_targets) == {
        "accuracy": 1.0,
        "f1": 1.0,
    }


@pytest.mark.parametrize(
    "method", ["compute_loss", "compute_metrics", "loss_normalizers"]
)
def test_classification_requires_tensor_targets(method: str) -> None:
    """All task entry points reject non-tensor targets consistently."""
    args = ([0, 1], 2) if method == "loss_normalizers" else (torch.zeros(2, 3), [0, 1])
    with pytest.raises(ValueError, match="require tensor targets"):
        getattr(ClassificationTask(), method)(*args)
