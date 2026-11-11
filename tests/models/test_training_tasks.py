"""Task selector contracts and validation of malformed training batches."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
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
        (torch.ones(2, 3), {}),
        # Some models return several outputs only in training mode.
        (torch.ones(2, 3), {"output_key": 0}),
        ([None, torch.ones(2, 3)], {"output_key": 1}),
        ((None, torch.ones(2, 3)), {"output_key": -1}),
        ({"scores": torch.ones(2, 3)}, {"output_key": "scores"}),
        ({"aux": [None, {"out": torch.ones(2, 3)}]}, {"output_key": ("aux", 1, "out")}),
    ],
)
def test_classification_output_selectors(output: object, kwargs: dict) -> None:
    """Selectors return the requested tensor unchanged."""
    task = ClassificationTask(**kwargs)
    torch.testing.assert_close(task.select_output(output), torch.ones(2, 3))
    torch.testing.assert_close(
        task.compute_loss(output, torch.tensor([0, 1])), torch.tensor(math.log(3))
    )


@pytest.mark.parametrize(
    ("output", "kwargs", "message"),
    [
        ({"metadata": None, "scores": torch.zeros(1, 2)}, {}, "Set `output_key`"),
        ((torch.zeros(1, 2), torch.zeros(1, 2)), {}, "Set `output_key`"),
        (None, {}, "Set `output_key`"),
        ({"scores": None}, {"output_key": "scores"}, "tensor output"),
        ([None], {"output_key": 0}, "tensor output"),
        ([torch.zeros(1, 2)], {"output_key": 1}, "could not be resolved"),
        ([torch.zeros(1, 2)], {"output_key": "scores"}, "could not be resolved"),
        ({"scores": torch.zeros(1, 2)}, {"output_key": "x"}, "could not be resolved"),
        ({"scores": None}, {"output_key": ("scores", "x")}, "expects nested"),
    ],
)
def test_invalid_classification_outputs(
    output: object, kwargs: dict, message: str
) -> None:
    """Outputs are never guessed: ambiguous or wrong selections raise."""
    with pytest.raises(ValueError, match=message):
        ClassificationTask(**kwargs).compute_loss(output, torch.tensor([0]))


@pytest.mark.parametrize(
    ("heads", "message"),
    [
        ([], "at least one"),
        ([DenseHeadSpec("x", "mse"), DenseHeadSpec("x", "mse")], "unique"),
        (
            [DenseHeadSpec("x", nn.L1Loss(reduction="none"), target_mode="invalid")],
            "Unsupported dense target",
        ),
        ([DenseHeadSpec("x", "mse", loss_weight=0)], "positive `loss_weight`"),
        ([DenseHeadSpec("x", "auto")], "Unsupported dense loss"),
    ],
)
def test_invalid_dense_head_configuration(heads: list, message: str) -> None:
    """Reject empty, duplicate, unsupported, or unweighted heads."""
    with pytest.raises(ValueError, match=message):
        StructuredDenseTask(heads)


@pytest.mark.parametrize("selector", ["nested", "index", "named", "tensor"])
def test_dense_output_selection(selector: str) -> None:
    """Nested keys, indices, head names and bare tensors select one prediction."""
    logits = torch.tensor([[[[2.0, -1.0]]]], requires_grad=True)
    targets = torch.tensor([[[[1.0, 0.0]]]])
    outputs = {
        "nested": {"aux": {"out": logits}},
        "index": [None, logits],
        "named": {"mask": logits, "other": None},
        "tensor": logits,
    }
    head_kwargs = {"nested": {"output_key": ("aux", "out")}, "index": {"output_key": 1}}
    task = StructuredDenseTask(
        [
            DenseHeadSpec(
                "mask",
                "bce_with_logits",
                target_key=("labels", "mask"),
                **head_kwargs.get(selector, {}),
            )
        ],
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
            "expects nested",
        ),
        ({"output_key": "missing"}, {}, torch.zeros(1), "could not be resolved"),
        ({"output_key": "out"}, {"out": None}, torch.zeros(1), "tensor output"),
        ({"output_key": 0}, {}, torch.zeros(1), "could not be resolved"),
        ({"output_key": 1}, [torch.zeros(1)], torch.zeros(1), "could not be resolved"),
        ({}, {"mask": None}, torch.zeros(1), "tensor output"),
        ({}, {"other": torch.zeros(1)}, torch.zeros(1), "named after it"),
        ({}, [torch.zeros(1)], torch.zeros(1), "Set `output_key`"),
        ({}, (torch.zeros(1), torch.zeros(1)), torch.zeros(1), "Set `output_key`"),
        ({}, torch.zeros(1, 1, 2, 2), {"mask": None}, "tensor targets"),
        ({"target_key": "mask"}, torch.zeros(1, 1, 2, 2), {}, "could not be resolved"),
    ],
)
def test_invalid_dense_selections(
    head_kwargs: dict, output: object, target: object, message: str
) -> None:
    """Malformed output and target structures cannot silently select another head."""
    task = StructuredDenseTask(
        [DenseHeadSpec("mask", "bce_with_logits", **head_kwargs)]
    )
    with pytest.raises(ValueError, match=message):
        task.compute_loss(output, target)


def test_auxiliary_outputs_are_selected_by_position() -> None:
    """A model returning a main map before auxiliary maps supervises each by index."""
    logits = [torch.zeros(1, 2, 2, 2) for _ in range(2)]
    logits[1][:, 1] = 9.0
    targets = {"mask": torch.zeros(1, 2, 2, dtype=torch.long)}
    main = SegmentationTask(output_key=0)
    expected = functional.cross_entropy(logits[0], targets["mask"])
    torch.testing.assert_close(
        main.compute_loss(tuple(logits), targets["mask"]), expected
    )
    task = StructuredDenseTask(
        [
            DenseHeadSpec("main", "cross_entropy", output_key=0, target_key="mask"),
            DenseHeadSpec("aux", "cross_entropy", output_key=1, target_key="mask"),
        ]
    )
    components = task.loss_components(tuple(logits), targets)
    torch.testing.assert_close(components["main"], expected)
    assert components["aux"] > components["main"]


DENSE_LOSSES = {
    "multiclass": "cross_entropy",
    "binary": "bce_with_logits",
    "regression": "mse",
}


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
        SegmentationTask(loss=DENSE_LOSSES[mode]).compute_loss(logits, targets)


@pytest.mark.parametrize("mode", ["multiclass", "binary", "regression"])
def test_dense_singleton_target_channel(mode: str) -> None:
    """Dense targets may omit the binary/regression channel or include a label axis."""
    channels = 2 if mode == "multiclass" else 1
    logits = torch.zeros(2, channels, 3, 4)
    targets = torch.zeros(2, 3, 4)
    task = SegmentationTask(loss=DENSE_LOSSES[mode])
    torch.testing.assert_close(
        task.compute_loss(logits, targets),
        task.compute_loss(logits, targets.unsqueeze(1)),
    )


def test_dense_heads_name_their_loss() -> None:
    """Dense heads have no default loss, and only callables declare a target mode."""
    with pytest.raises(TypeError, match="loss"):
        DenseHeadSpec("mask")
    with pytest.raises(ValueError, match="Unsupported dense loss"):
        SegmentationTask(loss="auto")
    with pytest.raises(ValueError, match="only for callable losses"):
        StructuredDenseTask([DenseHeadSpec("x", "mse", target_mode="regression")])
    with pytest.raises(ValueError, match="explicit `target_mode`"):
        StructuredDenseTask([DenseHeadSpec("x", nn.L1Loss(reduction="none"))])


def test_invalid_classification_mode() -> None:
    """Unknown target modes fail at construction."""
    with pytest.raises(ValueError, match="Unsupported classification target mode"):
        ClassificationTask(
            loss=nn.BCEWithLogitsLoss(reduction="none"), target_mode="invalid"
        )


def test_named_bce_follows_the_number_of_outputs() -> None:
    """One output is binary and several are multi-label, as metrics show."""
    logits = torch.tensor([[2.0], [-2.0], [2.0], [-2.0]])
    targets = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    task = ClassificationTask(loss="bce_with_logits")
    assert task.required_output_mode(1) == "binary"
    assert task.required_output_mode(3) == "multi_label"
    # Macro-F1 over both classes.
    assert task.compute_metrics(logits, targets)["f1"] == pytest.approx(11 / 15)
    callable_task = ClassificationTask(
        loss=nn.BCEWithLogitsLoss(reduction="none"), target_mode="multi_label"
    )
    assert callable_task.required_output_mode(1) == "multi_label"
    # F1 of the one output, with the same loss.
    assert callable_task.compute_metrics(logits, targets)["f1"] == pytest.approx(2 / 3)
    torch.testing.assert_close(
        task.compute_loss(logits, targets), callable_task.compute_loss(logits, targets)
    )
    several = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
    assert task.compute_metrics(several, torch.tensor([[1.0, 0.0], [0.0, 1.0]])) == {
        "accuracy": 1.0,
        "f1": 1.0,
    }
    with pytest.raises(ValueError, match="match the logits shape"):
        task.compute_loss(several, torch.zeros(2))


def _classification_loss(mode: str) -> nn.Module:
    """Return a callable loss, which can state any target mode."""
    if mode == "single_label":
        return nn.CrossEntropyLoss(reduction="none")
    return nn.BCEWithLogitsLoss(reduction="none")


@pytest.mark.parametrize(
    ("mode", "logit_shape", "target_shape", "message"),
    [
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
    task = ClassificationTask(target_mode=mode, loss=_classification_loss(mode))
    with pytest.raises(ValueError, match=message):
        task.compute_loss(torch.zeros(logit_shape), torch.zeros(target_shape))


@pytest.mark.parametrize("mode", ["single_label", "binary", "multi_label"])
def test_classification_singleton_dimensions(mode: str) -> None:
    """Column labels and unbatched multilabel vectors retain their loss and metrics."""
    task = ClassificationTask(target_mode=mode, loss=_classification_loss(mode))
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
