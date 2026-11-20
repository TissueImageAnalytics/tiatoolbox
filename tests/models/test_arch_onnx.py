"""Unit tests for the ONNX-backed ModelABC wrapper.

The tests build tiny deterministic ONNX models (a flatten + linear layer)
with the ``onnx`` package, so no external weights or downloads are needed.
They are skipped when ``onnxruntime`` or ``onnx`` is not installed; the
``tiatoolbox[onnx]`` extra provides ``onnxruntime``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

import onnx
from onnx import TensorProto, helper

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

from tiatoolbox.models import IOPatchPredictorConfig
from tiatoolbox.models.architecture.onnx_model import ONNXModel
from tiatoolbox.models.engine.engine_abc import EngineABC
from tiatoolbox.models.engine.patch_predictor import PatchPredictor

RNG = np.random.default_rng(0)
NUM_CLASSES = 3
PATCH_SHAPE = (4, 4, 3)


def _make_linear_onnx_model(
    path: Path,
    *,
    input_layout: str,
    spatial: int = 4,
) -> tuple[np.ndarray, ...]:
    """Build a tiny flatten + linear ONNX model and save it.

    Args:
        path (Path):
            Where to save the ``.onnx`` file.
        input_layout (str):
            Layout of the model input, ``"NHWC"`` or ``"NCHW"``.
        spatial (int):
            Height/width of the model input. Defaults to 4; use a value
            outside ``(1, 3, 4)`` when the test needs the spatial axes to
            be unambiguous with a channel count.

    Returns:
        tuple[np.ndarray, ...]:
            The ``(weights, bias)`` used by the model, for computing the
            expected outputs in the tests.

    """
    weights = (
        RNG.standard_normal((spatial * spatial * 3, NUM_CLASSES)).astype(np.float32)
        * 0.1
    )
    bias = np.zeros(NUM_CLASSES, dtype=np.float32)
    if input_layout == "NHWC":
        input_shape = ["N", spatial, spatial, 3]
    elif input_layout == "NCHW":
        input_shape = ["N", 3, spatial, spatial]
    else:  # pragma: no cover - test helper, guarded by the caller
        msg = f"Unknown layout `{input_layout}`."
        raise ValueError(msg)

    weight_init = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        weights.shape,
        weights.tobytes(),
        raw=True,
    )
    bias_init = helper.make_tensor(
        "B",
        TensorProto.FLOAT,
        bias.shape,
        bias.tobytes(),
        raw=True,
    )
    nodes = [
        helper.make_node("Flatten", ["X"], ["Xf"], axis=1),
        helper.make_node("Gemm", ["Xf", "W", "B"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_linear",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", NUM_CLASSES])
        ],
        initializer=[weight_init, bias_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return weights, bias


def _expected_logits(
    batch_nhwc: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    input_layout: str,
) -> np.ndarray:
    """Compute the expected linear-model logits for an NHWC batch."""
    if input_layout == "NHWC":
        flat = batch_nhwc.reshape(batch_nhwc.shape[0], -1)
    else:
        flat = batch_nhwc.transpose(0, 3, 1, 2).reshape(batch_nhwc.shape[0], -1)
    return flat @ weights + bias


def _expected_probabilities(
    batch_nhwc: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    input_layout: str,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Compute the expected softmax probabilities for an NHWC batch."""
    batch = batch_nhwc.astype(np.float64)
    if mean is not None and std is not None:
        batch = (batch - np.asarray(mean)) / np.asarray(std)
    logits = _expected_logits(batch, weights, bias, input_layout=input_layout)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    return (probs / probs.sum(axis=-1, keepdims=True)).astype(np.float32)


@pytest.fixture
def nhwc_model(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """Path to a tiny NHWC ONNX model, with its (weights, bias)."""
    path = tmp_path / "tiny_nhwc.onnx"
    weights, bias = _make_linear_onnx_model(path, input_layout="NHWC")
    return path, weights, bias


def _batch(*, n: int = 2, dtype: np.dtype = np.float32) -> np.ndarray:
    """Random NHWC batch with values in [0, 1]."""
    return RNG.random((n, *PATCH_SHAPE)).astype(dtype)


def test_infer_batch_returns_probabilities(
    nhwc_model: tuple[Path, np.ndarray, np.ndarray],
) -> None:
    """infer_batch returns softmax probabilities matching a manual computation."""
    nhwc_model_path, weights, bias = nhwc_model
    model = ONNXModel(onnx_path=nhwc_model_path, num_classes=NUM_CLASSES)
    batch = _batch()
    output = ONNXModel.infer_batch(model, batch, device="cpu")

    assert output.shape == (2, NUM_CLASSES)
    assert output.dtype == np.float32
    assert np.all(output >= 0.0)
    assert np.all(output <= 1.0)
    np.testing.assert_allclose(output.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)
    expected = _expected_probabilities(
        batch,
        weights,
        bias,
        input_layout="NHWC",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_infer_batch_logits_without_preprocessing_or_softmax(
    nhwc_model: tuple[Path, np.ndarray, np.ndarray],
) -> None:
    """With mean/std disabled and no softmax, raw logits are returned."""
    nhwc_model_path, weights, bias = nhwc_model
    model = ONNXModel(
        onnx_path=nhwc_model_path,
        num_classes=NUM_CLASSES,
        mean=None,
        std=None,
        apply_softmax=False,
    )
    batch = _batch()
    output = ONNXModel.infer_batch(model, batch, device="cpu")

    expected = _expected_logits(batch, weights, bias, input_layout="NHWC")
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_infer_batch_accepts_torch_tensor(
    nhwc_model: tuple[Path, np.ndarray, np.ndarray],
) -> None:
    """infer_batch accepts an NHWC torch tensor as well as a numpy array."""
    nhwc_model_path, weights, bias = nhwc_model
    model = ONNXModel(
        onnx_path=nhwc_model_path,
        num_classes=NUM_CLASSES,
        mean=None,
        std=None,
        apply_softmax=False,
    )
    batch = torch.from_numpy(_batch())
    output = ONNXModel.infer_batch(model, batch, device="cpu")

    expected = _expected_logits(batch.numpy(), weights, bias, input_layout="NHWC")
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_infer_batch_scales_uint8_input(
    nhwc_model: tuple[Path, np.ndarray, np.ndarray],
) -> None:
    """Integer batches are scaled to [0, 1] like floating batches."""
    nhwc_model_path, _, _ = nhwc_model
    model = ONNXModel(
        onnx_path=nhwc_model_path,
        num_classes=NUM_CLASSES,
        mean=None,
        std=None,
        apply_softmax=False,
    )
    batch_uint8 = np.full((2, *PATCH_SHAPE), 255, dtype=np.uint8)
    batch_float = np.ones((2, *PATCH_SHAPE), dtype=np.float32)
    output_uint8 = ONNXModel.infer_batch(model, batch_uint8, device="cpu")
    output_float = ONNXModel.infer_batch(model, batch_float, device="cpu")
    np.testing.assert_allclose(output_uint8, output_float, rtol=1e-5, atol=1e-5)


def test_forward_accepts_nchw_torch_tensor(
    nhwc_model: tuple[Path, np.ndarray, np.ndarray],
) -> None:
    """Forward takes an NCHW torch tensor and returns probabilities."""
    nhwc_model_path, _, _ = nhwc_model
    model = ONNXModel(onnx_path=nhwc_model_path, num_classes=NUM_CLASSES)
    model.eval()
    batch_nhwc = _batch()
    batch_nchw = torch.from_numpy(
        np.ascontiguousarray(batch_nhwc.transpose(0, 3, 1, 2))
    )
    output = model(batch_nchw)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, NUM_CLASSES)
    np.testing.assert_allclose(
        output.numpy().sum(axis=1),
        1.0,
        rtol=1e-5,
        atol=1e-5,
    )
    expected = ONNXModel.infer_batch(model, batch_nhwc, device="cpu")
    np.testing.assert_allclose(output.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_nchw_layout_model(
    tmp_path: Path,
) -> None:
    """An NCHW model receives the batch transposed from NHWC."""
    path = tmp_path / "tiny_nchw.onnx"
    weights, bias = _make_linear_onnx_model(path, input_layout="NCHW")
    model = ONNXModel(
        onnx_path=path,
        num_classes=NUM_CLASSES,
        input_layout="NCHW",
        mean=None,
        std=None,
        apply_softmax=False,
    )
    batch = _batch()
    output = ONNXModel.infer_batch(model, batch, device="cpu")

    expected = _expected_logits(batch, weights, bias, input_layout="NCHW")
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_postproc_argmax() -> None:
    """Postproc takes the argmax along the last axis."""
    assert ONNXModel.postproc(np.array([0.1, 0.7, 0.2])) == 1
    np.testing.assert_array_equal(
        ONNXModel.postproc(np.array([[0.8, 0.2], [0.3, 0.9]])),
        np.array([0, 1]),
    )


def test_invalid_layout_raises(
    nhwc_model: tuple[Path, np.ndarray, np.ndarray],
) -> None:
    """An unknown input_layout is rejected."""
    nhwc_model_path, _, _ = nhwc_model
    with pytest.raises(ValueError, match=r".*input_layout.*"):
        ONNXModel(
            onnx_path=nhwc_model_path,
            num_classes=NUM_CLASSES,
            input_layout="CHWN",
        )


def test_layout_mismatch_raises(
    nhwc_model: tuple[Path, np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    """A declared layout contradicting the model shape fails at construction."""
    _, _, _ = nhwc_model
    # Spatial size 5 keeps the spatial axes unambiguous with a channel
    # count, so the layout check (not the mean-length check) fires.
    nhwc_path = tmp_path / "tiny_nhwc5.onnx"
    _make_linear_onnx_model(nhwc_path, input_layout="NHWC", spatial=5)
    with pytest.raises(ValueError, match=r".*does not match.*"):
        ONNXModel(
            onnx_path=nhwc_path,
            num_classes=NUM_CLASSES,
            input_layout="NCHW",
        )

    nchw_path = tmp_path / "tiny_nchw.onnx"
    _make_linear_onnx_model(nchw_path, input_layout="NCHW", spatial=5)
    with pytest.raises(ValueError, match=r".*does not match.*"):
        ONNXModel(
            onnx_path=nchw_path,
            num_classes=NUM_CLASSES,
            input_layout="NHWC",
        )


def test_num_classes_mismatch_raises(
    nhwc_model: tuple[Path, np.ndarray, np.ndarray],
) -> None:
    """num_classes disagreeing with the model output shape is rejected."""
    nhwc_model_path, _, _ = nhwc_model
    with pytest.raises(ValueError, match=r".*num_classes.*"):
        ONNXModel(onnx_path=nhwc_model_path, num_classes=5)


def test_missing_file_raises(tmp_path: Path) -> None:
    """A missing ONNX file is rejected."""
    with pytest.raises(FileNotFoundError, match=r".*does not exist.*"):
        ONNXModel(onnx_path=tmp_path / "nope.onnx", num_classes=NUM_CLASSES)


def test_mean_std_must_match_channels(
    nhwc_model: tuple[Path, np.ndarray, np.ndarray],
) -> None:
    """mean/std of the wrong length, or only one of them, is rejected."""
    nhwc_model_path, _, _ = nhwc_model
    with pytest.raises(ValueError, match=r".*mean.*"):
        ONNXModel(
            onnx_path=nhwc_model_path,
            num_classes=NUM_CLASSES,
            mean=(0.5, 0.5),
            std=(0.5, 0.5),
        )
    with pytest.raises(ValueError, match=r".*mean and std.*"):
        ONNXModel(
            onnx_path=nhwc_model_path,
            num_classes=NUM_CLASSES,
            mean=None,
            std=(0.229, 0.224, 0.225),
        )


def test_wrong_channel_batch_raises(
    nhwc_model: tuple[Path, np.ndarray, np.ndarray],
) -> None:
    """A batch whose channel count disagrees with the model is rejected."""
    nhwc_model_path, _, _ = nhwc_model
    model = ONNXModel(onnx_path=nhwc_model_path, num_classes=NUM_CLASSES)
    batch = RNG.random((2, 4, 4, 1)).astype(np.float32)
    with pytest.raises(ValueError, match=r".*input channels.*"):
        ONNXModel.infer_batch(model, batch, device="cpu")


def test_missing_onnxruntime_raises(
    monkeypatch: pytest.MonkeyPatch,
    nhwc_model: tuple[Path, np.ndarray, np.ndarray],
) -> None:
    """Construction fails with a helpful error when onnxruntime is absent."""
    nhwc_model_path, _, _ = nhwc_model
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match=r".*onnxruntime.*"):
        ONNXModel(onnx_path=nhwc_model_path, num_classes=NUM_CLASSES)


def test_patch_predictor_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    nhwc_model: tuple[Path, np.ndarray, np.ndarray],
) -> None:
    """The wrapper plugs into PatchPredictor like any other model."""
    nhwc_model_path, _, _ = nhwc_model
    model = ONNXModel(
        onnx_path=nhwc_model_path,
        num_classes=NUM_CLASSES,
        mean=None,
        std=None,
    )

    def _fake_get_dataloader(
        self: EngineABC,
        images: object,
        masks: object | None = None,
        labels: list | None = None,
        ioconfig: object | None = None,
        *,
        patch_mode: bool = True,
        auto_get_mask: bool = True,
        wsireader_kwargs: object | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Fake get dataloader function."""
        _ = (
            self,
            images,
            masks,
            labels,
            ioconfig,
            patch_mode,
            auto_get_mask,
            wsireader_kwargs,
        )
        batch_1 = {"image": np.zeros((1, 4, 4, 3), dtype=np.uint8)}
        batch_2 = {"image": np.ones((1, 4, 4, 3), dtype=np.uint8)}
        return [batch_1, batch_2]

    monkeypatch.setattr(EngineABC, "get_dataloader", _fake_get_dataloader)

    predictor = PatchPredictor(model=model, batch_size=2, verbose=False)
    patches = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    ioconfig = IOPatchPredictorConfig(
        patch_input_shape=(224, 224),
        stride_shape=(224, 224),
        input_resolutions=[{"units": "mpp", "resolution": 0.5}],
    )
    output = predictor.run(
        images=patches,
        patch_mode=True,
        output_type="dict",
        return_probabilities=True,
        return_labels=False,
        device="cpu",
        ioconfig=ioconfig,
    )

    assert sorted(output.keys()) == ["predictions", "probabilities"]
    assert output["probabilities"].shape == (2, NUM_CLASSES)
    assert output["predictions"].shape == (2,)
    assert np.all(output["probabilities"] >= 0.0)
    assert np.all(output["probabilities"] <= 1.0)
