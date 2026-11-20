"""Define an ONNX-backed ModelABC wrapper for patch classification.

This module lets models trained in other frameworks (e.g. Keras/TensorFlow,
exported to ONNX with ``tf2onnx``) be used directly with TIAToolbox's patch
prediction pipeline, without re-implementing or converting them to PyTorch.
TensorFlow is not required at inference time: the model is executed with
ONNX Runtime.

``onnxruntime`` is an optional dependency and is imported lazily, so it is
only required when this model is instantiated::

    pip install tiatoolbox[onnx]

Example:
    >>> from tiatoolbox.models.architecture.onnx_model import ONNXModel
    >>> from tiatoolbox.models.engine.patch_predictor import PatchPredictor
    >>> model = ONNXModel(
    ...     onnx_path="my_keras_model.onnx",
    ...     num_classes=9,
    ...     input_layout="NHWC",
    ... )
    >>> predictor = PatchPredictor(model=model, batch_size=32)
    >>> output = predictor.run(
    ...     images=["patch_000.tif"],
    ...     patch_mode=True,
    ...     return_probabilities=True,
    ...     device="cpu",
    ... )

"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from tiatoolbox.models.architecture.utils import argmax_last_axis
from tiatoolbox.models.models_abc import ModelABC

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import onnxruntime as ort

#: Default per-channel mean used for preprocessing (ImageNet statistics).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
#: Default per-channel standard deviation for preprocessing (ImageNet statistics).
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Layouts an ONNX model may expect its input in.
_VALID_LAYOUTS = ("NHWC", "NCHW")

#: Number of axes of a 4D image input batch.
_INPUT_RANK = 4
#: Number of axes of a ``(batch, num_classes)`` model output.
_OUTPUT_RANK = 2

#: Channel counts a 4D image input may plausibly have. Used to detect a
#: declared ``input_layout`` that contradicts the model's input shape.
_VALID_CHANNEL_COUNTS = (1, 3, 4)


def _import_onnxruntime() -> ort:
    """Import :mod:`onnxruntime`, raising a helpful error if it is missing.

    Returns:
        Module:
            The imported ``onnxruntime`` module.

    Raises:
        ImportError:
            If ``onnxruntime`` is not installed.

    """
    try:
        import onnxruntime as ort_module  # noqa: PLC0415
    except ImportError as err:  # pragma: no cover - optional dependency
        msg = (
            "ONNXModel requires the `onnxruntime` package, which is not "
            "installed. Install it with `pip install tiatoolbox[onnx]` "
            "or `pip install onnxruntime`."
        )
        raise ImportError(msg) from err
    return ort_module


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Apply a numerically stable softmax along the last axis.

    Args:
        logits (np.ndarray):
            Unnormalised model outputs of shape ``(batch, num_classes)``.

    Returns:
        np.ndarray:
            Per-class probabilities of the same shape, as ``float32``.

    """
    shifted = logits.astype(np.float64)
    shifted -= shifted.max(axis=-1, keepdims=True)
    np.exp(shifted, out=shifted)
    shifted /= shifted.sum(axis=-1, keepdims=True)
    return shifted.astype(np.float32)


class ONNXModel(ModelABC):
    """Patch-classification model backed by an ONNX Runtime session.

    A :class:`~tiatoolbox.models.models_abc.ModelABC` subclass that loads an
    ONNX model file and plugs it into
    :class:`~tiatoolbox.models.engine.patch_predictor.PatchPredictor`, so
    models trained in Keras/TensorFlow (or any framework that exports ONNX)
    can reuse TIAToolbox's data loading and prediction engine.

    The model must take a single 4D image input and produce a single
    ``(batch, num_classes)`` output. The expected input layout is declared
    explicitly with ``input_layout`` and validated against the model's input
    shape at construction time, so a mismatch fails early instead of silently
    producing wrong predictions.

    Batches handed to :meth:`infer_batch` are NHWC, as for every other
    TIAToolbox model. They are converted to the layout the ONNX model
    expects, normalised with ``mean``/``std`` when given, and the raw
    outputs are converted to probabilities with a softmax unless
    ``apply_softmax`` is ``False`` (for models that already output
    probabilities).

    Args:
        onnx_path (str or Path):
            Path to the ``.onnx`` model file.
        num_classes (int):
            Number of classes output by the model. Validated against the
            model's output shape when that shape is fully specified.
        input_layout (str):
            Layout the ONNX model expects its input in, ``"NHWC"`` or
            ``"NCHW"``. Defaults to ``"NHWC"``.
        input_name (str or None):
            Name of the model's input node. Defaults to the model's first
            (and expected only) input.
        output_name (str or None):
            Name of the model's output node. Defaults to the model's first
            (and expected only) output.
        mean (Sequence[float] or None):
            Per-channel mean subtracted from each patch before inference,
            in the channel order of the model's input. Defaults to the
            ImageNet mean. Pass ``None`` (with ``std=None``) to disable
            normalisation.
        std (Sequence[float] or None):
            Per-channel standard deviation each patch is divided by before
            inference. Defaults to the ImageNet standard deviation. Pass
            ``None`` (with ``mean=None``) to disable normalisation.
        apply_softmax (bool):
            Whether to apply a softmax to the model outputs so
            :meth:`infer_batch` returns probabilities. Set to ``False`` if
            the model already outputs probabilities. Defaults to ``True``.
        providers (Sequence[str] or None):
            ONNX Runtime execution providers to use, e.g.
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]``. When
            ``None`` (the default), the CPU provider is used unless
            :meth:`infer_batch` is called with a CUDA ``device`` and a CUDA
            provider is available.

    Attributes:
        onnx_path (Path):
            Path to the loaded ``.onnx`` model file.
        num_classes (int):
            Number of classes output by the model.
        input_layout (str):
            Layout the ONNX model expects its input in.
        input_name (str):
            Name of the model's input node.
        output_name (str):
            Name of the model's output node.
        input_shape (tuple):
            Shape of the model's input, with ``None`` for dynamic axes.
        input_channels (int or None):
            Number of input channels, or ``None`` if the model leaves the
            channel axis dynamic.

    Example:
        >>> model = ONNXModel(
        ...     onnx_path="my_keras_model.onnx",
        ...     num_classes=9,
        ...     input_layout="NHWC",
        ... )
        >>> output = ONNXModel.infer_batch(
        ...     model,
        ...     np.random.rand(2, 224, 224, 3).astype(np.float32),
        ...     device="cpu",
        ... )
        >>> output.shape
        (2, 9)

    """

    def __init__(
        self: ONNXModel,
        onnx_path: str | Path,
        num_classes: int,
        *,
        input_layout: str = "NHWC",
        input_name: str | None = None,
        output_name: str | None = None,
        mean: Sequence[float] | None = IMAGENET_MEAN,
        std: Sequence[float] | None = IMAGENET_STD,
        apply_softmax: bool = True,
        providers: Sequence[str] | None = None,
    ) -> None:
        """Initialize :class:`ONNXModel`."""
        super().__init__()
        _import_onnxruntime()

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            msg = f"ONNX model file `{self.onnx_path}` does not exist."
            raise FileNotFoundError(msg)

        if input_layout not in _VALID_LAYOUTS:
            msg = f"input_layout must be one of {_VALID_LAYOUTS}, got `{input_layout}`."
            raise ValueError(msg)
        self.input_layout = input_layout

        if not isinstance(num_classes, int) or num_classes < 1:
            msg = f"num_classes must be a positive integer, got `{num_classes}`."
            raise ValueError(msg)
        self.num_classes = num_classes

        if (mean is None) != (std is None):
            msg = "mean and std must be given together, or both be None."
            raise ValueError(msg)
        self.apply_softmax = apply_softmax
        self._providers = list(providers) if providers is not None else None
        self._sessions: dict[str, ort.InferenceSession] = {}

        # A CPU session is enough to inspect the graph; device-specific
        # sessions are created lazily by `_session_for_device`.
        session = self._session_for_device("cpu")
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1:
            msg = (
                f"ONNXModel expects a model with a single input, "
                f"got {len(inputs)} inputs."
            )
            raise ValueError(msg)
        if len(outputs) != 1:
            msg = (
                f"ONNXModel expects a model with a single output, "
                f"got {len(outputs)} outputs."
            )
            raise ValueError(msg)

        self.input_name = input_name or inputs[0].name
        self.output_name = output_name or outputs[0].name
        if self.input_name not in {node.name for node in inputs}:
            msg = f"Input `{self.input_name}` not found in the ONNX model."
            raise ValueError(msg)
        if self.output_name not in {node.name for node in outputs}:
            msg = f"Output `{self.output_name}` not found in the ONNX model."
            raise ValueError(msg)

        self.input_shape = tuple(inputs[0].shape)
        self.input_channels = self._validate_input_shape(self.input_shape)
        self._validate_output_shape(tuple(outputs[0].shape))
        self._set_normalization(mean, std)

    def _set_normalization(
        self: ONNXModel,
        mean: Sequence[float] | None,
        std: Sequence[float] | None,
    ) -> None:
        """Validate and store the per-channel normalisation statistics.

        Args:
            mean (Sequence[float] or None):
                Per-channel mean, or ``None`` to disable normalisation.
            std (Sequence[float] or None):
                Per-channel standard deviation, or ``None`` to disable
                normalisation.

        Raises:
            ValueError:
                If only one of ``mean``/``std`` is given, or their length
                disagrees with the model's channel count.

        """
        if mean is None or std is None:
            self.mean = None
            self.std = None
            return
        if self.input_channels is not None and len(mean) != self.input_channels:
            msg = (
                f"mean has length {len(mean)} but the model expects "
                f"{self.input_channels} input channels."
            )
            raise ValueError(msg)
        if len(mean) != len(std):
            msg = (
                f"mean (length {len(mean)}) and std "
                f"(length {len(std)}) must have the same length."
            )
            raise ValueError(msg)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def _validate_input_shape(self: ONNXModel, shape: tuple) -> int | None:
        """Validate the model input shape against the declared layout.

        Args:
            shape (tuple):
                Shape of the model's input node.

        Returns:
            int or None:
                Number of input channels, or ``None`` if the channel axis
                is dynamic.

        Raises:
            ValueError:
                If the input is not 4D, or the channel axis holds a value
                that cannot be a channel count, which means the declared
                ``input_layout`` contradicts the model's input shape.

        """
        if len(shape) != _INPUT_RANK:
            msg = (
                f"ONNXModel expects a 4D image input, got shape {shape} "
                f"for input `{self.input_name}`."
            )
            raise ValueError(msg)
        channel_axis = 3 if self.input_layout == "NHWC" else 1
        channel_dim = shape[channel_axis]
        if isinstance(channel_dim, int) and channel_dim not in _VALID_CHANNEL_COUNTS:
            msg = (
                f"input_layout `{self.input_layout}` does not match the "
                f"model's input shape {shape}: axis {channel_axis} holds "
                f"{channel_dim}, which is not a channel count. Check that "
                f"the layout matches the model (NHWC vs NCHW)."
            )
            raise ValueError(msg)
        return channel_dim if isinstance(channel_dim, int) else None

    def _validate_output_shape(self: ONNXModel, shape: tuple) -> None:
        """Validate the model output shape against ``num_classes``.

        Args:
            shape (tuple):
                Shape of the model's output node.

        Raises:
            ValueError:
                If the output is not ``(batch, num_classes)``, or its class
                axis disagrees with ``num_classes`` when fully specified.

        """
        if len(shape) != _OUTPUT_RANK:
            msg = (
                f"ONNXModel expects a (batch, num_classes) output, got "
                f"shape {shape} for output `{self.output_name}`."
            )
            raise ValueError(msg)
        class_dim = shape[1]
        if isinstance(class_dim, int) and class_dim != self.num_classes:
            msg = (
                f"num_classes={self.num_classes} does not match the "
                f"model's output shape {shape}."
            )
            raise ValueError(msg)

    def _session_for_device(
        self: ONNXModel,
        device: str,
    ) -> ort.InferenceSession:
        """Get (creating if needed) the session for a device.

        Args:
            device (str):
                Device inference runs on (e.g. ``"cpu"``, ``"cuda"``).

        Returns:
            onnxruntime.InferenceSession:
                Session using the requested execution provider when
                available, falling back to CPU otherwise.

        """
        ort = _import_onnxruntime()
        key = "cuda" if str(device).startswith("cuda") else "cpu"
        if key in self._sessions:
            return self._sessions[key]

        if self._providers is not None:
            providers = self._providers
        elif key == "cuda" and (
            "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            if key == "cuda":  # pragma: no cover - needs a CUDA build
                warnings.warn(
                    "CUDA was requested but no CUDAExecutionProvider is "
                    "available; falling back to CPUExecutionProvider.",
                    stacklevel=3,
                )
            providers = ["CPUExecutionProvider"]

        session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self._sessions[key] = session
        return session

    def _predict(
        self: ONNXModel,
        batch_data: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """Run the ONNX session on an NHWC batch and return probabilities.

        Args:
            batch_data (np.ndarray):
                Batch of patches in NHWC layout, with values in ``[0, 1]``
                for floating input (integer input is scaled to ``[0, 1]``).
            device (str):
                Device inference runs on (e.g. ``"cpu"``, ``"cuda"``).

        Returns:
            np.ndarray:
                Per-class probabilities of shape ``(batch, num_classes)``.

        Raises:
            ValueError:
                If the batch is not 4D, its channel count disagrees with
                the model, or the model output is not ``(batch,
                num_classes)``.

        """
        batch = np.asarray(batch_data)
        if np.issubdtype(batch.dtype, np.integer) or np.issubdtype(
            batch.dtype,
            np.bool_,
        ):
            batch = batch.astype(np.float32) / 255.0
        else:
            batch = batch.astype(np.float32, copy=False)

        if batch.ndim != _INPUT_RANK:
            msg = f"Expected a 4D NHWC batch, got shape {batch.shape}."
            raise ValueError(msg)
        if self.input_channels is not None and batch.shape[3] != self.input_channels:
            msg = (
                f"Expected {self.input_channels} input channels, got {batch.shape[3]}."
            )
            raise ValueError(msg)

        if self.input_layout == "NCHW":
            batch = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))

        if self.mean is not None and self.std is not None:
            if self.input_layout == "NHWC":
                batch = (batch - self.mean) / self.std
            else:
                mean = self.mean.reshape(1, -1, 1, 1)
                std = self.std.reshape(1, -1, 1, 1)
                batch = (batch - mean) / std

        session = self._session_for_device(device)
        outputs = session.run(
            [self.output_name],
            {self.input_name: np.ascontiguousarray(batch)},
        )
        result = np.asarray(outputs[0])
        if result.ndim != _OUTPUT_RANK or result.shape[1] != self.num_classes:
            msg = (
                f"Model output has shape {result.shape}, expected "
                f"(batch, {self.num_classes})."
            )
            raise ValueError(msg)

        if self.apply_softmax:
            return _softmax(result)
        return result.astype(np.float32, copy=False)

    # pylint: disable=W0221
    # because abc is generic, this is actual definition
    def forward(self: ONNXModel, imgs: torch.Tensor) -> torch.Tensor:
        """Pass input data through the model.

        Args:
            imgs (torch.Tensor):
                Batch of patches in NCHW layout, as for other
                :class:`ModelABC` models.

        Returns:
            torch.Tensor:
                Per-class probabilities of shape ``(batch, num_classes)``.

        """
        batch = imgs.detach().cpu().numpy()
        batch = np.ascontiguousarray(batch.transpose(0, 2, 3, 1))
        probabilities = self._predict(batch, device=str(imgs.device))
        return torch.from_numpy(probabilities)

    @staticmethod
    def postproc(image: np.ndarray) -> np.ndarray:
        """Define the post-processing of this class of model.

        This simply applies argmax along last axis of the input.

        Args:
            image (np.ndarray):
                The input image array.

        Returns:
            np.ndarray:
                The post-processed image array.

        """
        return argmax_last_axis(image=image)

    @staticmethod
    def infer_batch(
        model: ONNXModel,
        batch_data: np.ndarray | torch.Tensor,
        device: str = "cpu",
    ) -> np.ndarray:
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o aggregation.

        Args:
            model (ONNXModel):
                ONNX-backed model.
            batch_data (np.ndarray | torch.Tensor):
                A batch of data generated by
                `torch.utils.data.DataLoader`, in NHWC layout.
            device (str):
                Transfers model to the specified device. Default is "cpu".

        Returns:
            np.ndarray:
                Per-class probabilities of shape ``(batch, num_classes)``.

        Example:
            >>> output = ONNXModel.infer_batch(model, batch_data, "cpu")
            >>> print(output.shape)

        """
        if isinstance(batch_data, torch.Tensor):
            batch = batch_data.detach().cpu().numpy()
        else:
            batch = np.asarray(batch_data)
        return model._predict(batch, device=device)
