"""Define utility layers and operators for models in tiatoolbox."""

from __future__ import annotations

import sys
from typing import Callable

import numpy as np
import torch
from torch import nn

from tiatoolbox import logger


def is_torch_compile_compatible() -> NoReturn:
    """Check if the current GPU is compatible with torch-compile.

    Raises:
        Warning if GPU is not compatible with `torch.compile`.
        bool:
            True if the GPU is compatible with torch-compile, False
            otherwise.

    """
    if torch.cuda.is_available():  # pragma: no cover
        device_cap = torch.cuda.get_device_capability()
        if device_cap not in ((7, 0), (8, 0), (9, 0)):
            logger.warning(
                "GPU is not compatible with torch.compile. "
                "Compatible GPUs include NVIDIA V100, A100, and H100. "
                "Speedup numbers may be lower than expected.",
                stacklevel=2,
            )
    else:
        logger.warning(
            "No GPU detected or cuda not installed, "
            "torch.compile is only supported on selected NVIDIA GPUs. "
            "Speedup numbers may be lower than expected.",
            stacklevel=2,
        )



def compile_model(
    model: nn.Module | None = None,
    *,
    mode: str = "default",
) -> Callable:
    """A decorator to compile a model using torch-compile.

    Args:
        model (torch.nn.Module):
            Model to be compiled.
        mode (str):
            Mode to be used for torch-compile. Available modes are
            `disable`, `default`, `reduce-overhead`, `max-autotune`, and
            `max-autotune-no-cudagraphs`.

    Returns:
        Callable:
            Compiled model.

    """
    if mode == "disable":
        return model

    # Check if GPU is compatible with torch.compile
    is_torch_compile_compatible()

    # This check will be removed when torch.compile is supported in Python 3.12+
    if sys.version_info >= (3, 12):  # pragma: no cover
        logger.warning(
            ("torch-compile is currently not supported in Python 3.12+. ",),
        )
        return model

    if isinstance(  # pragma: no cover
        model,
        torch._dynamo.eval_frame.OptimizedModule,  # skipcq: PYL-W0212 # noqa: SLF001
    ):
        logger.info(
            ("The model is already compiled. ",),
        )
        return model

    return torch.compile(model, mode=mode)  # pragma: no cover


def centre_crop(
    img: np.ndarray | torch.tensor,
    crop_shape: np.ndarray | torch.tensor,
    data_format: str = "NCHW",
) -> np.ndarray | torch.Tensor:
    """A function to center crop image with given crop shape.

    Args:
        img (:class:`numpy.ndarray`, torch.Tensor):
            Input image, should be of 3 channels.
        crop_shape (:class:`numpy.ndarray`, torch.Tensor):
            The subtracted amount in the form of `[subtracted height,
            subtracted width]`.
        data_format (str):
            Either `"NCHW"` or `"NHWC"`.

    Returns:
        (:class:`numpy.ndarray`, torch.Tensor):
            Cropped image.

    """
    if data_format not in ["NCHW", "NHWC"]:
        msg = f"Unknown input format `{data_format}`."
        raise ValueError(msg)

    crop_t = crop_shape[0] // 2
    crop_b = crop_shape[0] - crop_t
    crop_l = crop_shape[1] // 2
    crop_r = crop_shape[1] - crop_l
    if data_format == "NCHW":
        return img[:, :, crop_t:-crop_b, crop_l:-crop_r]

    return img[:, crop_t:-crop_b, crop_l:-crop_r, :]


def centre_crop_to_shape(
    x: np.ndarray | torch.tensor,
    y: np.ndarray | torch.tensor,
    data_format: str = "NCHW",
) -> np.ndarray | torch.Tensor:
    """A function to center crop image to shape.

    Centre crop `x` so that `x` has shape of `y` and `y` height and
    width must be smaller than `x` height width.

    Args:
        x (:class:`numpy.ndarray`, torch.Tensor):
            Image to be cropped.
        y (:class:`numpy.ndarray`, torch.Tensor):
            Reference image for getting cropping shape, should be of 3
            channels.
        data_format:
            Either `"NCHW"` or `"NHWC"`.

    Returns:
        (:class:`numpy.ndarray`, torch.Tensor):
            Cropped image.

    """
    if data_format not in ["NCHW", "NHWC"]:
        msg = f"Unknown input format `{data_format}`."
        raise ValueError(msg)

    if data_format == "NCHW":
        _, _, h1, w1 = x.shape
        _, _, h2, w2 = y.shape
    else:
        _, h1, w1, _ = x.shape
        _, h2, w2, _ = y.shape

    if h1 <= h2 or w1 <= w2:
        raise ValueError(
            (
                "Height or width of `x` is smaller than `y` ",
                f"{[h1, w1]} vs {[h2, w2]}",
            ),
        )

    x_shape = x.shape
    y_shape = y.shape
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])

    return centre_crop(x, crop_shape, data_format)


class UpSample2x(nn.Module):
    """A layer to scale input by a factor of 2.

    This layer uses Kronecker product underneath rather than the default
    pytorch interpolation.

    """

    def __init__(self: UpSample2x) -> None:
        """Initialize :class:`UpSample2x`."""
        super().__init__()
        # correct way to create constant within module
        self.register_buffer(
            "unpool_mat",
            torch.from_numpy(np.ones((2, 2), dtype="float32")),
        )
        self.unpool_mat.unsqueeze(0)

    def forward(self: UpSample2x, x: torch.Tensor) -> torch.Tensor:
        """Logic for using layers defined in init.

        Args:
            x (torch.Tensor):
                Input images, the tensor is in the shape of NCHW.

        Returns:
            torch.Tensor:
                Input images upsampled by a factor of 2 via nearest
                neighbour interpolation. The tensor is the shape as
                NCHW.

        """
        input_shape = list(x.shape)
        # un-squeeze is the same as expand_dims
        # permute is the same as transpose
        # view is the same as reshape
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        return ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
