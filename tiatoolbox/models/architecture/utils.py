"""Define utility layers and operators for models in tiatoolbox."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast

import numpy as np
import scipy.ndimage as ndimage

import torch
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from torch import nn

from tiatoolbox import logger

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.models.models_abc import ModelABC


def is_torch_compile_compatible() -> bool:
    """Check if the current GPU is compatible with torch-compile.

    Returns:
        True if current GPU is compatible with torch-compile, False otherwise.

    Raises:
        Warning if GPU is not compatible with `torch.compile`.

    """
    gpu_compatibility = True
    if torch.cuda.is_available():  # pragma: no cover
        device_cap = torch.cuda.get_device_capability()
        if device_cap not in ((7, 0), (8, 0), (9, 0)):
            logger.warning(
                "GPU is not compatible with torch.compile. "
                "Compatible GPUs include NVIDIA V100, A100, and H100. "
                "Speedup numbers may be lower than expected.",
                stacklevel=2,
            )
            gpu_compatibility = False
    else:
        logger.warning(
            "No GPU detected or cuda not installed, "
            "torch.compile is only supported on selected NVIDIA GPUs. "
            "Speedup numbers may be lower than expected.",
            stacklevel=2,
        )
        gpu_compatibility = False

    return gpu_compatibility


def compile_model(
    model: nn.Module | ModelABC | None = None,
    *,
    mode: str = "default",
) -> torch.nn.Module | ModelABC:
    """A decorator to compile a model using torch-compile.

    Args:
        model (torch.nn.Module):
            Model to be compiled.
        mode (str):
            Mode to be used for torch-compile. Available modes are:

            - `disable` disables torch-compile
            - `default` balances performance and overhead
            - `reduce-overhead` reduces overhead of CUDA graphs (useful for small
              batches)
            - `max-autotune` leverages Triton/template based matrix multiplications
              on GPUs
            - `max-autotune-no-cudagraphs` similar to “max-autotune” but without
              CUDA graphs

    Returns:
        torch.nn.Module or ModelABC:
            Compiled model.

    """
    if mode == "disable":
        return model

    # Check if GPU is compatible with torch.compile
    gpu_compatibility = is_torch_compile_compatible()

    if not gpu_compatibility:
        return model

    if sys.platform == "win32":  # pragma: no cover
        msg = (
            "`torch.compile` is not supported on Windows. Please see "
            "https://github.com/pytorch/pytorch/issues/122094."
        )
        logger.warning(msg=msg)
        return model

    if isinstance(  # pragma: no cover
        model,
        torch._dynamo.eval_frame.OptimizedModule,  # skipcq: PYL-W0212 # noqa: SLF001
    ):
        logger.info(
            ("The model is already compiled. ",),
        )
        return model

    return cast("nn.Module", torch.compile(model, mode=mode))  # pragma: no cover


def centre_crop(
    img: np.ndarray | torch.Tensor,
    crop_shape: np.ndarray | torch.Tensor | tuple[int, int],
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

    crop_t: int = int(crop_shape[0] // 2)
    crop_b: int = int(crop_shape[0] - crop_t)
    crop_l: int = int(crop_shape[1] // 2)
    crop_r: int = int(crop_shape[1] - crop_l)
    if data_format == "NCHW":
        return img[:, :, crop_t:-crop_b, crop_l:-crop_r]

    return img[:, crop_t:-crop_b, crop_l:-crop_r, :]


def centre_crop_to_shape(
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
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
        self.unpool_mat: torch.Tensor
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
    
class SegmentationHead(nn.Sequential):
    """Segmentation head for UNet++ architecture.

    This class defines the final segmentation layer for the UNet++ model.
    It applies a convolution followed by optional upsampling and activation
    to produce the segmentation output.

    Attributes:
        conv2d (nn.Conv2d):
            Convolutional layer for feature transformation.
        upsampling_layer (nn.Module):
            Upsampling layer (bilinear interpolation or identity).
        activation (nn.Module):
            Activation function applied after upsampling.

    Example:
        >>> head = SegmentationHead(in_channels=64, out_channels=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = head(x)
        >>> output.shape
        ... torch.Size([1, 2, 128, 128])

    """

    def __init__(
        self: SegmentationHead,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: nn.Module | None = None,
        upsampling: int = 1,
    ) -> None:
        """Initialize the SegmentationHead module.

        This method sets up the segmentation head by creating a convolutional layer,
        an optional upsampling layer, and an activation function. It is typically
        used as the final stage in UNet++ architectures for semantic segmentation.

        Args:
            in_channels (int):
                Number of input channels to the segmentation head.
            out_channels (int):
                Number of output channels (usually equal to the number of classes).
            kernel_size (int):
                Size of the convolution kernel. Defaults to 3.
            activation (nn.Module | None):
                Activation function applied after convolution. Defaults to None.
            upsampling (int):
                Upsampling factor applied to the output. Defaults to 1.

        Raises:
            ValueError:
                If `kernel_size` or `upsampling` is not a positive integer.

        """
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling_layer = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        if activation is None:
            activation = nn.Identity()
        super().__init__(conv2d, upsampling_layer, activation)


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
    



def argmax_last_axis(image: np.ndarray) -> np.ndarray:
    """Define the post-processing of this class of model.

    This simply applies argmax along last axis of the input.

    Args:
        image (np.ndarray):
            The input image array.

    Returns:
        np.ndarray:
            The post-processed image array.

    """
    return image.argmax(axis=-1)


def peak_detection_map_overlap(
    block: np.ndarray,
    min_distance: int,
    threshold_abs: float | None = None,
    threshold_rel: float | None = None,
    block_info: dict | None = None,
    depth_h: int = 0,
    depth_w: int = 0,
    return_probability: bool = False,
) -> np.ndarray:
    """Post-processing function for peak detection.

    Builds a processed mask per input channel. Runs peak_local_max then
    writes 1.0 at peak pixels.

    Can be called from dask.da.map_overlap on a padded NumPy block
    (h_pad, w_pad, C) to process large prediction maps in chunks with overlap.
    Keeps only centroids whose (row, col) lie in the interior window:
    rows [depth_h : depth_h + core_h), cols [depth_w : depth_w + core_w)

    Returns same spatial shape as the input block

    Args:
        block: NumPy array (H, W, C).
        min_distance: Minimum number of pixels separating peaks.
        threshold_abs: Minimum intensity of peaks. By default, None.
        threshold_rel: Minimum relative intensity of peaks. By default, None.
        block_info: Dask block info dict.
            Only used when called from dask.array.map_overlap.
        depth_h: Halo size in pixels for height (rows).
            Only used when called from dask.array.map_overlap.
        depth_w: Halo size in pixels for width (cols).
            Only used when it's called from dask.array.map_overlap.

    Returns:
        out: NumPy array (H, W, C) with 1.0 at peaks, 0 elsewhere.

    """
    block_height, block_width, block_channels = block.shape

    # --- derive core (pre-overlap) size for THIS block ---
    if block_info is None:
        core_h = block_height - 2 * depth_h
        core_w = block_width - 2 * depth_w
    else:
        info = block_info[0]
        locs = info["array-location"]  # a list of (start, stop) coordinates per axis
        core_h = int(locs[0][1] - locs[0][0])  # r1 - r0
        core_w = int(locs[1][1] - locs[1][0])

    rmin, rmax = depth_h, depth_h + core_h
    cmin, cmax = depth_w, depth_w + core_w

    out = np.zeros((block_height, block_width, block_channels), dtype=np.float32)
    out_probs = np.zeros((block_height, block_width, block_channels), dtype=np.float32)

    for ch in range(block_channels):
        img = np.asarray(block[..., ch])  # NumPy 2D view

        coords = peak_local_max(
            img,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            threshold_rel=threshold_rel,
            exclude_border=False,
        )

        for r, c in coords:
            if (rmin <= r < rmax) and (cmin <= c < cmax):
                out[r, c, ch] = 1.0

        if return_probability:
            labeled_peaks = label(out[..., ch])
            peak_stats = regionprops(labeled_peaks, intensity_image=img)
            for peak in peak_stats:
                centroid = peak["centroid"]
                r, c, confidence = (
                    centroid[0],
                    centroid[1],
                    peak["mean_intensity"],
                )
                out_probs[int(r), int(c), ch] = confidence
        
    return out if not return_probability else out_probs



def nms_on_detection_maps(
    detection_maps: np.ndarray,
    min_distance: int,
) -> np.ndarray:
    """Apply NMS to pre-processed peak maps to handle cross-channel conflicts.

    Args:
        detection_maps (np.ndarray): Sparse input (H, W, C) where pixels are already local peaks.
        min_distance (int): Minimum distance required between ANY detections (even across classes). 

    Returns:
        np.ndarray: The filtered maps with cross-channel suppression applied.
    """
    # 1. Collapse channels to find the "Global Best" at every spatial location
    # Shape becomes (H, W). Contains the highest probability found across all classes at each pixel.
    max_across_channels = np.max(detection_maps, axis=2)

    # 2. Handle Spatial Conflicts Across Channels (Global NMS)
    filter_size = 2 * min_distance + 1
    dilated_global_max = ndimage.maximum_filter(
        max_across_channels, 
        size=filter_size,
        mode='constant',
        cval=0.0
    )

    # 3. Create the Keep Mask
    # A pixel is kept IF:
    # A) It is the max value across its own channels
    # B) It is the max value in its spatial neighborhood
    # C) It is non-zero
    keep_mask = (detection_maps == dilated_global_max[..., None]) & (detection_maps > 0)

    # Apply mask
    return np.where(keep_mask, detection_maps, 0)
