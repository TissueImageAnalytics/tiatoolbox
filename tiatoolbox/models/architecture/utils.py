"""Define utility layers and operators for models in tiatoolbox."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from torch import nn

from tiatoolbox import logger

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from tiatoolbox.models.models_abc import ModelABC

DEFAULT_IN_CHANNELS = 3


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
    if model is None:
        msg = "`model` must not be None."
        raise ValueError(msg)

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


class Conv2dReLU(nn.Sequential):
    """Conv2d + BatchNorm + ReLU block."""

    def __init__(
        self: Conv2dReLU,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        *,
        bias: bool = False,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        """Initialize Conv2dReLU block.

        Args:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels.
            kernel_size (int):
                Convolution kernel size.
            padding (int):
                Padding applied to the input.
            stride (int):
                Convolution stride.
            bias (bool):
                If `True`, adds a learnable bias to the convolution output.
            batch_norm_eps (float):
                Epsilon value for batch normalization.
            batch_norm_momentum (float):
                Momentum value for batch normalization.

        """
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(
                out_channels,
                eps=batch_norm_eps,
                momentum=batch_norm_momentum,
            ),
            nn.ReLU(inplace=True),
        )


class EncoderMixin:
    """Mixin class adding encoder-specific functionality."""

    _is_torch_scriptable = True
    _is_torch_exportable = True
    _is_torch_compilable = True

    def __init__(self) -> None:
        """Initialize EncoderMixin defaults."""
        self._depth = 5
        self._in_channels = 3
        self._output_stride = 32
        self._out_channels: list[int] = []

    @property
    def out_channels(self) -> list[int]:
        """Return channels dimensions for each tensor of forward output of encoder."""
        return self._out_channels[: self._depth + 1]

    @property
    def output_stride(self) -> int:
        """Return the effective output stride of the encoder."""
        return min(self._output_stride, 2**self._depth)

    def set_in_channels(self, in_channels: int, *, pretrained: bool = True) -> None:
        """Change first convolution channels."""
        if in_channels == DEFAULT_IN_CHANNELS:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == DEFAULT_IN_CHANNELS:
            self._out_channels = [in_channels, *self._out_channels[1:]]

        patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)  # type: ignore[arg-type]

    def get_stages(self) -> dict[int, Sequence[torch.nn.Module]]:
        """Return encoder stages for dilation modification."""
        raise NotImplementedError

    def make_dilated(self, output_stride: int) -> None:
        """Convert encoder to a dilated version for segmentation."""
        if output_stride not in [8, 16]:
            msg = f"Output stride should be 16 or 8, got {output_stride}."
            raise ValueError(msg)

        stages = self.get_stages()
        for stage_stride, stage_modules in stages.items():
            if stage_stride <= output_stride:
                continue

            dilation_rate = stage_stride // output_stride
            for module in stage_modules:
                replace_strides_with_dilation(module, dilation_rate)


class AttentionModule(nn.Module):
    """Attention module to apply attention mechanism on feature maps."""

    def __init__(self, name: str | None, in_channels: int, reduction: int = 16) -> None:
        """Initialize the Attention module.

        Args:
            name (str | None):
                Name of the attention mechanism.
                Only "scse" is implemented. If None, identity is used.
            in_channels (int):
                Number of input channels.
            reduction (int):
                Reduction ratio for channel attention.

        """
        super().__init__()

        self.attention: nn.Module

        if name is None:
            self.attention = nn.Identity()
        elif name == "scse":
            self.attention = SCSEModule(in_channels=in_channels, reduction=reduction)
        else:
            msg = f"Attention {name} is not implemented"
            raise ValueError(msg)

    def forward(self: AttentionModule, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor):
                Input feature map of shape (N, C, H, W).

        Returns:
            torch.Tensor:
                Output feature map after applying attention.

        """
        return self.attention(x)


class SCSEModule(nn.Module):
    """Spatial and Channel Squeeze & Excitation (SCSE) module."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        """Initialize the SCSE module.

        Args:
            in_channels (int):
                Number of input channels.
            reduction (int):
                Reduction ratio for channel attention.

        """
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self: SCSEModule, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SCSE module.

        Args:
            x (torch.Tensor):
                Input feature map of shape (N, C, H, W).

        Returns:
            torch.Tensor:
                Output feature map after applying SCSE attention.

        """
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
    *,
    return_probability: bool = False,
) -> np.ndarray:
    """Post-processing function for peak detection.

    Builds a processed mask per input channel. Runs peak_local_max then
    writes 1.0 at peak pixels if return_probability is False, otherwise writes
    the confidence scores at peak locations.

    Can be called from dask.da.map_overlap on a padded NumPy block
    (h_pad, w_pad, C) to process large prediction maps in chunks with overlap.
    Keeps only centroids whose (row, col) lie in the interior window:
    rows [depth_h : depth_h + core_h), cols [depth_w : depth_w + core_w)

    Returns same spatial shape as the input block

    Args:
        block:
            NumPy array (H, W, C).
        min_distance:
            Minimum number of pixels separating peaks.
        threshold_abs:
            Minimum intensity of peaks. By default, None.
        threshold_rel:
            Minimum relative intensity of peaks. By default, None.
        block_info:
            Dask block info dict.
            Only used when called from dask.array.map_overlap.
        depth_h:
            Halo size in pixels for height (rows).
            Only used when called from dask.array.map_overlap.
        depth_w:
            Halo size in pixels for width (cols).
            Only used when it's called from dask.array.map_overlap.
        return_probability:
            If True, returns the confidence scores at peak
            locations instead of binary peak map.

    Returns:
        out:
            NumPy array (H, W, C) with 1.0 at peaks, 0 elsewhere
            if return_probability is False, otherwise with confidence
            scores at peak locations.

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
    if return_probability:
        out_probs = np.zeros(
            (block_height, block_width, block_channels), dtype=np.float32
        )

    for ch in range(block_channels):
        probs_map = np.asarray(block[..., ch])  # NumPy 2D view

        coords = peak_local_max(
            probs_map,
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
            peak_stats = regionprops(labeled_peaks, intensity_image=probs_map)
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
        detection_maps (np.ndarray):
            (H, W, C) where pixels are already local peaks.
        min_distance (int):
            Minimum distance required between ANY detections.

    Returns:
        np.ndarray:
            The filtered maps with cross-channel suppression applied.

    """
    # 1. Collapse channels to find the "Global Best" at every spatial location
    # Contains the highest probability found across all classes at each pixel.
    max_across_channels = np.max(detection_maps, axis=2)

    # 2. Handle Spatial Conflicts Across Channels (Global NMS)
    filter_size = 2 * min_distance + 1
    dilated_global_max = ndimage.maximum_filter(
        max_across_channels, size=filter_size, mode="constant", cval=0.0
    )

    # 3. Create the Keep Mask
    # A pixel is kept IF:
    # A) It is the max value across its own channels
    # B) It is the max value in its spatial neighborhood
    # C) It is non-zero
    keep_mask = (detection_maps == dilated_global_max[..., None]) & (detection_maps > 0)

    # Apply mask
    return np.where(keep_mask, detection_maps, 0)


def replace_strides_with_dilation(module: nn.Module, dilation_rate: int) -> None:
    """Replace strides with dilation in Conv2d layers.

    Converts convolutional layers to use dilation instead of stride, enabling
    atrous convolutions for semantic segmentation tasks.

    Args:
        module (nn.Module):
            Module containing Conv2d layers to patch.
        dilation_rate (int):
            Dilation rate to apply to all Conv2d layers.

    Example:
        >>> replace_strides_with_dilation(model, dilation_rate=2)

    """
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, _ = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

            # Workaround for EfficientNet
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()  # type: ignore[attr-defined]


def patch_first_conv(
    model: nn.Module,
    new_in_channels: int,
    default_in_channels: int = 3,
    *,
    pretrained: bool = True,
) -> None:
    """Update the first convolution layer for a new input channel size.

    This function updates the first convolutional layer of a model to handle
    arbitrary input channels. It optionally reuses pretrained weights or
    initializes weights randomly.

    Args:
        model (nn.Module):
            The neural network model whose first convolution layer will be patched.
        new_in_channels (int):
            Number of input channels for the new first layer.
        default_in_channels (int):
            Original number of input channels. Defaults to 3.
        pretrained (bool):
            Whether to reuse pretrained weights. Defaults to True.

    Notes:
        - If `new_in_channels` == 1 or 2 → reuse original weights.
        - If `new_in_channels` > 3 → initialize weights using Kaiming normal.

    Example:
        >>> patch_first_conv(model, new_in_channels=1, pretrained=True)

    """
    # get first conv
    conv_module: nn.Conv2d | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            conv_module = module
            break

    if conv_module is None:
        return

    weight = conv_module.weight.detach()
    conv_module.in_channels = new_in_channels

    if not pretrained:
        conv_module.weight = nn.parameter.Parameter(
            torch.Tensor(
                conv_module.out_channels,
                new_in_channels // conv_module.groups,
                *conv_module.kernel_size,
            )
        )
        conv_module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        conv_module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(
            conv_module.out_channels,
            new_in_channels // conv_module.groups,
            *conv_module.kernel_size,
        )

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        conv_module.weight = nn.parameter.Parameter(new_weight)
