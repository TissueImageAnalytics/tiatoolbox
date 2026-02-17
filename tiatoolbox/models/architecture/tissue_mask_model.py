"""Tissue Mask Detection Model Architecture.

This module defines a tissue detection model based on an EfficientNet-UNet
architecture for identifying tissue regions in digital pathology images.
The model implements an EfficientNetB0 encoder with a UNet-style decoder
and segmentation head for high-resolution tissue segmentation.

Key Components:
---------------
- SiLU:
    Sigmoid Linear Unit activation function.
- Conv2dStaticSamePadding:
    Convolutional layer with static same padding.
- MBConvBlock:
    Mobile Inverted Residual Bottleneck block.
- EfficientNetEncoder:
    EfficientNetB0 encoder for feature extraction.
- Conv2dReLU:
    Convolutional block with BatchNorm and ReLU activation.
- UnetDecoderBlock:
    Decoder block with skip connections for feature fusion.
- UnetDecoder:
    Decoder with skip connections for UNet architecture.
- SegmentationHead:
    Final layer for segmentation output.
- EfficientNetUnet:
    Main model class implementing encoder-decoder architecture for tissue detection.

Features:
---------
- ImageNet normalization during preprocessing.
- Morphological postprocessing for generating clean tissue masks.
- Efficient inference pipeline for batch processing.

Example:
    >>> from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
    >>> segmentor = SemanticSegmentor(model="tissue_mask_detection")
    >>> results = segmentor.run(
    ...     ["/example_wsi.svs"],
    ...     masks=None,
    ...     auto_get_mask=False,
    ...     patch_mode=False,
    ...     save_dir=Path("/tissue_mask/"),
    ...     output_type="annotationstore",
    ... )

"""

from __future__ import annotations

import math

import cv2
import dask.array as da
import numpy as np
import torch
from torch import nn

from tiatoolbox.models.models_abc import ModelABC


class SiLU(nn.Module):
    """Sigmoid Linear Unit (SiLU) activation function.

    Also known as Swish activation function. Computes element-wise
    x * sigmoid(x) for improved gradient flow in deep networks.

    Example:
        >>> activation = SiLU()
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = activation(x)
        >>> output.shape
        ... torch.Size([1, 64, 32, 32])

    """

    def forward(self: SiLU, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SiLU activation.

        Args:
            x (torch.Tensor):
                Input tensor of any shape.

        Returns:
            torch.Tensor:
                Output tensor with same shape as input.

        """
        return x * torch.sigmoid(x)


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolution with static same padding.

    Inherits from nn.Conv2d to match state_dict keys (weight/bias directly accessible).
    This layer computes padding dynamically based on input size to achieve
    'same' padding behavior, ensuring output spatial dimensions are predictable.

    Attributes:
        stride (tuple[int, int]):
            Stride of the convolution operation.
        kernel_size (tuple[int, int]):
            Size of the convolution kernel.
        dilation (tuple[int, int]):
            Dilation rate of the convolution.
        static_padding (nn.Module):
            Identity layer for module tree matching.

    Example:
        >>> conv = Conv2dStaticSamePadding(32, 64, kernel_size=3, stride=2)
        >>> x = torch.randn(1, 32, 128, 128)
        >>> output = conv(x)
        >>> output.shape
        ... torch.Size([1, 64, 64, 64])

    """

    def __init__(
        self: Conv2dStaticSamePadding,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        groups: int = 1,
        dilation: int | tuple[int, int] = 1,
        *,
        bias: bool = False,
        **kwargs: dict,
    ) -> None:
        """Initialize Conv2dStaticSamePadding.

        Creates a 2D convolutional layer with dynamic same padding.

        Args:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels.
            kernel_size (int | tuple[int, int]):
                Size of the convolution kernel.
            stride (int | tuple[int, int]):
                Stride of the convolution. Defaults to 1.
            bias (bool):
                If `True`, adds a learnable bias. Default: `False`.
            groups (int):
                Number of blocked connections from input to output. Defaults to 1.
            dilation (int | tuple[int, int]):
                Dilation rate of the convolution. Defaults to 1.
            **kwargs (dict):
                Additional keyword arguments for nn.Conv2d.

        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        self.stride = stride if isinstance(stride, (list, tuple)) else (stride, stride)
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, (list, tuple))
            else (kernel_size, kernel_size)
        )
        self.dilation = (
            dilation if isinstance(dilation, (list, tuple)) else (dilation, dilation)
        )

        # We define static_padding as a layer so it appears in the module tree
        # though it has no parameters to load.
        self.static_padding = nn.Identity()

    def forward(self: Conv2dStaticSamePadding, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic same padding.

        Computes padding dynamically based on input spatial dimensions to
        achieve 'same' padding behavior.

        Args:
            x (torch.Tensor):
                (B, C_in, H, W). Input tensor.

        Returns:
            torch.Tensor:
                (B, C_out, H', W'). Output tensor after convolution.

        """
        h, w = x.shape[-2:]
        extra_h = (
            (math.ceil(w / self.stride[1]) - 1) * self.stride[1]
            - w
            + self.kernel_size[1]
        )
        extra_v = (
            (math.ceil(h / self.stride[0]) - 1) * self.stride[0]
            - h
            + self.kernel_size[0]
        )
        extra_h = max(extra_h, 0)
        extra_v = max(extra_v, 0)

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        # Perform padding manually
        if left > 0 or right > 0 or top > 0 or bottom > 0:
            x = torch.nn.functional.pad(x, [left, right, top, bottom])

        return super().forward(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck block.

    This block implements the MBConv block used in EfficientNet architectures.
    It consists of expansion, depthwise convolution, squeeze-and-excitation,
    and projection phases with optional residual connection.

    Attributes:
        use_residual (bool):
            Whether to use residual connection.
        _expand_conv (nn.Module):
            1x1 convolution for channel expansion.
        _bn0 (nn.Module):
            Batch normalization after expansion.
        _swish (SiLU):
            SiLU activation function.
        _depthwise_conv (Conv2dStaticSamePadding):
            Depthwise convolution layer.
        _bn1 (nn.BatchNorm2d):
            Batch normalization after depthwise convolution.
        _se_reduce (Conv2dStaticSamePadding):
            Squeeze-and-excitation reduction layer.
        _se_expand (Conv2dStaticSamePadding):
            Squeeze-and-excitation expansion layer.
        _project_conv (Conv2dStaticSamePadding):
            1x1 convolution for projection to output channels.
        _bn2 (nn.BatchNorm2d):
            Batch normalization after projection.

    Example:
        >>> block = MBConvBlock(
        ...     in_planes=32, out_planes=64, expand_ratio=6,
        ...     kernel_size=3, stride=2
        ... )
        >>> x = torch.randn(1, 32, 64, 64)
        >>> output = block(x)
        >>> output.shape
        ... torch.Size([1, 64, 32, 32])

    """

    def __init__(
        self: MBConvBlock,
        in_planes: int,
        out_planes: int,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        reduction_ratio: int = 4,
    ) -> None:
        """Initialize MBConvBlock.

        Creates a mobile inverted residual bottleneck block with expansion,
        depthwise convolution, squeeze-and-excitation, and projection.

        Args:
            in_planes (int):
                Number of input channels.
            out_planes (int):
                Number of output channels.
            expand_ratio (int):
                Expansion ratio for the hidden dimension.
            kernel_size (int):
                Size of the depthwise convolution kernel.
            stride (int):
                Stride of the depthwise convolution.
            reduction_ratio (int):
                Reduction ratio for squeeze-and-excitation. Defaults to 4.

        """
        super().__init__()
        self.use_residual = in_planes == out_planes and stride == 1
        hidden_dim = in_planes * expand_ratio

        # 1. Expansion Phase
        # Defined as separate layers (_expand_conv, _bn0) instead of Sequential
        if expand_ratio != 1:
            self._expand_conv = Conv2dStaticSamePadding(
                in_planes, hidden_dim, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01)
        else:
            self._expand_conv = nn.Identity()
            self._bn0 = nn.Identity()

        self._swish = SiLU()

        # 2. Depthwise Convolution
        self._depthwise_conv = Conv2dStaticSamePadding(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            groups=hidden_dim,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01)

        # 3. Squeeze and Excitation
        # bias=True is required here to match the checkpoint keys
        reduced_dim = max(1, in_planes // reduction_ratio)
        self._se_reduce = Conv2dStaticSamePadding(
            hidden_dim, reduced_dim, kernel_size=1, bias=True
        )
        self._se_expand = Conv2dStaticSamePadding(
            reduced_dim, hidden_dim, kernel_size=1, bias=True
        )

        # 4. Projection
        self._project_conv = Conv2dStaticSamePadding(
            hidden_dim, out_planes, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.01)

    def forward(self: MBConvBlock, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MBConvBlock.

        Applies expansion, depthwise convolution, squeeze-and-excitation,
        projection, and optional residual connection.

        Args:
            x (torch.Tensor):
                (B, C_in, H, W). Input tensor.

        Returns:
            torch.Tensor:
                (B, C_out, H', W'). Output tensor after block processing.

        """
        residual = x

        # Expansion
        x = self._expand_conv(x)
        x = self._bn0(x)
        x = self._swish(x)

        # Depthwise
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # SE
        x_squeezed = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = x * torch.sigmoid(x_squeezed)

        # Projection
        x = self._project_conv(x)
        x = self._bn2(x)

        if self.use_residual:
            return residual + x
        return x


class EfficientNetEncoder(nn.Module):
    """EfficientNetB0 encoder for feature extraction.

    This encoder extracts multi-scale features from input images using
    EfficientNetB0 architecture. It consists of a stem convolution followed
    by multiple MBConv blocks organized into stages.

    Attributes:
        _conv_stem (Conv2dStaticSamePadding):
            Initial stem convolution layer.
        _bn0 (nn.BatchNorm2d):
            Batch normalization after stem convolution.
        _swish (SiLU):
            SiLU activation function.
        block_args (list):
            Configuration for MBConv blocks.
        _blocks (nn.ModuleList):
            List of MBConv blocks.
        _conv_head (Conv2dStaticSamePadding):
            Head convolution layer.
        _bn1 (nn.BatchNorm2d):
            Batch normalization after head convolution.
        _avg_pooling (nn.AdaptiveAvgPool2d):
            Global average pooling layer.
        _dropout (nn.Dropout):
            Dropout layer.

    Example:
        >>> encoder = EfficientNetEncoder()
        >>> x = torch.randn(1, 3, 224, 224)
        >>> features = encoder(x)
        >>> len(features)
        5
        >>> [f.shape for f in features]
        [torch.Size([1, 32, 112, 112]), torch.Size([1, 24, 56, 56]),
         torch.Size([1, 40, 28, 28]), torch.Size([1, 80, 14, 14]),
         torch.Size([1, 112, 14, 14])]

    """

    def __init__(self: EfficientNetEncoder) -> None:
        """Initialize EfficientNetEncoder.

        Sets up the EfficientNetB0 encoder with stem, MBConv blocks, and head.

        """
        super().__init__()

        self._conv_stem = Conv2dStaticSamePadding(
            3, 32, kernel_size=3, stride=2, bias=False
        )
        self._bn0 = nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)
        self._swish = SiLU()

        self.block_args = [
            [32, 16, 1, 3, 1, 1],
            [16, 24, 6, 3, 2, 2],
            [24, 40, 6, 5, 2, 2],
            [40, 80, 6, 3, 2, 3],
            [80, 112, 6, 5, 1, 3],
            [112, 192, 6, 5, 2, 4],
            [192, 320, 6, 3, 1, 1],
        ]

        self._blocks = nn.ModuleList([])
        for in_c, out_c, expand, k, s, repeats in self.block_args:
            self._blocks.append(MBConvBlock(in_c, out_c, expand, k, s))
            for _ in range(repeats - 1):
                self._blocks.append(MBConvBlock(out_c, out_c, expand, k, 1))

        self._conv_head = Conv2dStaticSamePadding(320, 1280, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(1280, eps=1e-3, momentum=0.01)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.2)

    def forward(self: EfficientNetEncoder, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through EfficientNet encoder.

        Extracts multi-scale features from input image at different stages
        of the encoder network.

        Args:
            x (torch.Tensor):
                (B, 3, H, W). Input image tensor.

        Returns:
            list[torch.Tensor]:
                List of feature maps at different scales.
                - features[0]: (B, 32, H/2, W/2)
                - features[1]: (B, 24, H/4, W/4)
                - features[2]: (B, 40, H/8, W/8)
                - features[3]: (B, 80, H/16, W/16)
                - features[4]: (B, 112, H/16, W/16)

        """
        features = []
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._swish(x)
        features.append(x)

        x = self._blocks[0](x)

        x = self._blocks[1](x)
        x = self._blocks[2](x)
        features.append(x)

        x = self._blocks[3](x)
        x = self._blocks[4](x)
        features.append(x)

        x = self._blocks[5](x)
        x = self._blocks[6](x)
        x = self._blocks[7](x)

        x = self._blocks[8](x)
        x = self._blocks[9](x)
        x = self._blocks[10](x)
        features.append(x)

        for i in range(11, 16):
            x = self._blocks[i](x)
        features.append(x)

        return features


class Conv2dReLU(nn.Sequential):
    """Conv2d + BatchNorm + ReLU block.

    This class implements a common convolutional block used in UNet decoder.
    It consists of a 2D convolution followed by batch normalization and a
    ReLU activation function.

    Attributes:
        conv (nn.Conv2d):
            Convolutional layer for feature extraction.
        norm (nn.BatchNorm2d):
            Batch normalization layer for stabilizing training.
        activation (nn.ReLU):
            ReLU activation function applied after normalization.

    Example:
        >>> block = Conv2dReLU(in_channels=32, out_channels=64)
        >>> x = torch.randn(1, 32, 128, 128)
        >>> output = block(x)
        >>> output.shape
        ... torch.Size([1, 64, 128, 128])

    """

    def __init__(
        self: Conv2dReLU,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        """Initialize Conv2dReLU block.

        Creates a convolutional layer followed by batch normalization and a ReLU
        activation function.

        Args:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels.
            kernel_size (int):
                Size of the convolution kernel. Defaults to 3.
            padding (int):
                Padding applied to the input. Defaults to 1.

        """
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )


class UnetDecoderBlock(nn.Module):
    """Decoder block for UNet architecture.

    This block performs upsampling and feature fusion using skip connections
    from the encoder. It consists of two convolutional layers with ReLU activation.

    Attributes:
        conv1 (Conv2dReLU):
            First convolutional block applied after concatenating input
            and skip features.
        attention1 (nn.Module):
            Attention mechanism applied before the first convolution
            (currently Identity).
        conv2 (Conv2dReLU):
            Second convolutional block for further refinement.
        attention2 (nn.Module):
            Attention mechanism applied after the second convolution
            (currently Identity).

    Example:
        >>> block = UnetDecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> skip = torch.randn(1, 64, 64, 64)
        >>> output = block(input_tensor, skip)
        >>> output.shape
        ... torch.Size([1, 64, 64, 64])

    """

    def __init__(
        self: UnetDecoderBlock,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        """Initialize UnetDecoderBlock.

        Creates two convolutional layers and optional attention modules for
        feature refinement during decoding.

        Args:
            in_channels (int):
                Number of input channels from the previous decoder layer.
            skip_channels (int):
                Number of channels from the skip connection.
            out_channels (int):
                Number of output channels for this block.

        """
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels)
        self.attention1 = nn.Identity()
        self.conv2 = Conv2dReLU(out_channels, out_channels)
        self.attention2 = nn.Identity()

    def forward(
        self: UnetDecoderBlock,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the decoder block.

        Upsamples the input tensor, concatenates it with the skip connection
        (if provided), and applies two convolutional layers with attention.

        Args:
            x (torch.Tensor):
                (B, C_in, H, W). Input tensor from the previous decoder layer.
            skip (torch.Tensor | None):
                (B, C_skip, H*2, W*2).
                Skip connection tensor from the encoder. Defaults to None.

        Returns:
            torch.Tensor:
                (B, C_out, H*2, W*2).
                Output tensor after decoding and feature refinement.

        """
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="nearest"
                )
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.attention1(x)
        x = self.conv2(x)
        return self.attention2(x)


class UnetDecoder(nn.Module):
    """UNet decoder with skip connections.

    This class implements the decoder portion of the UNet architecture.
    It reconstructs high-resolution feature maps from encoder outputs using
    multiple decoder blocks with skip connections.

    Attributes:
        center (nn.Module):
            Center block (currently Identity).
        blocks (nn.ModuleList):
            List of decoder blocks for upsampling and feature fusion.

    Example:
        >>> decoder = UnetDecoder()
        >>> # Generate dummy feature maps for testing
        >>> features = [
        ...     torch.randn(1, 32, 112, 112),
        ...     torch.randn(1, 24, 56, 56),
        ...     torch.randn(1, 40, 28, 28),
        ...     torch.randn(1, 80, 14, 14),
        ...     torch.randn(1, 320, 14, 14)
        ... ]
        >>> output = decoder(features)
        >>> output.shape
        ... torch.Size([1, 16, 224, 224])

    """

    def __init__(self: UnetDecoder) -> None:
        """Initialize UnetDecoder.

        Sets up the decoder blocks with skip connections for UNet architecture.

        """
        super().__init__()
        self.center = nn.Identity()
        self.blocks = nn.ModuleList(
            [
                UnetDecoderBlock(320, 112, 256),
                UnetDecoderBlock(256, 40, 128),
                UnetDecoderBlock(128, 24, 64),
                UnetDecoderBlock(64, 32, 32),
                UnetDecoderBlock(32, 0, 16),
            ]
        )

    def forward(self: UnetDecoder, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through UNet decoder.

        Reconstructs high-resolution feature maps from encoder outputs using
        skip connections and multiple decoder blocks.

        Args:
            features (list[torch.Tensor]):
                List of feature maps from the encoder, ordered from shallow to deep.

        Returns:
            torch.Tensor:
                Decoded output tensor with spatial resolution restored.

        """
        x = features[4]
        skips = features[:-1][::-1]

        x = self.center(x)
        x = self.blocks[0](x, skips[0])
        x = self.blocks[1](x, skips[1])
        x = self.blocks[2](x, skips[2])
        x = self.blocks[3](x, skips[3])
        return self.blocks[4](x)


class SegmentationHead(nn.Sequential):
    """Segmentation head for UNet architecture.

    This class defines the final segmentation layer for the UNet model.
    It applies a convolution to produce the segmentation output.

    Attributes:
        conv2d (nn.Conv2d):
            Convolutional layer for feature transformation to output classes.

    Example:
        >>> head = SegmentationHead(in_channels=16, out_channels=1)
        >>> x = torch.randn(1, 16, 224, 224)
        >>> output = head(x)
        >>> output.shape
        ... torch.Size([1, 1, 224, 224])

    """

    def __init__(
        self: SegmentationHead,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        """Initialize the SegmentationHead module.

        This method sets up the segmentation head by creating a convolutional layer.
        It is typically used as the final stage in UNet architectures for
        semantic segmentation.

        Args:
            in_channels (int):
                Number of input channels to the segmentation head.
            out_channels (int):
                Number of output channels (usually equal to the number of classes).
            kernel_size (int):
                Size of the convolution kernel. Defaults to 3.

        """
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.Identity(),
            nn.Identity(),
        )


class EfficientNetUnet(ModelABC):
    """EfficientNet-UNet Tissue Detection Model.

    This model implements a UNet architecture with an EfficientNetB0 encoder
    for tissue detection in whole slide images (WSIs). It is designed to
    identify tissue regions and background areas in digital pathology workflows.

    The model uses ImageNet normalization during preprocessing and applies
    morphological postprocessing to generate clean tissue masks.

    Attributes:
        encoder (EfficientNetEncoder):
            EfficientNetB0 encoder for feature extraction.
        decoder (UnetDecoder):
            UNet decoder for upsampling and feature fusion.
        segmentation_head (SegmentationHead):
            Final segmentation layer.

    Example:
        >>> from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
        >>> segmentor = SemanticSegmentor(model="tissue_mask")
        >>> results = segmentor.run(
        ...     ["/example_wsi.svs"],
        ...     masks=None,
        ...     auto_get_mask=False,
        ...     patch_mode=False,
        ...     save_dir=Path("/tissue_mask/"),
        ...     output_type="annotationstore",
        ... )

    """

    def __init__(
        self: EfficientNetUnet, num_classes: int = 1, threshold: float = 0.95
    ) -> None:
        """Initialize EfficientNetUnet.

        Sets up the UNet decoder, EfficientNet encoder, and segmentation head
        for tissue detection.

        Args:
            num_classes (int):
                Number of output classes. Defaults to 1 (binary segmentation).
            threshold (float):
                Threshold for binary segmentation. Defaults to 0.95.

        """
        super().__init__()
        self.encoder = EfficientNetEncoder()
        self.decoder = UnetDecoder()
        self.segmentation_head = SegmentationHead(16, num_classes)
        self.threshold = threshold

    def forward(self: EfficientNetUnet, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the EfficientNetUnet model.

        Sequentially processes the input tensor through the encoder, decoder,
        and segmentation head to produce tissue segmentation predictions.

        Args:
            x (torch.Tensor):
                (B, 3, H, W). Input image tensor.

        Returns:
            torch.Tensor:
                (B, num_classes, H, W). Segmentation output tensor.

        """
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        return self.segmentation_head(decoder_output)

    @staticmethod
    def preproc(image: np.ndarray) -> np.ndarray:
        """Preprocess input image for inference.

        Applies ImageNet normalization to the input image.

        Args:
            image (np.ndarray):
                Input image as a NumPy array of shape (H, W, C) in uint8 format.

        Returns:
            np.ndarray:
                Preprocessed image normalized to ImageNet statistics.

        Example:
            >>> img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            >>> processed = EfficientNetUnet.preproc(img)
            >>> processed.shape
            (256, 256, 3)

        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (image / 255.0 - mean) / std

    def postproc(self: EfficientNetUnet, image: np.ndarray) -> np.ndarray:
        """Postprocess model output to generate tissue mask.

        Applies thresholding and morphological operations to classify pixels
        as tissue or background and clean up the mask.

        Args:
            image (np.ndarray):
                Input probability map as a NumPy array of shape (H, W, C).

        Returns:
            np.ndarray:
                Binary tissue mask where 1 = Tissue and 0 = Background.

        Example:
            >>> probs = np.random.rand(256, 256, 1)
            >>> mask = EfficientNetUnet.postproc(probs)
            >>> mask.shape
            (256, 256)

        """
        if isinstance(image, da.Array):
            image = image.compute()
        binary_image = np.where(image[..., 0] >= self.threshold, 1, 0).astype(np.uint8)

        kernel_diameter = 31
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_diameter, kernel_diameter)
        )
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def infer_batch(
        model: torch.nn.Module,
        batch_data: torch.Tensor,
        *,
        device: str,
    ) -> np.ndarray:
        """Run inference on a batch of images.

        Transfers the model and input batch to the specified device, performs
        forward pass, and returns sigmoid probabilities.

        Args:
            model (torch.nn.Module):
                PyTorch model instance.
            batch_data (torch.Tensor):
                Batch of input images in NHWC format.
            device (str):
                Device for inference (e.g., "cpu" or "cuda").

        Returns:
            np.ndarray:
                Inference results as a NumPy array of shape (N, H, W, C).

        Example:
            >>> batch = torch.randn(4, 256, 256, 3)
            >>> probs = EfficientNetUnet.infer_batch(model, batch, device="cpu")
            >>> probs.shape
            (4, 256, 256, 1)

        """
        model = model.to(device)
        model.eval()

        imgs = batch_data
        imgs = imgs.to(device).type(torch.float32)
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        with torch.inference_mode():
            logits = model(imgs)
            probs = torch.nn.functional.sigmoid(logits)
            probs = probs.permute(0, 2, 3, 1)  # to NHWC

        return probs.cpu().numpy()
