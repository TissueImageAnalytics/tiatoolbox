"""GrandQC Tissue Detection Model Architecture [1].

This module defines the GrandQC model for tissue detection in digital pathology.
It implements a UNet++ architecture with an EfficientNetB0 encoder and a segmentation
head for high-resolution tissue segmentation. The model is designed to identify
tissue regions and background areas for quality control in whole slide images (WSIs).
Please cite the paper [1], if you use this model.

Key Components:
---------------
- SegmentationHead:
    Final layer for segmentation output.
- Conv2dReLU:
    Convolutional block with BatchNorm and ReLU activation.
- DecoderBlock:
    Decoder block with skip connections for feature fusion.
- CenterBlock:
    Bottleneck block for deep feature processing.
- UnetPlusPlusDecoder:
    Decoder with dense skip connections for UNet++ architecture.
- GrandQCModel:
    Main model class implementing encoder-decoder architecture for tissue detection.

Features:
---------
- JPEG compression and ImageNet normalization during preprocessing.
- Argmin-based postprocessing for generating tissue masks.
- Efficient inference pipeline for batch processing.

Example:
    >>> from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
    >>> segmentor = SemanticSegmentor(model="grandqc_tissue_detection_mpp10")
    >>> results = segmentor.run(
    ...     ["/example_wsi.svs"],
    ...     masks=None,
    ...     auto_get_mask=False,
    ...     patch_mode=False,
    ...     save_dir=Path("/tissue_mask/"),
    ...     output_type="annotationstore",
    ... )

References:
    [1] Weng, Zhilong et al. "GrandQC: A comprehensive solution to quality control
    problem in digital pathology." Nature Communications, 2024.
    DOI: 10.1038/s41467-024-54769-y
    URL: https://doi.org/10.1038/s41467-024-54769-y

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

import cv2
import numpy as np
import torch
from torch import nn

from tiatoolbox.models.architecture.timm_efficientnet import EfficientNetEncoder
from tiatoolbox.models.models_abc import ModelABC


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


class Conv2dReLU(nn.Sequential):
    """Conv2d + BatchNorm + ReLU block.

    This class implements a common convolutional block used in encoder-decoder
    architectures. It consists of a 2D convolution followed by batch normalization
    and a ReLU activation function.

    Attributes:
        conv (nn.Conv2d):
            Convolutional layer for feature extraction.
        norm (nn.BatchNorm2d):
            Batch normalization layer for stabilizing training.
        activation (nn.ReLU):
            ReLU activation function applied after normalization.

    Example:
        >>> block = Conv2dReLU(
        ... in_channels=32, out_channels=64, kernel_size=3, padding=1
        ... )
        >>> x = torch.randn(1, 32, 128, 128)
        >>> output = block(x)
        >>> output.shape
        ... torch.Size([1, 64, 128, 128])

    """

    def __init__(
        self: Conv2dReLU,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        """Initialize Conv2dReLU block.

        Creates a convolutional layer followed by batch normalization and a ReLU
        activation function. This block is commonly used in UNet++ and similar
        architectures for feature extraction.

        Args:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels.
            kernel_size (int):
                Size of the convolution kernel.
            padding (int):
                Padding applied to the input. Defaults to 0.
            stride (int):
                Stride of the convolution. Defaults to 1.

        """
        norm = nn.BatchNorm2d(out_channels)

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        activation = nn.ReLU(inplace=True)

        super().__init__(conv, norm, activation)


class DecoderBlock(nn.Module):
    """Decoder block for UNet++ architecture.

    This block performs upsampling and feature fusion using skip connections
    from the encoder. It consists of two convolutional layers with ReLU activation
    and optional attention mechanisms (not implemented).

    Attributes:
        conv1 (Conv2dReLU):
            First convolutional block applied after concatenating input
            and skip features.
        conv2 (Conv2dReLU):
            Second convolutional block for further refinement.
        attention1 (nn.Module):
            Attention mechanism applied before the first convolution
            (currently Identity).
        attention2 (nn.Module):
            Attention mechanism applied after the second convolution
            (currently Identity).

    Example:
        >>> block = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> skip = torch.randn(1, 64, 128, 128)
        >>> output = block(input_tensor, skip)
        >>> output.shape
        ... torch.Size([1, 64, 128, 128])

    """

    def __init__(
        self: DecoderBlock,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        """Initialize DecoderBlock.

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
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.attention1 = nn.Identity()
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.attention2 = nn.Identity()

    def forward(
        self: DecoderBlock,
        input_tensor: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the decoder block.

        Upsamples the input tensor, concatenates it with the skip connection
        (if provided), and applies two convolutional layers with attention.

        Args:
            input_tensor (torch.Tensor):
                (B, C_in, H, W). Input tensor from the previous decoder layer.
            skip (torch.Tensor | None):
                (B, C_skip, H*2, W*2).
                Skip connection tensor from the encoder. Defaults to None.

        Returns:
            torch.Tensor:
                (B, C_out, H*2, W*2).
                Output tensor after decoding and feature refinement.

        """
        input_tensor = torch.nn.functional.interpolate(
            input_tensor, scale_factor=2.0, mode="nearest"
        )
        if skip is not None:
            input_tensor = torch.cat([input_tensor, skip], dim=1)
            input_tensor = self.attention1(input_tensor)
        input_tensor = self.conv1(input_tensor)
        input_tensor = self.conv2(input_tensor)
        return self.attention2(input_tensor)


class CenterBlock(nn.Sequential):
    """Center block for UNet++ architecture.

    This block can be placed at the bottleneck of the UNet++ architecture.
    It consists of two convolutional layers with ReLU activation, used
    to process the deepest feature maps before decoding begins.

    Attributes:
        conv1 (Conv2dReLU):
            First convolutional block for feature transformation.
        conv2 (Conv2dReLU):
            Second convolutional block for further refinement.

    Example:
        >>> center = CenterBlock(in_channels=256, out_channels=512)
        >>> input_tensor = torch.randn(1, 256, 32, 32)
        >>> output = center(input_tensor)
        >>> output.shape
        ... torch.Size([1, 512, 32, 32])

    """

    def __init__(
        self: CenterBlock,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """Initialize CenterBlock.

        Creates two convolutional layers with batch normalization and ReLU
        activation for processing the deepest encoder features.

        Args:
            in_channels (int):
                Number of input channels from the encoder.
            out_channels (int):
                Number of output channels for the center block.

        """
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        super().__init__(conv1, conv2)


class UnetPlusPlusDecoder(nn.Module):
    """UNet++ decoder with dense skip connections.

    This class implements the decoder portion of the UNet++ architecture.
    It reconstructs high-resolution feature maps from encoder outputs using
    multiple decoder blocks and dense connections between intermediate layers.

    Raises:
        ValueError:
            If the number of decoder blocks does not match the length of
            `decoder_channels`.

    Attributes:
        blocks (nn.ModuleDict):
            Dictionary of decoder blocks organized by depth and layer index.
        center (nn.Module):
            Center block (currently Identity).
        depth (int):
            Depth of the decoder network.

    Example:
        >>> decoder = UnetPlusPlusDecoder(
        ...     encoder_channels=[3, 32, 64, 128, 256, 512],
        ...     decoder_channels=[256, 128, 64, 32, 16],
        ...     n_blocks=5
        ... )
        >>> # Generate dummy feature maps for testing
        >>> features = [
        ...     torch.randn(1, c, 64 // (2**i), 64 // (2**i))
        ...     for i, c in enumerate([3, 32, 64, 128, 256, 512])
        ... ]
        >>> output = decoder(features)
        >>> output.shape
        ... torch.Size([1, 16, 64, 64])

    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        n_blocks: int = 5,
    ) -> None:
        """Initialize UnetPlusPlusDecoder.

        Sets up the decoder blocks and dense connections for UNet++ architecture.

        Args:
            encoder_channels (Sequence[int]):
                List of channel sizes from the encoder stages.
            decoder_channels (Sequence[int]):
                List of channel sizes for each decoder block.
            n_blocks (int):
                Number of decoder blocks. Defaults to 5.

        Raises:
            ValueError:
                If `n_blocks` does not match the length of `decoder_channels`.

        """
        super().__init__()

        if n_blocks != len(decoder_channels):
            msg = (
                f"Model depth is {n_blocks}, but you provide "
                f"`decoder_channels` for {len(decoder_channels)} blocks."
            )
            raise ValueError(msg)

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels, *list(decoder_channels[:-1])]
        self.skip_channels = [*list(encoder_channels[1:]), 0]
        self.out_channels = decoder_channels

        self.center = nn.Identity()

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (
                        layer_idx + 1 - depth_idx
                    )
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(
                    in_ch, skip_ch, out_ch
                )
        blocks[f"x_{0}_{len(self.in_channels) - 1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1]
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through UNet++ decoder.

        Reconstructs high-resolution feature maps from encoder outputs using
        dense skip connections and multiple decoder blocks.

        Args:
            features (list[torch.Tensor]):
                List of feature maps from the encoder, ordered from shallow to deep.

        Returns:
            torch.Tensor:
                Decoded output tensor with spatial resolution restored.

        """
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](
                        features[depth_idx], features[depth_idx + 1]
                    )
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [
                        dense_x[f"x_{idx}_{dense_l_i}"]
                        for idx in range(depth_idx + 1, dense_l_i + 1)
                    ]
                    cat_features = torch.cat(
                        [*cat_features, features[dense_l_i + 1]], dim=1
                    )
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[
                        f"x_{depth_idx}_{dense_l_i}"
                    ](dense_x[f"x_{depth_idx}_{dense_l_i - 1}"], cat_features)
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](
            dense_x[f"x_{0}_{self.depth - 1}"]
        )
        return dense_x[f"x_{0}_{self.depth}"]


class GrandQCModel(ModelABC):
    """GrandQC Tissue Detection Model.

    This model implements a UNet++ architecture with an EfficientNet encoder
    for tissue detection in whole slide images (WSIs). It is designed to
    identify tissue regions and background areas for quality control in
    digital pathology workflows.

    The model uses JPEG compression and ImageNet normalization during
    preprocessing and applies argmin-based postprocessing to generate
    tissue masks.

    Example:
        >>> from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
        >>> segmentor = SemanticSegmentor(model="grandqc_tissue_detection")
        >>> results = segmentor.run(
        ...     ["/example_wsi.svs"],
        ...     masks=None,
        ...     auto_get_mask=False,
        ...     patch_mode=False,
        ...     save_dir=Path("/tissue_mask/"),
        ...     output_type="annotationstore",
        ... )

    References:
        [1] Weng, Zhilong et al. "GrandQC: A comprehensive solution to quality control
        problem in digital pathology." Nature Communications, 2024.
        DOI: 10.1038/s41467-024-54769-y
        URL: https://doi.org/10.1038/s41467-024-54769-y

    """

    def __init__(self: GrandQCModel, num_output_channels: int = 2) -> None:
        """Initialize GrandQCModel.

        Sets up the UNet++ decoder, EfficientNet encoder, and segmentation head
        for tissue detection.

        Args:
            num_output_channels (int):
                Number of output classes. Defaults to 2 (Tissue and Background).

        """
        super().__init__()
        self.num_output_channels = num_output_channels
        self.decoder_channels = (256, 128, 64, 32, 16)

        self.encoder = EfficientNetEncoder(
            out_channels=[3, 32, 24, 40, 112, 320],
            stage_idxs=[2, 3, 5],
            channel_multiplier=1.0,
            depth_multiplier=1.0,
            drop_rate=0.2,
        )
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=5,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=num_output_channels,
            kernel_size=3,
        )

        self.name = "unetplusplus-efficientnetb0"

    def forward(  # skipcq: PYL-W0613
        self: GrandQCModel,
        x: torch.Tensor,
        *args: tuple[Any, ...],  # noqa: ARG002
        **kwargs: dict,  # noqa: ARG002
    ) -> torch.Tensor:
        """Forward pass through the GrandQC model.

        Sequentially processes the input tensor through the encoder, decoder,
        and segmentation head to produce tissue segmentation predictions.

        Args:
            x (torch.Tensor):
                Input tensor of shape (N, C, H, W).
            *args (tuple):
                Additional positional arguments (unused).
            **kwargs (dict):
                Additional keyword arguments (unused).

        Returns:
            torch.Tensor:
                Segmentation output tensor of shape (N, num_classes, H, W).

        """
        features = self.encoder(x)
        decoder_output = self.decoder(features)

        return self.segmentation_head(decoder_output)

    @staticmethod
    def preproc(image: np.ndarray) -> np.ndarray:
        """Preprocess input image for inference.

        Applies JPEG compression and ImageNet normalization to the input image.

        Args:
            image (np.ndarray):
                Input image as a NumPy array of shape (H, W, C) in uint8 format.

        Returns:
            np.ndarray:
                Preprocessed image normalized to ImageNet statistics.

        Example:
            >>> img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            >>> processed = GrandQCModel.preproc(img)
            >>> processed.shape
            ... (256, 256, 3)

        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, compressed_image = cv2.imencode(".jpg", image, encode_param)
        compressed_image = np.array(cv2.imdecode(compressed_image, 1))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (compressed_image / 255.0 - mean) / std

    @staticmethod
    def postproc(image: np.ndarray) -> np.ndarray:
        """Postprocess model output to generate tissue mask.

        Applies argmin across channels to classify pixels as tissue or background.

        Args:
            image (np.ndarray):
                Input probability map as a NumPy array of shape (H, W, C).

        Returns:
            np.ndarray:
                Binary tissue mask where 0 = Tissue and 1 = Background.

        Example:
            >>> probs = np.random.rand(256, 256, 2)
            >>> mask = GrandQCModel.postproc(probs)
            >>> mask.shape
            ... (256, 256)

        """
        return image.argmin(axis=-1)

    @staticmethod
    def infer_batch(
        model: torch.nn.Module,
        batch_data: torch.Tensor,
        *,
        device: str,
    ) -> np.ndarray:
        """Run inference on a batch of images.

        Transfers the model and input batch to the specified device, performs
        forward pass, and returns softmax probabilities.

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
            >>> probs = GrandQCModel.infer_batch(model, batch, device="cpu")
            >>> probs.shape
            (4, 256, 256, 2)

        """
        model = model.to(device)
        model.eval()

        imgs = batch_data
        imgs = imgs.to(device).type(torch.float32)
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        with torch.inference_mode():
            logits = model(imgs)
            probs = torch.nn.functional.softmax(logits, 1)
            probs = probs.permute(0, 2, 3, 1)  # to NHWC

        return probs.cpu().numpy()
