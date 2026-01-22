from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

import numpy as np
import timm
import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from tiatoolbox.models.architecture.utils import (
    Attention,
    SegmentationHead,
    nms_on_detection_maps,
    peak_detection_map_overlap,
)
from tiatoolbox.models.models_abc import ModelABC
from tiatoolbox.type_hints import IntPair


class TimmEncoderFixed(nn.Module):
    """Fixed version of TIMM encoder that handles drop_path_rate parameter properly.

    This encoder wraps TIMM models to provide consistent feature extraction interface
    for segmentation tasks. It extracts features at multiple scales from the encoder
    backbone.

    Args:
        name (str): Name of the TIMM model to use as backbone
        pretrained (bool): Whether to use pretrained weights. Default: True
        in_channels (int): Number of input channels. Default: 3
        depth (int): Number of encoder stages to extract features from. Default: 5
        output_stride (int): Output stride of the encoder. Default: 32
        drop_rate (float): Dropout rate. Default: 0.5
        drop_path_rate (Optional[float]): Drop path rate for stochastic depth. Default: 0.0
    """

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 5,
        output_stride: int = 32,
        drop_rate: float = 0.5,
        drop_path_rate: float | None = 0.0,
    ) -> None:
        super().__init__()
        if drop_path_rate is None:
            kwargs = dict(
                in_chans=in_channels,
                features_only=True,
                pretrained=pretrained,
                out_indices=tuple(range(depth)),
                drop_rate=drop_rate,
            )
        else:
            kwargs = dict(
                in_chans=in_channels,
                features_only=True,
                pretrained=pretrained,
                out_indices=tuple(range(depth)),
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )

        self.model = timm.create_model(name, **kwargs)

        self._in_channels = in_channels
        self._out_channels = [
            in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            List[torch.Tensor]: List of feature tensors at different scales,
                including the input as the first element
        """
        features = self.model(x)
        features = [
            x,
        ] + features
        return features

    @property
    def out_channels(self) -> list[int]:
        """Get output channels for each feature level.

        Returns:
            List[int]: Number of channels at each feature level
        """
        return self._out_channels

    @property
    def output_stride(self) -> int:
        """Get the output stride of the encoder.

        Returns:
            int: Output stride value
        """
        return min(self._output_stride, 2**self._depth)


class SubPixelUpsample(nn.Module):
    """Sub-pixel upsampling module using PixelShuffle.

    This module performs upsampling using sub-pixel convolution (PixelShuffle)
    which is more efficient than transposed convolution and produces better results.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        upscale_factor (int): Factor to increase spatial resolution. Default: 2
    """

    def __init__(
        self, in_channels: int, out_channels: int, upscale_factor: int = 2
    ) -> None:
        super(SubPixelUpsample, self).__init__()
        self.conv1 = Conv2dNormActivation(
            in_channels,
            out_channels * upscale_factor**2,
            kernel_size=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv2 = Conv2dNormActivation(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sub-pixel upsampling.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Upsampled tensor of shape (B, out_channels, H*upscale_factor, W*upscale_factor)
        """
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.conv2(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, skip connection, and attention.

    This block performs upsampling of the input features, concatenates with skip connections
    from the encoder, applies attention mechanisms, and processes through convolutions.

    Args:
        in_channels (int): Number of input channels
        skip_channels (int): Number of channels from skip connection
        out_channels (int): Number of output channels
        attention_type (str): Type of attention mechanism. Default: 'scse'
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        attention_type: str = "scse",
    ) -> None:
        super().__init__()
        self.up = SubPixelUpsample(in_channels, in_channels, upscale_factor=2)
        self.conv1 = Conv2dNormActivation(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )
        self.attention1 = Attention(
            name=attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = Conv2dNormActivation(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )
        self.attention2 = Attention(name=attention_type, in_channels=out_channels)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through decoder block.

        Args:
            x (torch.Tensor): Input tensor to be upsampled
            skip (Optional[torch.Tensor]): Skip connection tensor from encoder. Default: None

        Returns:
            torch.Tensor: Processed output tensor
        """
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Module):
    """Center block that applies attention mechanism at the bottleneck.

    This block is placed at the center of the U-Net architecture (deepest level)
    to enhance feature representation using attention mechanisms.

    Args:
        in_channels (int): Number of input channels
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.attention = Attention(name="scse", in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through center block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor with attention applied
        """
        x = self.attention(x)
        return x


class KongNetDecoder(nn.Module):
    """Decoder module for KongNet architecture.

    This decoder implements a U-Net style decoder with multiple decoder blocks,
    attention mechanisms, and optional center block at the bottleneck.

    Args:
        encoder_channels (List[int]): Number of channels at each encoder level
        decoder_channels (Tuple[int, ...]): Number of channels at each decoder level
        n_blocks (int): Number of decoder blocks. Default: 5
        attention_type (str): Type of attention mechanism. Default: 'scse'
        center (bool): Whether to use center block at bottleneck. Default: True

    Raises:
        ValueError: If n_blocks doesn't match length of decoder_channels
    """

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: tuple[int, ...],
        n_blocks: int = 5,
        attention_type: str = "scse",
        center: bool = True,
    ) -> None:
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but you provide `decoder_channels` for {len(decoder_channels)} blocks."
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            # Bug fix: CenterBlock only takes in_channels parameter
            self.center = CenterBlock(head_channels)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        # Bug fix: DecoderBlock doesn't use use_batchnorm parameter
        kwargs = dict(attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            *features: Variable number of feature tensors from encoder at different scales

        Returns:
            torch.Tensor: Decoded output tensor
        """
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class KongNet(ModelABC):
    """KongNet: Multi-head segmentation model.

    KongNet is a segmentation model with multiple decoder heads that can
    produce different types of segmentation outputs simultaneously. It uses
    a shared encoder and multiple task-specific decoders.

    Args:
        encoder: Encoder module (e.g., TimmEncoderFixed)
        decoder_list (List[nn.Module]): List of decoder modules
        head_list (List[nn.Module]): List of segmentation heads

    Raises:
        ValueError: If decoder_list and head_list have different lengths
    """

    def __init__(
        self: KongNet,
        num_heads: int,
        num_channels_per_head: list[int],
        target_channels: list[int],
        min_distance: int,
        threshold_abs: float,
        wide_decoder: bool = False,
        class_dict: dict | None = None,
        postproc_tile_shape: IntPair = (2048, 2048),
    ) -> None:
        super(KongNet, self).__init__()

        # Bug fix: Add validation for matching decoder and head lists
        if len(num_channels_per_head) != num_heads:
            raise ValueError(
                f"Number of decoders ({len(num_channels_per_head)}) must match "
                f"number of heads ({num_heads})"
            )

        self.encoder = TimmEncoderFixed(
            name="tf_efficientnetv2_l.in21k_ft_in1k",
            in_channels=3,
            depth=5,
            output_stride=32,
            drop_rate=0.5,
            drop_path_rate=0.25,
            pretrained=False,
        )

        decoder_channels = (256, 128, 64, 32, 16)
        if wide_decoder:
            decoder_channels = (512, 256, 128, 64, 32)

        decoders = []
        for i in range(num_heads):
            decoders.append(
                KongNetDecoder(
                    encoder_channels=self.encoder.out_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=len(decoder_channels),
                    center=True,
                    attention_type="scse",
                )
            )

        heads = []
        for i in range(num_heads):
            heads.append(
                SegmentationHead(
                    in_channels=decoders[i].blocks[-1].conv2[0].out_channels,
                    out_channels=num_channels_per_head[i],  # instance channels
                    activation=None,
                    kernel_size=1,
                )
            )

        self.decoders = nn.ModuleList(decoders)
        self.heads = nn.ModuleList(heads)
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs
        self.target_channels = target_channels
        self.output_class_dict = class_dict
        self.postproc_tile_shape = postproc_tile_shape

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
            >>> processed = KongNet.preproc(img)
            >>> processed.shape
            ... (256, 256, 3)

        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (image / 255.0 - mean) / std

    def forward(  # skipcq: PYL-W0613
        self: KongNet,
        x: torch.Tensor,
        *args: tuple[Any, ...],  # noqa: ARG002
        **kwargs: dict,  # noqa: ARG002
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Concatenated output from all heads of shape (B, sum(num_channels_per_head), H, W)
        """
        features = self.encoder(x)
        decoder_outputs = []
        for decoder in self.decoders:
            decoder_outputs.append(decoder(*features))

        segmentation_head_outputs = []
        for head, decoder_output in zip(self.heads, decoder_outputs):
            segmentation_head_outputs.append(head(decoder_output))

        output_all_channels = torch.cat(segmentation_head_outputs, 1)
        return output_all_channels

    @staticmethod
    def infer_batch(
        model: KongNet,
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
            >>> probs = KongNet.infer_batch(model, batch, device="cpu")
            >>> probs.shape
            (4, 256, 256, sum(model.target_channels))

        """
        model = model.to(device)
        model.eval()

        imgs = batch_data
        imgs = imgs.to(device).type(torch.float32)
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        with torch.inference_mode():
            logits = model(imgs)
            target_logits = logits[:, model.target_channels, :, :]
            probs = torch.nn.functional.sigmoid(target_logits)
            probs = probs.permute(0, 2, 3, 1)  # to NHWC

        return probs.cpu().numpy()

    #  skipcq: PYL-W0221  # noqa: ERA001
    def postproc(
        self: KongNet,
        block: np.ndarray,
        min_distance: int | None = None,
        threshold_abs: float | None = None,
        threshold_rel: float | None = None,
        block_info: dict | None = None,
        depth_h: int = 0,
        depth_w: int = 0,
    ) -> np.ndarray:
        """MapDe post-processing function.

        Builds a processed mask per input channel, runs peak_local_max then
        writes 1.0 at peak pixels.

        Returns same spatial shape as the input block

        Args:
            block (np.ndarray):
                shape (H, W, C).
            min_distance (int | None):
                The minimal allowed distance separating peaks.
            threshold_abs (float | None):
                Minimum intensity of peaks.
            threshold_rel (float | None):
                Minimum intensity of peaks.
            block_info (dict | None):
                Dask block info dict. Only used when called from
                dask.array.map_overlap.
            depth_h (int):
                Halo size in pixels for height (rows). Only used
                when it's called from dask.array.map_overlap.
            depth_w (int):
                Halo size in pixels for width (cols). Only used
                when it's called from dask.array.map_overlap.

        Returns:
            out: NumPy array (H, W, C) with 1.0 at peaks, 0 elsewhere.
        """
        min_distance_to_use = (
            self.min_distance if min_distance is None else min_distance
        )
        threshold_abs_to_use = (
            self.threshold_abs if threshold_abs is None else threshold_abs
        )
        peak_map = peak_detection_map_overlap(
            block,
            min_distance=min_distance_to_use,
            threshold_abs=threshold_abs_to_use,
            threshold_rel=threshold_rel,
            block_info=block_info,
            depth_h=depth_h,
            depth_w=depth_w,
            return_probability=True,
        )

        return nms_on_detection_maps(
            peak_map,
            min_distance=min_distance_to_use,
        )

    def load_state_dict(
        self: KongNet,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        return super().load_state_dict(state_dict["model"], strict, assign)
