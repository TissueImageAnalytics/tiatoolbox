"""Define Unet++ architecture from Segmentation Models Pytorch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from tiatoolbox.models.architecture.timm_efficientnet import EfficientNetEncoder
from tiatoolbox.models.models_abc import ModelABC


class SegmentationHead(nn.Sequential):
    """Segmentation head for UNet++ model."""

    def __init__(
        self: SegmentationHead,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: nn.Module | None = None,
        upsampling: int = 1,
    ) -> None:
        """Initialize SegmentationHead.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size. Defaults to 3.
            activation: Activation function. Defaults to None.
            upsampling: Upsampling factor. Defaults to 1.
        """
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        if activation is None:
            activation = nn.Identity()
        super().__init__(conv2d, upsampling, activation)


class Conv2dReLU(nn.Sequential):
    """Conv2d + BatchNorm + ReLU block."""

    def __init__(
        self: Conv2dReLU,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        """Initialize Conv2dReLU block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            padding: Padding size. Defaults to 0.
            stride: Stride size. Defaults to 1.
        """
        norm = nn.BatchNorm2d(out_channels)

        is_identity = isinstance(norm, nn.Identity)
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=is_identity,
        )

        activation = nn.ReLU(inplace=True)

        super().__init__(conv, norm, activation)


class DecoderBlock(nn.Module):
    """Decoder block for UNet++ architecture."""

    def __init__(
        self: DecoderBlock,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        """Initialize DecoderBlock.

        Args:
            in_channels: Number of input channels.
            skip_channels: Number of skip connection channels.
            out_channels: Number of output channels.
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
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through decoder block.

        Args:
            x: Input tensor.
            skip: Skip connection tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after decoding.
        """
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.attention2(x)


class CenterBlock(nn.Sequential):
    """Center block for UNet++ architecture."""

    def __init__(
        self: CenterBlock,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """Initialize CenterBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
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
    """UNet++ decoder with dense connections."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        n_blocks: int = 5,
    ) -> None:
        """Initialize UnetPlusPlusDecoder.

        Args:
            encoder_channels: List of encoder output channels.
            decoder_channels: List of decoder output channels.
            n_blocks: Number of decoder blocks. Defaults to 5.

        Raises:
            ValueError: If model depth doesn't match decoder_channels length.
        """
        super().__init__()

        if n_blocks != len(decoder_channels):
            msg = f"Model depth is {n_blocks}, but you provide  \
            `decoder_channels` for {len(decoder_channels)} blocks."
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

        Args:
            features: List of encoder feature maps.

        Returns:
            torch.Tensor: Decoded output tensor.
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


class UNetPlusPlusModel(ModelABC):
    """UNet++ Model."""

    def __init__(
        self: UNetPlusPlusModel,
        encoder_depth: int = 5,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        classes: int = 1,
    ) -> None:
        """Initialize UNet++ model.

        Args:
            encoder_depth: Depth of the encoder. Defaults to 5.
            decoder_channels: Number of channels in decoder layers.
                Defaults to (256, 128, 64, 32, 16).
            classes: Number of output classes. Defaults to 1.
        """
        super().__init__()

        self.encoder = EfficientNetEncoder(
            out_channels=[3, 32, 24, 40, 112, 320],
            stage_idxs=[2, 3, 5],
            channel_multiplier=1.0,
            depth_multiplier=1.0,
            drop_rate=0.2,
        )
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,
        )

        self.name = "unetplusplus-efficientnetb0"

    def forward(
        self: UNetPlusPlusModel,
        x: torch.Tensor,
        *args: tuple[Any, ...],  # skipcq: PYL-W0613  # noqa: ARG002
        **kwargs: dict,  # skipcq: PYL-W0613  # noqa: ARG002
    ) -> torch.Tensor:
        """Sequentially pass `x` through model's encoder, decoder and heads.

        Args:
            x: Input tensor.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Segmentation output.
        """
        features = self.encoder(x)
        decoder_output = self.decoder(features)

        return self.segmentation_head(decoder_output)

    @staticmethod
    def infer_batch(
        model: torch.nn.Module,
        batch_data: torch.Tensor | np.ndarray,
        *,
        device: str,
    ) -> np.ndarray:
        """Run inference on an input batch.

        This contains logic for forward operation as well as i/o

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (:class:`torch.Tensor`):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            device (str):
                Transfers model to the specified device. Default is "cpu".

        Returns:
            np.ndarray:
                The inference results as a numpy array.

        """
        model.eval()

        imgs = batch_data
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        imgs = imgs.to(device).type(torch.float32)
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        with torch.inference_mode():
            logits = model(imgs)
            probs = torch.nn.functional.softmax(logits, 1)
            probs = probs.permute(0, 2, 3, 1)  # to NHWC

        return probs.cpu().numpy()
