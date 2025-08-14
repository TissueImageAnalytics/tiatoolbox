"""Define a set of UNet variants to be used within tiatoolbox."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet

from tiatoolbox.models.architecture.utils import UpSample2x, centre_crop
from tiatoolbox.models.models_abc import ModelABC


class ResNetEncoder(ResNet):
    """A subclass of ResNet defined in torch.

    This class overwrites the `forward` implementation within pytorch
    to return features of each downsampling level. This is necessary
    for segmentation.

    """

    def _forward_impl(self: ResNetEncoder, x: torch.Tensor) -> list:
        """Overwriting default torch forward so that it returns features.

        Args:
            x (:class:`torch.Tensor`): Input images, the tensor is in the shape of NCHW.
              For this method, C=3 (i.e 3 channels images are used as input).

        Returns:
            list:
                List of features for each down-sample block. Each
                feature tensor is of the shape NCHW.

        """
        # See note [TorchScript super()]
        x0 = x = self.conv1(x)
        x0 = x = self.bn1(x)
        x0 = x = self.relu(x)
        x1 = x = self.maxpool(x)
        x1 = x = self.layer1(x)
        x2 = x = self.layer2(x)
        x3 = x = self.layer3(x)
        x4 = x = self.layer4(x)
        return [x0, x1, x2, x3, x4]

    @staticmethod
    def resnet50(num_input_channels: int) -> torch.nn.Module:
        """Shortcut method to create ResNet50."""
        return ResNetEncoder.resnet(num_input_channels, [3, 4, 6, 3])

    @staticmethod
    def resnet(
        num_input_channels: int,
        downsampling_levels: list[int],
    ) -> torch.nn.Module:
        """Shortcut method to create customised ResNet.

        Args:
            num_input_channels (int):
                Number of channels in the input images.
            downsampling_levels (list):
                A list of integers where each number defines the number
                of BottleNeck blocks at each down-sampling level.

        Returns:
            model (torch.nn.Module):
                A pytorch model.

        Examples:
            >>> # instantiate a resnet50
            >>> ResNetEncoder.resnet50(
            ...     num_input_channels,
            ...     [3, 4, 6, 3],
            ... )

        """
        model = ResNetEncoder(ResNetBottleneck, downsampling_levels)
        if num_input_channels != 3:  # noqa: PLR2004
            model.conv1 = nn.Conv2d(  # skipcq: PYL-W0201
                num_input_channels,
                64,
                7,
                stride=2,
                padding=3,
            )
        return model


class UnetEncoder(nn.Module):
    """Construct a basic UNet encoder.

    This class builds a basic UNet encoder with batch normalization.
    The number of channels in each down-sampling block and
    the number of down-sampling levels are customisable.

    Args:
        num_input_channels (int):
            Number of channels in the input images.
        layer_output_channels (list):
            A list of integers where each number defines the number of
            output channels at each down-sampling level.

    Returns:
        model (torch.nn.Module):
            A pytorch model.

    """

    def __init__(
        self: UnetEncoder,
        num_input_channels: int,
        layer_output_channels: list[int],
    ) -> None:
        """Initialize :class:`UnetEncoder`."""
        super().__init__()

        self.blocks = nn.ModuleList()
        input_channels = num_input_channels
        for output_channels in layer_output_channels:
            self.blocks.append(
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(
                                input_channels,
                                output_channels,
                                3,
                                1,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(output_channels),
                            nn.ReLU(),
                            nn.Conv2d(
                                output_channels,
                                output_channels,
                                3,
                                1,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(output_channels),
                            nn.ReLU(),
                        ),
                        nn.AvgPool2d(2, stride=2),
                    ],
                ),
            )
            input_channels = output_channels

    def forward(self: UnetEncoder, input_tensor: torch.Tensor) -> list:
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            input_tensor (:class:`torch.Tensor`):
                Input images, the tensor is in the shape of NCHW.

        Returns:
            list:
                A list of features for each down-sample block. Each
                feature tensor is of the shape NCHW.

        """
        features = []
        for block in self.blocks:
            input_tensor = block[0](input_tensor)
            features.append(input_tensor)
            input_tensor = block[1](input_tensor)  # down-sample
        return features


def create_block(
    kernels: list,
    input_ch: list,
    output_ch: int,
    *,
    pre_activation: bool,
) -> list:
    """Helper to create a block of Vanilla Convolution.

    This is in pre-activation style.

    Args:
        pre_activation (bool):
            Whether to apply activation layer before the convolution layer.
            Should be True for ResNet blocks.
        kernels (list):
            A list of convolution layers. Each item is an
            integer and denotes the layer kernel size.
        input_ch (int):
            Number of channels in the input images.
        output_ch (int):
            Number of channels in the output images.

    """
    layers = []
    for ksize in kernels:
        if pre_activation:
            layers.extend(
                [
                    nn.BatchNorm2d(input_ch),
                    nn.ReLU(),
                    nn.Conv2d(
                        input_ch,
                        output_ch,
                        (ksize, ksize),
                        padding=int((ksize - 1) // 2),  # same padding
                        bias=False,
                    ),
                ],
            )
        else:
            layers.extend(
                [
                    nn.Conv2d(
                        input_ch,
                        output_ch,
                        (ksize, ksize),
                        padding=int((ksize - 1) // 2),  # same padding
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_ch),
                    nn.ReLU(),
                ],
            )
        input_ch = output_ch
    return layers


class UNetModel(ModelABC):
    """Generate families of UNet model.

    This supports different encoders. However, the decoder is relatively
    simple, each upsampling block contains a number of vanilla
    convolution layers, that are not customizable. Additionally, the
    aggregation between down-sampling and up-sampling is addition, not
    concatenation.

    Args:
        num_input_channels (int):
            Number of channels in input images.
        num_output_channels (int):
            Number of channels in output images.
        encoder (str):
            Name of the encoder, currently supports:
                - "resnet50": The well-known ResNet50- this is not the
                  pre-activation model.
                - "unet": The vanilla UNet encoder where each down-sampling
                  level contains 2 blocks of Convolution-BatchNorm-ReLu.
        encoder_levels (list):
            A list of integers to configure "unet" encoder levels.
            Each number defines the number of output channels at each
            down-sampling level (2 convolutions). Number of intergers
            define the number down-sampling levels in the unet encoder.
            This is only applicable when `encoder="unet"`.
        decoder_block (list):
            A list of convolution layers. Each item is an integer and
            denotes the layer kernel size.
        skip_type (str):
            Choosing between "add" or "concat" method to be used for
            combining feature maps from encoder and decoder parts at
            skip connections. Default is "add".

    Returns:
        torch.nn.Module:
            A pytorch model.

    Examples:
        >>> # instantiate a UNet with resnet50 encoder and
        >>> # only 1 3x3 per each up-sampling block in the decoder
        >>> UNetModel.resnet50(
        ...     2, 2,
        ...     encoder="resnet50",
        ...     decoder_block=(3,)
        ... )

    """

    def __init__(
        self: UNetModel,
        num_input_channels: int = 2,
        num_output_channels: int = 2,
        encoder: str = "resnet50",
        encoder_levels: list[int] | None = None,
        decoder_block: tuple[int] | None = None,
        skip_type: str = "add",
    ) -> None:
        """Initialize :class:`UNetModel`."""
        super().__init__()

        if encoder.lower() not in {"resnet50", "unet"}:
            msg = f"Unknown encoder `{encoder}`"
            raise ValueError(msg)

        if encoder_levels is None:
            encoder_levels = [64, 128, 256, 512, 1024]

        if decoder_block is None:
            decoder_block = [3, 3]

        pre_activation = None
        if encoder == "resnet50":
            pre_activation = True
            self.backbone = ResNetEncoder.resnet50(num_input_channels)
        if encoder == "unet":
            pre_activation = False
            self.backbone = UnetEncoder(num_input_channels, encoder_levels)

        if skip_type.lower() not in {"add", "concat"}:
            msg = f"Unknown type of skip connection: `{skip_type}`"
            raise ValueError(msg)
        self.skip_type = skip_type.lower()

        img_list = torch.rand([1, num_input_channels, 256, 256])
        out_list = self.backbone(img_list)
        # ordered from low to high resolution
        down_ch_list = [v.shape[1] for v in out_list][::-1]

        # channel mapping for shortcut
        self.conv1x1 = nn.Conv2d(down_ch_list[0], down_ch_list[1], (1, 1), bias=False)

        self.uplist = nn.ModuleList()
        next_up_ch = None
        for ch_idx, ch in enumerate(down_ch_list[1:]):
            next_up_ch = ch
            if ch_idx + 2 < len(down_ch_list):
                next_up_ch = down_ch_list[ch_idx + 2]
            ch_ = ch
            if self.skip_type == "concat":
                ch_ *= 2
            layers = create_block(
                decoder_block,
                ch_,
                next_up_ch,
                pre_activation=pre_activation,
            )
            self.uplist.append(nn.Sequential(*layers))

        self.clf = nn.Conv2d(next_up_ch, num_output_channels, (1, 1), bias=True)
        self.upsample2x = UpSample2x()

    @staticmethod
    def _transform(image: torch.Tensor) -> torch.Tensor:
        """Transforming network input to desired format.

        This method is model and dataset specific, meaning that it can be replaced by
        user's desired transform function before training/inference.

        Args:
            image (:class:`torch.Tensor`): Input images, the tensor is of the shape
                NCHW.

        Returns:
            output (:class:`torch.Tensor`): The transformed input.

        """
        return image / 255.0

    # pylint: disable=W0221
    # because abc is generic, this is actual definition
    def forward(
        self: UNetModel,
        imgs: torch.Tensor,
        *args: tuple[Any, ...],  # skipcq: PYL-W0613  # noqa: ARG002
        **kwargs: dict,  # skipcq: PYL-W0613  # noqa: ARG002
    ) -> torch.Tensor:
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            imgs (:class:`torch.Tensor`):
                Input images, the tensor is of the shape NCHW.
            args (list):
                List of input arguments. Not used here.
                Provided for consistency with the API.
            kwargs (dict):
                Key-word arguments. Not used here.
                Provided for consistency with the API.

        Returns:
            :class:`torch.Tensor`:
                The inference output. The tensor is of the shape NCHW.
                However, `height` and `width` may not be the same as the
                input images.

        """
        # transform the input using network-specific transform function
        imgs = self._transform(imgs)

        # assume output is after each down-sample resolution
        en_list = self.backbone(imgs)
        x = self.conv1x1(en_list[-1])

        en_list = en_list[:-1]
        for idx in range(1, len(en_list) + 1):
            # up-sample feature from low-resolution
            # block, add it with features from the same resolution
            # coming from the encoder, then run it through the decoder
            # block
            y = en_list[-idx]
            x_ = self.upsample2x(x)
            x = x_ + y if self.skip_type == "add" else torch.cat([x_, y], dim=1)
            x = self.uplist[idx - 1](x)
        return self.clf(x)

    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: torch.Tensor,
        *,
        device: str,
    ) -> list:
        """Run inference on an input batch.

        This contains logic for forward operation as well as i/o
        aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (:class:`torch.Tensor`):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            device (str):
                Transfers model to the specified device. Default is "cpu".

        Returns:
            list:
                List of network output head, each output is an
                :class:`numpy.ndarray`.

        """
        model.eval()

        ####
        imgs = batch_data

        imgs = imgs.to(device).type(torch.float32)
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        _, _, h, w = imgs.shape
        crop_shape = [h // 2, w // 2]
        with torch.inference_mode():
            logits = model(imgs)
            probs = F.softmax(logits, 1)
            probs = F.interpolate(
                probs,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
            probs = centre_crop(probs, crop_shape)
            probs = probs.permute(0, 2, 3, 1)  # to NHWC

        probs = probs.cpu().numpy()
        return [probs]
