# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Defines a set of ResNet variants to be used within tiatoolbox."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet

from tiatoolbox.models.abc import ModelABC
from tiatoolbox.models.architecture.utils import UpSample2x, centre_crop
from tiatoolbox.utils import misc


class ResNetEncoder(ResNet):
    """A subclass of ResNet defined in torch.

    This class overwrites the `forward` implementation within pytorch
    to return features of each downsampling level. This is necessary
    for segmentation.

    """

    def _forward_impl(self, x):
        """Overwriting default torch forward so that it returns features.

        Args:
            x (torch.Tensor): Input images, the tensor is in the shape of NCHW.
              For this method, C=3 (i.e 3 channels images are used as input).

        Returns:
            features (list): List of features for each down-sample
              block. Each feature tensor is of the shape NCHW.

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
    def resnet50(num_input_channels: int):
        """Shortcut method to create ResNet50."""
        model = ResNetEncoder.resnet(num_input_channels, [3, 4, 6, 3])
        return model

    @staticmethod
    def resnet(
        num_input_channels: int,
        downsampling_levels: List[int],
    ):
        """Shortcut method to create customised ResNet.

        Args:
            num_input_channels (int): Number of channels in the input images.
            downsampling_levels (list): A list of integers where each number defines
              the number of BottleNeck blocks at each down-sampling level.

        Returns:
            model (torch.nn.Module): a pytorch model.

        Examples:
            >>> # instantiate a resnet50
            >>> ResNetEncoder.resnet50(
            ...     num_input_channels,
            ...     [3, 4, 6, 3],
            ...     pretrained
            ... )

        """
        model = ResNetEncoder(ResNetBottleneck, downsampling_levels)
        if num_input_channels != 3:
            model.conv1 = nn.Conv2d(num_input_channels, 64, 7, stride=2, padding=3)
        return model


class UnetEncoder(nn.Module):
    """Construct a basic unet encoder.

    This class builds a basic unet encoder with batch normalization.
    The number of channels in each down-sampling block and
    the number of down-sampling levels are customisable.

    Args:
        num_input_channels (int): Number of channels in the input images.
        layer_output_channels (list): A list of integers where each number
          defines the number of output channels at each down-sampling level.

    Returns:
        model (torch.nn.Module): a pytorch model.

    """

    def __init__(
        self,
        num_input_channels: int,
        layer_output_channels: List[int],
    ):
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
                    ]
                )
            )
            input_channels = output_channels

    def forward(self, x: torch.Tensor):
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            x (torch.Tensor): Input images- the tensor is of the shape NCHW.

        Returns:
            features (list): List of features for each down-sample
              block. Each feature tensor is of the shape NCHW.

        """
        features = []
        for block in self.blocks:
            x = block[0](x)
            features.append(x)
            x = block[1](x)  # down-sample
        return features


class UNetModel(ModelABC):
    """Generate families of UNet model.

    This supports different encoders. However, the decoder is relatively simple-
    each upsampling block contains a number of vanilla convolution layers, that
    are not customizable. Additionally, the aggregation between down-sampling and
    up-sampling is addition, not concatenation.

    Args:
        num_input_channels (int): Number of channels in input images.
        num_output_channels (int): Number of channels in output images.
        encoder (str): Name of the encoder, currently supports:
          - "resnet50": The well-known ResNet50- this is not the pre-activation model.
          - "unet": The vanilla UNet encoder where each down-sampling level
            contains 2 blocks of Convolution-BatchNorm-ReLu.
        decoder_block (list): A list of convolution layers. Each item is an
          integer and denotes the layer kernel size.
        classifier (list): A list of convolution layers before the final 1x1
          convolution. Each item is an integer denotes the layer kernel size. The
          default is `None` and contains only the 1x1 convolution.

    Returns:
        model (torch.nn.Module): a pytorch model.

    Examples:
        >>> # instantiate a UNet with resnet50 endcoder and
        >>> # only 1 3x3 per each up-sampling block in the decoder
        >>> UNetModel.resnet50(
        ...     2, 2
        ...     encoder="resnet50",
        ...     decoder_block=(3,)
        ... )

    """

    def __init__(
        self,
        num_input_channels: int = 2,
        num_output_channels: int = 2,
        encoder: str = "resnet50",
        decoder_block: Tuple[int] = (3, 3),
    ):
        super().__init__()

        if encoder == "resnet50":
            padding = 1
            preact = True
            self.backbone = ResNetEncoder.resnet50(num_input_channels)
        elif encoder == "unet":
            padding = 0
            preact = False
            self.backbone = UnetEncoder(num_input_channels, [64, 128, 256, 512, 2048])
        else:
            raise ValueError(f"Unknown encoder `{encoder}`")

        img_list = torch.rand([1, num_input_channels, 256, 256])
        out_list = self.backbone(img_list)
        # ordered from low to high resolution
        down_ch_list = [v.shape[1] for v in out_list][::-1]

        # channel mapping for shortcut
        self.conv1x1 = nn.Conv2d(down_ch_list[0], down_ch_list[1], (1, 1), bias=False)

        def create_block(kernels, input_ch, output_ch):
            """Helper to create a block of Vanilla Convolution.

            This is in pre-activation style.

            Args:
                kernels (list): A list of convolution layers. Each item is an
                  integer and denotes the layer kernel size.
                input_ch (int): Number of channels in the input images.
                output_ch (int): Number of channels in the output images.

            """
            layers = []
            for ksize in kernels:
                if preact:
                    layers.extend(
                        [
                            nn.BatchNorm2d(input_ch),
                            nn.ReLU(),
                            nn.Conv2d(
                                input_ch,
                                output_ch,
                                (ksize, ksize),
                                padding=padding,
                                bias=False,
                            ),
                        ]
                    )
                else:
                    layers.extend(
                        [
                            nn.Conv2d(
                                input_ch,
                                output_ch,
                                (ksize, ksize),
                                padding=padding,
                                bias=False,
                            ),
                            nn.BatchNorm2d(input_ch),
                            nn.ReLU(),
                        ]
                    )
                input_ch = output_ch
            return layers

        self.uplist = nn.ModuleList()
        for ch_idx, ch in enumerate(down_ch_list[1:]):
            next_up_ch = ch
            if ch_idx + 2 < len(down_ch_list):
                next_up_ch = down_ch_list[ch_idx + 2]
            layers = create_block(decoder_block, ch, next_up_ch)
            self.uplist.append(nn.Sequential(*layers))

        self.clf = nn.Conv2d(next_up_ch, num_output_channels, (1, 1), bias=True)
        self.upsample2x = UpSample2x()

    # pylint: disable=W0221
    # because abc is generic, this is actual definition
    def forward(self, imgs: torch.Tensor, *args, **kwargs):
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            imgs (torch.Tensor): Input images, the tensor is of the shape NCHW.

        Returns:
            output (torch.Tensor): The inference output. The tensor is of the shape
              NCHW. However, `height` and `width` may not be the same as the
              input images.

        """

        # scale to 0-1
        imgs = imgs / 255.0

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
            x = self.upsample2x(x) + y
            x = self.uplist[idx - 1](x)
        output = self.clf(x)
        return output

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        """Run inference on an input batch.

        This contains logic for forward operation as well as
        i/o aggregation.

        Args:
            model (nn.Module): PyTorch defined model.
            batch_data (ndarray): A batch of data generated by
              torch.utils.data.DataLoader.
            on_gpu (bool): Whether to run inference on a GPU.

        Returns:
            List of network output head, each output is a `ndarray`.

        """
        model.eval()
        device = misc.select_device(on_gpu)

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
                probs, scale_factor=2, mode="bilinear", align_corners=False
            )
            probs = centre_crop(probs, crop_shape)
            probs = probs.permute(0, 2, 3, 1)  # to NHWC

        probs = probs.cpu().numpy()
        return [probs]
