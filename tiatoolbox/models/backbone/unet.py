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
# The Original Code is Copyright (C) 2021, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Defines a set of ResNet variants to be used within tiatoolbox."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet

from tiatoolbox.models.abc import ModelABC
from tiatoolbox.models.backbone.utils import UpSample2x, crop_op


class ResNetEncoder(ResNet):
    """A subclass of ResNet defined in torch.

    This class overwrites the `forward` implementation within pytorch
    to return features of each downsampling level. This is necessary
    for segmentation.

    """

    def _forward_impl(self, x):
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
            num_input_channels (int): Number of channels in input images.
            downsampling_levels (list): A list of integer where each number defines
              the number of BottleNeck blocks and denotes a down-sampling levels.
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
    The number of channels in the per down-sampling block as well as
    the number of down-sampling levels are customisable.

    Args:
        num_input_channels (int): Number of channels in input images.
        layer_output_channels (list): A list of integer where each number
          defines the number of output channels of a down-sampling level.
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
                nn.Sequential(
                    nn.Conv2d(
                        input_channels, output_channels, 3, 1, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                    nn.Conv2d(
                        output_channels, output_channels, 3, 1, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2, stride=2),
                )
            )
            input_channels = output_channels

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class UNetModel(ModelABC):
    """Generate families of UNet model.

    This supports different encoders. However, the decoder is relatively simple,
    each upsampling block contains a single 3x3 Convolution Layer and is
    not customizable. Additionally, the aggregation between down-sampling and
    up-sampling is addition, not concatenation.

    Args:
        num_input_channels (int): Number of channels in input images.
        num_output_channels (list): Number of channels in output images.
        encoder (str): Name of the encoder, currently support:
            - "resnet50": The well-known ResNet50, this is not pre-activation
              model.
            - "unet": The vanilla UNet encoder where each down-sampling level
              contains 2 blocks of Convolution-BatchNorm-ReLu.
    Returns:
        model (torch.nn.Module): a pytorch model.

    """

    def __init__(
        self,
        num_input_channels: int = 2,
        num_output_channels: int = 2,
        encoder: str = "resnet50",
    ):
        super().__init__()

        if encoder == "resnet50":
            self.backbone = ResNetEncoder.resnet50(num_input_channels)
        elif encoder == "unet":
            self.backbone = UnetEncoder(num_input_channels, [64, 128, 256, 512, 2048])
        else:
            raise ValueError(f"Unknown encoder `{encoder}`")

        img_list = torch.rand([1, num_input_channels, 256, 256])
        out_list = self.backbone(img_list)
        # ordered from low to high resolution
        down_ch_list = [v.shape[1] for v in out_list][::-1]

        # channel mapping for shortcut
        self.conv1x1 = nn.Conv2d(down_ch_list[0], down_ch_list[1], (1, 1), bias=False)

        self.uplist = nn.ModuleList()
        for ch_idx, ch in enumerate(down_ch_list[1:]):
            next_up_ch = ch
            if ch_idx + 2 < len(down_ch_list):
                next_up_ch = down_ch_list[ch_idx + 2]
            self.uplist.append(
                nn.Sequential(
                    nn.BatchNorm2d(ch),
                    nn.ReLU(),
                    nn.Conv2d(ch, next_up_ch, (3, 3), padding=1, bias=False),
                )
            )

        self.clf = nn.Conv2d(next_up_ch, num_output_channels, (1, 1), bias=True)
        self.upsample2x = UpSample2x()

    def forward(self, img_list, *args, **kwargs):
        # scale to 0-1
        img_list = img_list / 255.0

        # assume output is after each down-sample resolution
        en_list = self.backbone(img_list)
        x = self.conv1x1(en_list[-1])

        en_list = en_list[:-1]
        for idx in range(1, len(en_list) + 1):
            y = en_list[-idx]
            x = self.upsample2x(x) + y
            x = self.uplist[idx - 1](x)
        output = self.clf(x)
        return output

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):

        ####
        model.eval()
        device = "cuda" if on_gpu else "cpu"

        ####
        imgs = batch_data

        imgs = imgs.to(device).type(torch.float32)
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        _, _, h, w = imgs.shape
        crop_shape = [h // 2, w // 2]
        with torch.no_grad():
            logits = model(imgs)
            probs = F.softmax(logits, 1)
            probs = F.interpolate(
                probs, scale_factor=2, mode="bilinear", align_corners=False
            )
            probs = crop_op(probs, crop_shape)
            probs = probs.permute(0, 2, 3, 1)  # to NHWC

        probs = probs.cpu().numpy()
        return [probs]
