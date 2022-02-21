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
"""Defines MicroNet architecture.

Raza, SEA et al., “Micro-Net: A unified model for segmentation of
various objects in microscopy images,” Medical Image Analysis,
Dec. 2018, vol. 52, p. 160–173.

"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as functional


def weights_init(m):
    """Initializes weights and biases for a torch Module e.g., Conv2d.

    Args:
        m (torch.nn.Module): :class:`torch.nn.Module` with
            weights and biases to initialize.

    """
    classname = m.__class__.__name__
    # ! Fixed the type checking
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.1)

    if "norm" in classname.lower():
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    if "linear" in classname.lower() and m.bias is not None:
        nn.init.constant_(m.bias, 0)


def resize_op(in_tensor, size):
    """Resize input tensor.

    Args:
        in_tensor (torch.Tensor): input tensor of images.
        size (tuple of int): output spatial size

    Returns:
        torch.Tensor: Down/up sampled tensor containing images.

    """
    return functional.interpolate(in_tensor, size=size, mode="bicubic")


class MicroNet(nn.Module):
    """Initialise MicroNet.

    Args:
        num_input_channels (int): Number of channels in input. default=3.
        num_class (int): Number of output channels. default=2.

    References:
        Raza, SEA et al., “Micro-Net: A unified model for segmentation of
        various objects in microscopy images,” Medical Image Analysis,
        Dec. 2018, vol. 52, p. 160–173.

    """

    def __init__(self, num_input_channels=3, num_class=2):
        super().__init__()
        if num_class < 2:
            ValueError("Number of classes should be >=2.")
        self.__num_class = num_class
        self.in_ch = num_input_channels

        def group1_branch(in_ch: int, resized_in_ch: int, out_ch: int):
            """MicroNet group1 branch.

            Args:
                in_ch (int): Number of input channels.
                resized_in_ch (int): Number of input channels from
                    resized input.
                out_ch (int): Number of output channels.

            Returns:
                torch.nn.ModuleDict: An output of type :class:`torch.nn.ModuleDict`

            """
            module_dict = OrderedDict()
            module_dict["conv1"] = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.Tanh(),
                nn.BatchNorm2d(out_ch),
            )
            module_dict["conv2"] = nn.Sequential(
                nn.Conv2d(
                    out_ch,
                    out_ch,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.Tanh(),
            )
            module_dict["pool"] = nn.MaxPool2d(2, padding=0)  # check padding

            module_dict["conv3"] = nn.Sequential(
                nn.Conv2d(
                    resized_in_ch,
                    out_ch,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.Tanh(),
                nn.BatchNorm2d(out_ch),
            )
            module_dict["conv4"] = nn.Sequential(
                nn.Conv2d(
                    out_ch,
                    out_ch,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.Tanh(),
            )
            return nn.ModuleDict(module_dict)

        def group2_branch(in_ch, out_ch):
            """MicroNet group2 branch.

            Args:
                in_ch (int): Number of input channels.
                out_ch (int): Number of output channels.

            Returns:
                torch.nn.ModuleDict: An output of type :class:`torch.nn.ModuleDict`

            """
            module_dict = OrderedDict()
            module_dict["conv1"] = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.Tanh(),
            )
            module_dict["conv2"] = nn.Sequential(
                nn.Conv2d(
                    out_ch,
                    out_ch,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.Tanh(),
            )
            return nn.ModuleDict(module_dict)

        def group3_branch(in_ch, skip, out_ch):
            """MicroNet group3 branch.

            Args:
                in_ch (int): Number of input channels.
                skip (int): Number of channels for the skip connection.
                out_ch (int): Number of output channels.

            Returns:
                torch.nn.ModuleDict: An output of type :class:`torch.nn.ModuleDict`

            """
            module_dict = OrderedDict()
            module_dict["up1"] = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=(2, 2), stride=(2, 2)
            )
            module_dict["conv1"] = nn.Sequential(
                nn.Conv2d(
                    out_ch,
                    out_ch,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.Tanh(),
            )
            module_dict["conv2"] = nn.Sequential(
                nn.Conv2d(
                    out_ch,
                    out_ch,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.Tanh(),
            )
            module_dict["up2"] = nn.ConvTranspose2d(
                out_ch, out_ch, kernel_size=(5, 5), stride=(1, 1)
            )

            module_dict["up3"] = nn.ConvTranspose2d(
                skip, out_ch, kernel_size=(5, 5), stride=(1, 1)
            )

            module_dict["conv3"] = nn.Sequential(
                nn.Conv2d(
                    2 * out_ch,
                    out_ch,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.Tanh(),
            )
            return nn.ModuleDict(module_dict)

        def group4_branch(in_ch, out_ch, up_kernel=(2, 2), up_strides=(2, 2)):
            """MicroNet group4 branch.

            Args:
                in_ch (int): Number of input channels.
                out_ch (int): Number of output channels.
                up_kernel (tuple of int): Kernel size for
                    :class:`torch.nn.ConvTranspose2d`.
                up_strides (tuple of int): Stride size for
                    :class:`torch.nn.ConvTranspose2d`.

            Returns:
                torch.nn.ModuleDict: An output of type :class:`torch.nn.ModuleDict`

            """
            module_dict = OrderedDict()
            module_dict["up1"] = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=up_kernel, stride=up_strides
            )
            module_dict["conv1"] = nn.Sequential(
                nn.Conv2d(
                    out_ch,
                    out_ch,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.Tanh(),
            )
            return nn.ModuleDict(module_dict)

        def out_branch(in_ch):
            """MicroNet group2 branch.

            Args:
                in_ch (int): Number of input channels.

            Returns:
                torch.nn.Sequential: An output of type :class:`torch.nn.Sequential`

            """
            return nn.Sequential(
                nn.Dropout2d(p=0.5),
                nn.Conv2d(
                    in_ch,
                    self.__num_class,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=0,
                    bias=True,
                ),
                nn.Softmax(),
            )

        module_dict = OrderedDict()
        module_dict["b1"] = group1_branch(num_input_channels, num_input_channels, 64)
        module_dict["b2"] = group1_branch(128, num_input_channels, 128)
        module_dict["b3"] = group1_branch(256, num_input_channels, 256)
        module_dict["b4"] = group1_branch(512, num_input_channels, 512)

        module_dict["b5"] = group2_branch(1024, 2048)

        module_dict["b6"] = group3_branch(2048, 1024, 1024)
        module_dict["b7"] = group3_branch(1024, 512, 512)
        module_dict["b8"] = group3_branch(512, 256, 256)
        module_dict["b9"] = group3_branch(256, 128, 128)

        module_dict["fm1"] = group4_branch(128, 64, (2, 2), (2, 2))
        module_dict["fm2"] = group4_branch(256, 128, (4, 4), (4, 4))
        module_dict["fm3"] = group4_branch(512, 256, (8, 8), (8, 8))

        module_dict["aux_out1"] = out_branch(64)
        module_dict["aux_out2"] = out_branch(128)
        module_dict["aux_out3"] = out_branch(256)

        module_dict["out"] = out_branch(64 + 128 + 256)

        self.layer = nn.ModuleDict(module_dict)

        self.apply(weights_init)

    def forward(self, inputs: torch.Tensor):
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            inputs (torch.Tensor): Input images, the tensor is in the shape of NCHW.

        Returns:
            list: A list of main and auxiliary outputs.
                The expected format is [main_output, aux1, aux2, aux3].

        """

        def group1_branch(layer, in_tensor, resized_feat):
            """Defines group 1 connections.

            Args:
                layer (torch.nn.Module): Network layer.
                in_tensor (torch.Tensor): Input tensor.
                resized_feat (torch.Tensor): Resized input.

            Returns:
                torch.Tensor: Output of group 1 layer.

            """
            a = layer["conv1"](in_tensor)
            a = layer["conv2"](a)
            a = layer["pool"](a)
            b = layer["conv3"](resized_feat)
            b = layer["conv4"](b)
            return torch.cat(tensors=(a, b), dim=1)

        def group2_branch(layer, in_tensor):
            """Defines group 1 connections.

            Args:
                layer (torch.nn.Module): Network layer.
                in_tensor (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output of group 1 layer.

            """
            a = layer["conv1"](in_tensor)
            return layer["conv2"](a)

        def group3_branch(layer, main_feat, skip):
            """Defines group 1 connections.

            Args:
                layer (torch.nn.Module): Network layer.
                main_feat (torch.Tensor): Input tensor.
                skip (torch.Tensor): Skip connection.

            Returns:
                torch.Tensor: Output of group 1 layer.

            """
            a = layer["up1"](main_feat)
            a = layer["conv1"](a)
            a = layer["conv2"](a)

            b1 = layer["up2"](a)
            b2 = layer["up3"](skip)
            b = torch.cat(tensors=(b1, b2), dim=1)
            return layer["conv3"](b)

        def group4_branch(layer, in_tensor):
            """Defines group 1 connections.

            Args:
                layer (torch.nn.Module): Network layer.
                in_tensor (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output of group 1 layer.

            """
            a = layer["up1"](in_tensor)
            return layer["conv1"](a)

        b1 = group1_branch(
            self.layer["b1"],
            inputs,
            functional.interpolate(inputs, size=(128, 128), mode="bicubic"),
        )
        b2 = group1_branch(
            self.layer["b2"],
            b1,
            functional.interpolate(inputs, size=(64, 64), mode="bicubic"),
        )
        b3 = group1_branch(
            self.layer["b3"],
            b2,
            functional.interpolate(inputs, size=(32, 32), mode="bicubic"),
        )
        b4 = group1_branch(
            self.layer["b4"],
            b3,
            functional.interpolate(inputs, size=(16, 16), mode="bicubic"),
        )
        b5 = group2_branch(self.layer["b5"], b4)
        b6 = group3_branch(self.layer["b6"], b5, b4)
        b7 = group3_branch(self.layer["b7"], b6, b3)
        b8 = group3_branch(self.layer["b8"], b7, b2)
        b9 = group3_branch(self.layer["b9"], b8, b1)
        fm1 = group4_branch(self.layer["fm1"], b9)
        fm2 = group4_branch(self.layer["fm2"], b8)
        fm3 = group4_branch(self.layer["fm3"], b7)

        aux1 = self.layer["aux_out1"](fm1)
        aux2 = self.layer["aux_out2"](fm2)
        aux3 = self.layer["aux_out3"](fm3)

        out = torch.cat(tensors=(fm1, fm2, fm3), dim=1)
        out = self.layer["out"](out)

        return [out, aux1, aux2, aux3]
