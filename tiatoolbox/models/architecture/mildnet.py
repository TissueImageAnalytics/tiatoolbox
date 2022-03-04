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
"""Defines MILD-Net architecture.

Graham, Simon et al., “MILD-Net: Minimal information loss dilated network for
gland instance segmentation in colon histology images.,” Medical Image Analysis,
Feb. 2019, vol. 52, p. 199–211.

"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from scipy import ndimage
from skimage import morphology

from tiatoolbox.models.abc import ModelABC
from tiatoolbox.models.architecture.hovernet import HoVerNet
from tiatoolbox.utils import misc



def conv_block_module(in_ch, out_ch):
    """Defines a convolution block, consisting of 2 convolution
    operations, with batch normalisation and ReLU activation.

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
        nn.ReLU(),
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
        nn.ReLU(),
    )
    
    return nn.ModuleDict(module_dict)


def mil_residual_block_module(in_ch, out_ch):
    """Defines a minimal information loss (MIL) residual block, as used in MILD-Net.

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
        nn.ReLU(),
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
        nn.ReLU(),
    )
    
    return nn.ModuleDict(module_dict)


def dilated_residual_block_module(in_ch, out_ch, dilation):
    """Defines a dilated residual block, as used in MILD-Net.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        dilation (int): Dilation rate used during convolution.

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
            dilation=dilation,
            bias=True,
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )
    module_dict["conv2"] = nn.Sequential(
        nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            dilation=dilation,
            bias=True,
        ),
        nn.BatchNorm2d(out_ch),
    )
    
    return nn.ModuleDict(module_dict)

def get_aspp_layer(in_ch, out_ch, dilation):
    """Defines a series of convolutions used in the ASPP unit, consisting of
    a single 3x3 dilated convolution and 2 1x1 convolutions.
    
    Args:
        in_ch (int): Number of input channels.
        out_ch (list): List of length 2 indicating the number of output channels.
        dilation (int): Dilation rate in the first 3x3 convolution."""
    
    module = nn.Sequential(
        nn.Conv2d(
            in_ch,
            out_ch[0],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            dilation=dilation,
            bias=True,
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(
            out_ch[0],
            out_ch[0],
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=True,
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(
            out_ch[0],
            out_ch[1],
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=True,
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )
    
    return module

def aspp_unit_module(in_ch, out_ch, dilation):
    """Defines the ASPP unit, as used in MILD-Net. The unit applies
    three 3x3 dilated convolutions and a single 1x1 convolution and 
    the resuls are concatenated together.

    Args:
        in_ch (int): Number of input channels.
        out_ch (list): List of out channels. Should be a list of length 2, denoting
            the number of output channels for each convolution.
        dilation (list): Dilation rates for each 3x3 convolution. Should be a list
            of length 4, denoting the dilation rate for each 3x3 dilated convolution.

    Returns:
        torch.nn.ModuleDict: An output of type :class:`torch.nn.ModuleDict`

    """

    module_dict = OrderedDict()
    module_dict["layer1"] = get_aspp_layer(in_ch, out_ch, dilation=dilation[0])
    module_dict["layer2"] = get_aspp_layer(in_ch, out_ch, dilation=dilation[1])
    module_dict["layer3"] = get_aspp_layer(in_ch, out_ch, dilation=dilation[2])
    module_dict["layer4"] = nn.Sequential(
        nn.Conv2d(
            in_ch,
            out_ch[0],
            kernel_size=(1, 1),
            bias=True,
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(
            out_ch[0],
            out_ch[1],
            kernel_size=(1, 1),
            bias=True,
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        )
    
    return nn.ModuleDict(module_dict)


def conv_block_forward(layer, in_tensor):
    """Defines convolution block forward pass.

    Args:
        layer (torch.nn.Module): Network layer.
        in_tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output of group 1 layer.

    """
    a = layer["conv1"](in_tensor)
    return layer["conv2"](a)


def mil_residual_block_forward(layer, in_tensor1, in_tensor2):
    """Defines minimal information loss residual unit forward pass.

    Args:
        layer (torch.nn.Module): Network layer.
        in_tensor1 (torch.Tensor): Input tensor from previous layer.
        in_tensor2 (torch.Tensor): Input tensor of original image.

    Returns:
        torch.Tensor: Output of group 1 layer.

    """
    a = layer["conv1"](in_tensor1)
    return layer["conv2"](a)
    

def dilated_residual_block_forward(layer, in_tensor):
    """Defines dilated residual unit forward pass.

    Args:
        layer (torch.nn.Module): Network layer.
        in_tensor (torch.Tensor): Input tensor from previous layer.

    Returns:
        torch.Tensor: Output of group 1 layer.

    """
    a = layer["conv1"](in_tensor)
    b = layer["conv2"](a)
    return 


def aspp_unit_forward(layer, in_tensor):
    """Defines ASPP unit forward pass.

    Args:
        layer (torch.nn.Module): Network layer.
        in_tensor (torch.Tensor): Input tensor from previous layer.

    Returns:
        torch.Tensor: Output of group 1 layer.

    """
    a = layer["layer1"](in_tensor)
    b = layer["layer2"](in_tensor)
    c = layer["layer3"](in_tensor)
    d = layer["layer4"](in_tensor)
    
    return torch.cat(input_tensors=(a, b, c, d))

class MILDNet(ModelABC):
    """Initialise MILDNet [1].
    The following models have been included in tiatoolbox.
    1. `mildnet-glas`: This is trained on
    `GlaS dataset <https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest//>`_
    The model is retrained in PyTorch, as the original model with results on GlaS
    was trained in TensorFlow.

    The tiatoolbox model should produce following results on the GlaS dataset.

    Args:
        num_input_channels (int): Number of channels in input. default=3.
        num_class (int): Number of output channels. default=2.

    References:
        [1] Graham, Simon et al., “MILD-Net: Minimal information loss dilated
        network for gland instance segmentation in colon histology images.,”
        Medical Image Analysis, Feb. 2019, vol. 52, p. 199–211.

    """

    def __init__(self, num_input_channels=3, num_class=2):
        super().__init__()
        if num_class < 2:
            raise ValueError("Number of classes should be >=2.")
        self.__num_class = num_class
        self.in_ch = num_input_channels

        module_dict = OrderedDict()
        module_dict["e1"] = conv_block_module(num_input_channels, 64)
        
        module_dict["e2"] = mil_residual_block_module(64, 128)
        module_dict["e3"] = mil_residual_block_module(128, 256)
        module_dict["e4"] = mil_residual_block_module(256, 512)
        
        module_dict["e5"] = dilated_residual_block_module(512, 512)
        module_dict["e6"] = dilated_residual_block_module(512, 512)
        
        module_dict["e7"] = aspp_unit_module(512, [1024, 128], dilation_rates=[1, 6, 12, 18])
        
        module_dict["d1"] = conv_block_module(1280, 256)
        module_dict["d2"] = conv_block_module(512, 128)
        module_dict["d3"] = conv_block_module(256, 64)
        
        module_dict["out"] = nn.Conv2d(64, num_class, kernel_size=(1, 1), bias=True)
        module_dict["aux"] = nn.Conv2d(512, num_class, kernel_size=(1, 1), bias=True)
        
        module_dict["skip1"] = nn.Conv2d(256, 640, kernel_size=(1, 1), bias=True)
        module_dict["skip2"] = nn.Conv2d(128, 256, kernel_size=(1, 1), bias=True)
        module_dict["skip3"] = nn.Conv2d(64, 128, kernel_size=(1, 1), bias=True)
        
        self.layer = nn.ModuleDict(module_dict)

    def forward(self, input_tensor: torch.Tensor):  # skipcq: PYL-W0221
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            input_tensor (torch.Tensor): Input images, the tensor is
                in the shape of NCHW.

        Returns:
            list: A list of main and auxiliary outputs.
                The expected format is [main_output, aux1, aux2, aux3].

        """
        def resize(feat, size=None, scale_factor=None, mode='bilinear'):
            return functional.interpolate(feat, size=size, scale_factor=scale_factor, mode=mode, align_corners=False)

        factor_list = [0.5, 0.25, 0.125]
        resize_input = [resize(input_tensor, scale_factor=x, mode='bicubic') for x in factor_list]
        
        #* encoder
        e1 = conv_block_forward(self.layer["e1"], input_tensor)
        p1 = functional.max_pool2d(e1, 2)
        ###
        e2 = mil_residual_block_forward(self.layer["e2"], p1, resize_input[0])
        p2 = functional.max_pool2d(e2, 2)
        ###
        e3 = mil_residual_block_forward(self.layer["e3"], p2, resize_input[1])
        p3 = functional.max_pool2d(e3, 2)
        ###
        e4 = mil_residual_block_forward(self.layer["e4"], p3, resize_input[2])
        e5 = dilated_residual_block_forward(self.layer["e5"], e4)
        e6 = dilated_residual_block_forward(self.layer["e6"], e5)
        e7 = aspp_unit_forward(self.layer["e7"], e6)
        
        #* decoder
        resized_e7 = resize(e7, scale_factor=2)
        skip1 = torch.cat(tensors=(self.layer["skip1"](e3), resized_e7))
        d1 = conv_block_forward(self.layer["d1"], skip1)
        ###
        resized_d1 = resize(d1, scale_factor=2)
        skip2 = torch.cat(tensors=(self.layer["skip2"](e2), resized_d1))
        d2 = conv_block_forward(self.layer["d2"], skip2)
        ###
        resized_d2 = resize(d2, scale_factor=2)
        skip3 = torch.cat(tensors=(self.layer["skip3"](e1), resized_d2))
        d3 = conv_block_forward(self.layer["d3"], skip3)
        ###
        out = self.layer["out"](d3)
        aux = resize(self.layer["aux"](e5), scale_factor=8)

        return [out, aux]

    @staticmethod
    def postproc(image: np.ndarray):
        """Post-processing script for MILDNet.

        Args:
            image (ndarray): input image of type numpy array.

        Returns:
            ndarray: pixel-wise nuclear instance segmentation
                prediction.

        """
        pred_bin = np.argmax(image[0], axis=2)
        pred_inst = ndimage.measurements.label(pred_bin)[0]
        pred_inst = morphology.remove_small_objects(pred_inst, min_size=50)
        canvas = np.zeros(pred_inst.shape[:2], dtype=np.int32)
        for inst_id in range(1, np.max(pred_inst) + 1):
            inst_map = np.array(pred_inst == inst_id, dtype=np.uint8)
            inst_map = ndimage.binary_fill_holes(inst_map)
            canvas[inst_map > 0] = inst_id
        nuc_inst_info_dict = HoVerNet.get_instance_info(canvas)
        return canvas, nuc_inst_info_dict

    @staticmethod
    def preproc(image: np.ndarray):
        """Preprocessing function for MicroNet.

        Performs per image standardization.

        Args:
            image (ndarray): input image of type numpy array.

        Returns:
            :class:`numpy.ndarray`: Pre-processed numpy array.

        """
        image = np.transpose(image, axes=(2, 0, 1))
        image = image / 255.0
        image = torch.from_numpy(image)

        image_mean = torch.mean(image, dim=(-1, -2, -3))
        stddev = torch.std(image, dim=(-1, -2, -3))
        num_pixels = torch.tensor(torch.numel(image), dtype=torch.float32)
        min_stddev = torch.rsqrt(num_pixels)
        adjusted_stddev = torch.max(stddev, min_stddev)

        image -= image_mean
        image = torch.div(image, adjusted_stddev)

        return np.transpose(image.numpy(), axes=(1, 2, 0))

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        """Run inference on an input batch.

        This contains logic for forward operation as well as batch i/o
        aggregation.

        Args:
            model (nn.Module): PyTorch defined model.
            batch_data (ndarray): a batch of data generated by
                torch.utils.data.DataLoader.
            on_gpu (bool): Whether to run inference on a GPU.

        Returns:
            List of output from each head, each head is expected to contain
            N predictions for N input patches. There are two cases, one
            with 2 heads (Nuclei Pixels `np` and Hover `hv`) or with 2 heads
            (`np`, `hv`, and Nuclei Types `tp`).

        """
        patch_imgs = batch_data

        device = misc.select_device(on_gpu)
        patch_imgs_gpu = patch_imgs.to(device).type(torch.float32)  # to NCHW
        patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

        model.eval()  # infer mode

        # --------------------------------------------------------------
        with torch.inference_mode():
            pred, _, _, _ = model(patch_imgs_gpu)

        pred = pred.permute(0, 2, 3, 1).contiguous()
        pred = pred.cpu().numpy()

        return [
            pred,
        ]
