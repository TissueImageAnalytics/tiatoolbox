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

"""Defines a set of models to be used within tiatoolbox."""

import torchvision.models as torch_models
import torch.nn as nn

__all__ = ["get_model"]


def get_model(backbone, pretrained=True, **kwargs):
    """Get a model.

    Models are either already defined within torchvision
    or they can be custom-made within tiatoolbox.

    Args:
        backbone (str): Model name.

    Returns:
        List of PyTorch network layers wrapped with nn.Sequential.
        https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html

    """
    backbone_dict = {
        "alexnet": torch_models.alexnet,
        "resnet18": torch_models.resnet18,
        "resnet34": torch_models.resnet34,
        "resnet50": torch_models.resnet50,
        "resnet101": torch_models.resnet101,
        "resnext50_32x4d": torch_models.resnext50_32x4d,
        "resnext101_32x8d": torch_models.resnext101_32x8d,
        "wide_resnet50_2": torch_models.wide_resnet50_2,
        "wide_resnet101_2": torch_models.wide_resnet101_2,
        "densenet121": torch_models.densenet121,
        "densenet161": torch_models.densenet161,
        "densenet169": torch_models.densenet169,
        "densenet201": torch_models.densenet201,
        "inception_v3": torch_models.inception_v3,
        "googlenet": torch_models.googlenet,
        "mobilenet_v2": torch_models.mobilenet_v2,
        "mobilenet_v3_large": torch_models.mobilenet_v3_large,
        "mobilenet_v3_small": torch_models.mobilenet_v3_small,
    }
    if backbone not in backbone_dict:
        raise ValueError("Backbone `%s` is not supported." % backbone)

    creator = backbone_dict[backbone]
    model = creator(pretrained=pretrained, **kwargs)

    # Unroll all the definition and strip off the final GAP and FCN
    if "resnet" in backbone or "resnext" in backbone:
        feat_extract = nn.Sequential(*list(model.children())[:-2])
    elif "densenet" in backbone:
        feat_extract = model.features
    elif "alexnet" in backbone:
        feat_extract = model.features
    if "inception_v3" in backbone or "googlenet" in backbone:
        feat_extract = nn.Sequential(*list(model.children())[:-3])
    if "mobilenet" in backbone:
        feat_extract = model.features
    return feat_extract
