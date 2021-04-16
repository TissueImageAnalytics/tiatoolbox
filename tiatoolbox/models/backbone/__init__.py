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
# The Original Code is Copyright (C) 2020, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****


import torchvision.models as torch_models
import torch.nn as nn


def get_model(backbone, **kwargs):
    """Get a model. Models are either already defined within torchvision
    or they can be custom-made within tiatoolbox.

    Args:
        backbone (str): model name.
        **kwargs:

    Returns:
        list of PyTorch network layers within nn.Sequential.

    """
    backbone_dict = {
        "resnet18": torch_models.resnet18,
        "resnet34": torch_models.resnet34,
        "resnet50": torch_models.resnet50,
        "densenet121": torch_models.densenet121,
    }
    creator = backbone_dict[backbone]
    model = creator(**kwargs)  # ! abit too hacky

    # unroll all the definition and strip off the final GAP and FCN
    # different model will have diffent form, sample resnet and densenet atm
    if "resnet" in backbone:
        feat_extract = nn.Sequential(*list(model.children())[:-2])
    elif "densenet" in backbone:
        feat_extract = nn.Sequential(*list(model.children())[0])
    return feat_extract
