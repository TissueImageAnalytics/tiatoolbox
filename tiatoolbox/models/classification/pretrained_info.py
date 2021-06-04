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

"""Module that defines the pretrained patch classification models available."""

import os

models_root = "https://tiatoolbox.dcs.warwick.ac.uk/models/"

# ! any additional dataset name should be also added to tiatoolbox.models.dataset
_pretrained_model = {
    "alexnet-kather100k": {
        "pretrained": os.path.join(models_root, "pc/alexnet-kather100k.pth"),
        "num_classes": 9,
    },
    "resnet18-kather100k": {
        "pretrained": os.path.join(models_root, "pc/resnet18-kather100k.pth"),
        "num_classes": 9,
    },
    "resnet34-kather100k": {
        "pretrained": os.path.join(models_root, "pc/resnet34-kather100k.pth"),
        "num_classes": 9,
    },
    "resnet50-kather100k": {
        "pretrained": os.path.join(models_root, "pc/resnet50-kather100k.pth"),
        "num_classes": 9,
    },
    "resnet101-kather100k": {
        "pretrained": os.path.join(models_root, "pc/resnet101-kather100k.pth"),
        "num_classes": 9,
    },
    "resnext50_32x4d-kather100k": {
        "pretrained": os.path.join(models_root, "pc/resnext50_32x4d-kather100k.pth"),
        "num_classes": 9,
    },
    "resnext101_32x8d-kather100k": {
        "pretrained": os.path.join(models_root, "pc/resnext101_32x8d-kather100k.pth"),
        "num_classes": 9,
    },
    "wide_resnet50_2-kather100k": {
        "pretrained": os.path.join(models_root, "pc/wide_resnet50_2-kather100k.pth"),
        "num_classes": 9,
    },
    "wide_resnet101_2-kather100k": {
        "pretrained": os.path.join(models_root, "pc/wide_resnet101_2-kather100k.pth"),
        "num_classes": 9,
    },
    "densenet121-kather100k": {
        "pretrained": os.path.join(models_root, "pc/densenet121-kather100k.pth"),
        "num_classes": 9,
    },
    "densenet161-kather100k": {
        "pretrained": os.path.join(models_root, "pc/densenet161-kather100k.pth"),
        "num_classes": 9,
    },
    "densenet169-kather100k": {
        "pretrained": os.path.join(models_root, "pc/densenet169-kather100k.pth"),
        "num_classes": 9,
    },
    "densenet201-kather100k": {
        "pretrained": os.path.join(models_root, "pc/densenet201-kather100k.pth"),
        "num_classes": 9,
    },
    "mobilenet_v2-kather100k": {
        "pretrained": os.path.join(models_root, "pc/mobilenet_v2-kather100k.pth"),
        "num_classes": 9,
    },
    "mobilenet_v3_large-kather100k": {
        "pretrained": os.path.join(models_root, "pc/mobilenet_v3_large-kather100k.pth"),
        "num_classes": 9,
    },
    "mobilenet_v3_small-kather100k": {
        "pretrained": os.path.join(models_root, "pc/mobilenet_v3_small-kather100k.pth"),
        "num_classes": 9,
    },
    "googlenet-kather100k": {
        "pretrained": os.path.join(models_root, "pc/googlenet-kather100k.pth"),
        "num_classes": 9,
    },
}

# To ensure easy matching
_pretrained_model = {k.lower(): v for k, v in _pretrained_model.items()}
