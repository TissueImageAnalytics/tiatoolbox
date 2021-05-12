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

"""Module that defines the pretrained patch classification models available."""

import os

models_root = "https://tiatoolbox.dcs.warwick.ac.uk/models/"

# ! any additional dataset name should be also added to tiatoolbox.models.dataset
__pretrained_model = {
    "alexnet-kather100K": {
        "pretrained": os.path.join(models_root, "alexnet-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "resnet18-kather100K": {
        "pretrained": os.path.join(models_root, "resnet18-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "resnet34-kather100K": {
        "pretrained": os.path.join(models_root, "resnet34-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "resnet50-kather100K": {
        "pretrained": os.path.join(models_root, "resnet50-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "resnet101-kather100K": {
        "pretrained": os.path.join(models_root, "resnet101-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "resnext50_32x4d-kather100K": {
        "pretrained": os.path.join(models_root, "resnext50_32x4d-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "resnext101_32x8d-kather100K": {
        "pretrained": os.path.join(models_root, "resnext101_32x8d-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "wide_resnet50_2-kather100K": {
        "pretrained": os.path.join(models_root, "wide_resnet50_2-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "wide_resnet101_2-kather100K": {
        "pretrained": os.path.join(models_root, "wide_resnet101_2-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "densenet121-kather100K": {
        "pretrained": os.path.join(models_root, "densenet121-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "densenet161-kather100K": {
        "pretrained": os.path.join(models_root, "densenet161-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "densenet169-kather100K": {
        "pretrained": os.path.join(models_root, "densenet169-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "densenet201-kather100K": {
        "pretrained": os.path.join(models_root, "densenet201-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "mobilenet_v2-kather100K": {
        "pretrained": os.path.join(models_root, "mobilenet_v2-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "mobilenet_v3_large-kather100K": {
        "pretrained": os.path.join(models_root, "mobilenet_v3_large-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "mobilenet_v3_small-kather100K": {
        "pretrained": os.path.join(models_root, "mobilenet_v3_small-kather100K-pc.pth"),
        "nr_classes": 9,
    },
    "googlenet-kather100K": {
        "pretrained": os.path.join(models_root, "googlenet-kather100K-pc.pth"),
        "nr_classes": 9,
    },
}

# to ensure easy matching
__pretrained_model = {k.lower(): v for k, v in __pretrained_model.items()}
