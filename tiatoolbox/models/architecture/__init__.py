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

import importlib
import os
from pydoc import locate

import torch

from tiatoolbox import rcParam
from tiatoolbox.models.dataset.classification import predefined_preproc_func
from tiatoolbox.utils.misc import download_data, get_pretrained_model_info

__all__ = ["get_pretrained_model"]


def get_pretrained_model(pretrained_model=None, pretrained_weights=None):
    """Load a predefined PyTorch model with the appropriate pretrained weights.

    Args:
        pretrained_model (str): Name of the existing models support by tiatoolbox
          for processing the data. Currently supports:
            - alexnet-kather100k: alexnet backbone trained on Kather 100k dataset.
            - resnet18-kather100k: resnet18 backbone trained on Kather 100k dataset.
            - resnet34-kather100k: resnet34 backbone trained on Kather 100k dataset.
            - resnet50-kather100k: resnet50 backbone trained on Kather 100k dataset.
            - resnet101-kather100k: resnet101 backbone trained on Kather 100k dataset.
            - resnext5032x4d-kather100k: resnext50_32x4d backbone trained on Kather
              100k dataset.
            - resnext101_32x8d-kather100k: resnext101_32x8d backbone trained on
              Kather 100k dataset.
            - wide_resnet50_2-kather100k: wide_resnet50_2 backbone trained on
              Kather 100k dataset.
            - wide_resnet101_2-kather100k: wide_resnet101_2 backbone trained on
              Kather 100k dataset.
            - densenet121-kather100k: densenet121 backbone trained on
              Kather 100k dataset.
            - densenet161-kather100k: densenet161 backbone trained on
              Kather 100k dataset.
            - densenet169-kather100k: densenet169 backbone trained on
              Kather 100k dataset.
            - densenet201-kather100k: densenet201 backbone trained on
              Kather 100k dataset.
            - mobilenet_v2-kather100k: mobilenet_v2 backbone trained on
              Kather 100k dataset.
            - mobilenet_v3_large-kather100k: mobilenet_v3_large backbone trained on
              Kather 100k dataset.
            - mobilenet_v3_small-kather100k: mobilenet_v3_small backbone trained on
              Kather 100k dataset.
            - googlenet-kather100k: googlenet backbone trained on Kather 100k dataset.

          By default, the corresponding pretrained weights will also be
          downloaded. However, you can override with your own set of weights via
          the `pretrained_weights` argument. Argument is case insensitive.
        pretrained_weights (str): Path to the weight of the
          corresponding `pretrained_model`.

    Examples:
        >>> # get mobilenet pretrained on kather by TIA team
        >>> model = get_pretrained_model(pretrained_model='mobilenet_v2-kather100k')
        >>> # get mobilenet defined by TIA team but loaded with user weight
        >>> model = get_pretrained_model(
        ...     pretrained_model='mobilenet_v2-kather100k'
        ...     pretrained_weights='/A/B/C/my_weights.tar'
        ... )

    """
    if not isinstance(pretrained_model, str):
        raise ValueError("pretrained_model must be a string.")

    # parsing protocol
    pretrained_yml = get_pretrained_model_info()

    if pretrained_model not in pretrained_yml:
        raise ValueError(f"Pretrained model `{pretrained_model}` does not exist.")

    info = pretrained_yml[pretrained_model]

    arch_info = info["architecture"]
    creator = locate((f"tiatoolbox.models.architecture" f'.{arch_info["class"]}'))

    model = creator(**arch_info["kwargs"])
    # TODO: a dictionary of dataset specific or transformation ?
    if "dataset" in info:
        # ! this is a hack currently, need another PR to clean up
        # ! associated pre-proc coming from dataset (Kumar, Kather, etc.)
        model.preproc_func = predefined_preproc_func(info["dataset"])

    if pretrained_weights is None:
        pretrained_weights_url = info["url"]
        pretrained_weights_url_split = pretrained_weights_url.split("/")
        pretrained_weights = os.path.join(
            rcParam["TIATOOLBOX_HOME"], "models/", pretrained_weights_url_split[-1]
        )
        if not os.path.exists(pretrained_weights):
            download_data(pretrained_weights_url, pretrained_weights)

    # ! assume to be saved in single GPU mode
    # always load on to the CPU
    saved_state_dict = torch.load(pretrained_weights, map_location="cpu")
    model.load_state_dict(saved_state_dict, strict=True)

    # !
    io_info = info["ioconfig"]
    creator = locate((f"tiatoolbox.models.controller" f'.{io_info["class"]}'))

    iostate = creator(**io_info["kwargs"])
    return model, iostate
