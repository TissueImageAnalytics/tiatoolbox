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

"""Defines a set of models to be used within tiatoolbox."""

import os
import pathlib
from pydoc import locate
from typing import Union

import torch

from tiatoolbox import rcParam
from tiatoolbox.models.architecture.vanilla import CNNBackbone, CNNModel
from tiatoolbox.models.dataset.classification import predefined_preproc_func
from tiatoolbox.utils.misc import download_data

__all__ = ["get_pretrained_model"]
PRETRAINED_INFO = rcParam["pretrained_model_info"]


def fetch_pretrained_weights(model_name: str, save_path: str, overwrite: bool = True):
    """Get the pretrained model information from yml file.
    Args:
        model_name (str): Refer to `::py::meth:get_pretrained_model` for
          all supported model names.
        save_path (str): Path to save the weight of the
          corresponding `model_name`.
        overwrite (bool): To always overwriting downloaded weights.
    """
    info = PRETRAINED_INFO[model_name]
    download_data(info["url"], save_path, overwrite)


def get_pretrained_model(
    pretrained_model: str = None,
    pretrained_weights: Union[str, pathlib.Path] = None,
    overwrite: bool = False,
):
    """Load a predefined PyTorch model with the appropriate pretrained weights.

    Args:
        pretrained_model (str): Name of the existing models support by tiatoolbox
          for processing the data. The models currently supported:
            - alexnet
            - resnet18
            - resnet34
            - resnet50
            - resnet101
            - resnext5032x4d
            - resnext101_32x8d
            - wide_resnet50_2
            - wide_resnet101_2
            - densenet121
            - densenet161
            - densenet169
            - densenet201
            - mobilenet_v2
            - mobilenet_v3_large
            - mobilenet_v3_small
            - googlenet

          Each model has been trained on the Kather100K and PCam datasets. The format
          of pretrained_model is <model_name>-<dataset_name>. For example, to use a
          resnet18 model trained on Kather100K, use `resnet18-kather100k and to use
          an alexnet model trained on PCam, use `alexnet-pcam`.


          By default, the corresponding pretrained weights will also be
          downloaded. However, you can override with your own set of weights via
          the `pretrained_weights` argument. Argument is case insensitive.
        pretrained_weights (str): Path to the weight of the
          corresponding `pretrained_model`.
        overwrite (bool): To always overwriting downloaded weights.

    Examples:
        >>> # get mobilenet pretrained on Kather100K dataset by the TIA team
        >>> model = get_pretrained_model(pretrained_model='mobilenet_v2-kather100k')
        >>> # get mobilenet defined by TIA team, but loaded with user defined weights
        >>> model = get_pretrained_model(
        ...     pretrained_model='mobilenet_v2-kather100k'
        ...     pretrained_weights='/A/B/C/my_weights.tar'
        ... )
        >>> # get resnet34 pretrained on PCam dataset by TIA team
        >>> model = get_pretrained_model(pretrained_model='resnet34-pcam')

    """
    if not isinstance(pretrained_model, str):
        raise ValueError("pretrained_model must be a string.")

    if pretrained_model not in PRETRAINED_INFO:
        raise ValueError(f"Pretrained model `{pretrained_model}` does not exist.")

    info = PRETRAINED_INFO[pretrained_model]

    arch_info = info["architecture"]
    creator = locate((f"tiatoolbox.models.architecture" f'.{arch_info["class"]}'))

    model = creator(**arch_info["kwargs"])
    # TODO: a dictionary of dataset specific or transformation ?
    if "dataset" in info:
        # ! this is a hack currently, need another PR to clean up
        # ! associated pre-processing coming from dataset (Kumar, Kather, etc.)
        model.preproc_func = predefined_preproc_func(info["dataset"])

    if pretrained_weights is None:
        file_name = info["url"].split("/")[-1]
        pretrained_weights = os.path.join(
            rcParam["TIATOOLBOX_HOME"], "models/", file_name
        )
        fetch_pretrained_weights(pretrained_model, pretrained_weights, overwrite)

    # ! assume to be saved in single GPU mode
    # always load on to the CPU
    saved_state_dict = torch.load(pretrained_weights, map_location="cpu")
    model.load_state_dict(saved_state_dict, strict=True)

    # !
    io_info = info["ioconfig"]
    creator = locate((f"tiatoolbox.models.engine" f'.{io_info["class"]}'))

    iostate = creator(**io_info["kwargs"])
    return model, iostate
