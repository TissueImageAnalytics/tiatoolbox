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

import torch

from tiatoolbox import rcParam
from tiatoolbox.utils.misc import download_data

from .hovernet import HoVerNet


models_root = "https://tiatoolbox.dcs.warwick.ac.uk/models/"

# ! any additional dataset name should be also added to tiatoolbox.models.dataset
_pretrained_model = {
    "hovernet-pannuke": {
        # "pretrained": os.path.join(models_root, "alexnet-kather100K-pc.pth"), # @simon to upload ?
        "pretrained" : '/home/tialab-dang/local/project/tiatoolbox/tests/pretrained/pannuke.pth',
        "num_types" : 6,
        "mode" : 'fast',
    },
}

# To ensure easy matching
_pretrained_model = {k.lower(): v for k, v in _pretrained_model.items()}


def get_pretrained_model(pretrained_model=None, pretrained_weight=None):
    """Load a predefined PyTorch model with the appropriate pretrained weights.


        pretrained_weight (str): Path to the weight of the
            corresponding `pretrained_model`.

    """
    if not isinstance(pretrained_model, str):
        raise ValueError("pretrained_model must be a string.")

    # parsing protocol
    pretrained_model = pretrained_model.lower()

    if pretrained_model not in _pretrained_model:
        raise ValueError("Pretrained model `%s` does not exist." % pretrained_model)
    cfg = _pretrained_model[pretrained_model]
    backbone, dataset = pretrained_model.split("-")

    if 'hovernet' in backbone:
        pretrained_weight_url = cfg.pop('pretrained')
        model = HoVerNet(**cfg)

    if pretrained_weight is None:
        pretrained_weight_url = cfg["pretrained"]
        pretrained_weight_url_split = pretrained_weight_url.split("/")
        pretrained_weight = os.path.join(
            rcParam["TIATOOLBOX_HOME"], "models/", pretrained_weight_url_split[-1]
        )
        if not os.path.exists(pretrained_weight):
            download_data(pretrained_weight_url, pretrained_weight)

    # ! assume to be saved in single GPU mode
    # always load on to the CPU
    saved_state_dict = torch.load(pretrained_weight, map_location="cpu")
    model.load_state_dict(saved_state_dict, strict=True)

    return model
