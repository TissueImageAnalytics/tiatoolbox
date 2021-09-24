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

"""Unit test package for Unet."""

import pathlib

import numpy as np
import pytest
import torch

from tiatoolbox.models.backbone.unet import UNetModel
from tiatoolbox.wsicore.wsireader import get_wsireader

# Test pretrained Model =============================


def test_functional_unet(_sample_wsi_dict, _sample_pretrained_dict):
    # convert to pathlib Path to prevent wsireader complaint
    _mini_wsi_svs = pathlib.Path(_sample_wsi_dict["CMU-mini"])
    _pretrained_path = str(_sample_pretrained_dict["fcn-tissue_mask"])
    print(_pretrained_path)

    reader = get_wsireader(_mini_wsi_svs)
    with pytest.raises(ValueError, match=r".*Unknown encoder*"):
        model = UNetModel(3, 2, encoder="resnet101")

    # test creation
    model = UNetModel(5, 5, encoder="resnet50")
    model = UNetModel(3, 2, encoder="resnet50")
    model = UNetModel(3, 2, encoder="unet")

    # test inference
    read_kwargs = dict(resolution=2.0, units="mpp", coord_space="resolution")
    batch = np.array(
        [
            # noqa
            reader.read_bounds([0, 0, 1024, 1024], **read_kwargs),
            reader.read_bounds([1024, 1024, 2048, 2048], **read_kwargs),
        ]
    )
    batch = torch.from_numpy(batch)

    model = UNetModel(3, 2, encoder="resnet50")
    pretrained = torch.load(_pretrained_path, map_location="cpu")
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, on_gpu=False)
    output = output[0]
