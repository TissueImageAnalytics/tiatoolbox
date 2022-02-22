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

"""Unit test package for HoVerNet."""

import numpy as np
import pytest
import torch

from tiatoolbox import utils
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.architecture.micronet import MicroNet
from tiatoolbox.wsicore.wsireader import WSIReader


def test_functionality(remote_sample, tmp_path):
    """Functionality test."""
    tmp_path = str(tmp_path)
    sample_wsi = str(remote_sample("wsi1_2k_2k_svs"))
    reader = WSIReader.open(sample_wsi)

    # * test fast mode (architecture used in PanNuke paper)
    patch = reader.read_bounds(
        (0, 0, 252, 252), resolution=0.25, units="mpp", coord_space="resolution"
    )
    patch = MicroNet.preproc(patch)
    batch = torch.from_numpy(patch)[None]
    model = MicroNet()
    fetch_pretrained_weights("micronet_hovernet-consep", f"{tmp_path}/weights.pth")
    map_location = utils.misc.select_device(utils.env_detection.has_gpu())
    pretrained = torch.load(f"{tmp_path}/weights.pth", map_location=map_location)
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, on_gpu=False)
    output = model.postproc(output[0][0])
    assert np.max(np.unique(output)) == 33


def test_value_error():
    """Test to generate value error is num_classes < 2."""
    with pytest.raises(ValueError, match="Number of classes should be >=2"):
        _ = MicroNet(num_class=1)
