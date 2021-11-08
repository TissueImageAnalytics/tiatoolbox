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

"""Unit test package for HoVerNet+."""

import pytest
import torch

from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.architecture.hovernet_plus import HoVerNetPlus
from tiatoolbox.wsicore.wsireader import get_wsireader


def test_functionality(remote_sample, tmp_path):
    """Functionality test."""
    tmp_path = str(tmp_path)
    sample_wsi = str(remote_sample("wsi1_2k_2k_svs"))
    reader = get_wsireader(sample_wsi)

    #
    patch = reader.read_bounds(
        [0, 0, 256, 256], resolution=0.25, units="mpp", coord_space="resolution"
    )
    batch = torch.from_numpy(patch)[None]
    model = HoVerNetPlus(num_types=3, num_layers=5, mode="fast")
    fetch_pretrained_weights("hovernetplus-oed", f"{tmp_path}/weigths.pth")
    pretrained = torch.load(f"{tmp_path}/weigths.pth")
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, on_gpu=False)
    output = [v[0] for v in output]
    output = model.postproc(output)
    assert len(output[1]) > 0, "Must have some nuclei."
    assert len(output[3]) > 0, "Must have some layers."

    # test crash when providing exotic mode
    with pytest.raises(ValueError, match=r".*Invalid mode.*"):
        model = HoVerNetPlus(num_types=None, num_layers=None, mode="super")
