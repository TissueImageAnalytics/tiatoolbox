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

import numpy as np
import pytest
import torch

from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.architecture.hovernet_plus import HoVerNetPlus
from tiatoolbox.utils.misc import imread
from tiatoolbox.utils.transforms import imresize


def test_functionality(remote_sample, tmp_path):
    """Functionality test."""
    tmp_path = str(tmp_path)
    sample_patch = str(remote_sample("stainnorm-source"))
    patch = imread(sample_patch)
    patch = imresize(patch, scale_factor=0.5)
    patch = patch[0:256, 0:256]
    batch = torch.from_numpy(patch)[None]

    # Test functionality with both nuclei and layer segmentation
    model = HoVerNetPlus(num_types=3, num_layers=5, mode="fast")
    # Test decoder as expected
    assert len(model.decoder["np"]) > 0, "Decoder must contain np branch."
    assert len(model.decoder["hv"]) > 0, "Decoder must contain hv branch."
    assert len(model.decoder["tp"]) > 0, "Decoder must contain tp branch."
    assert len(model.decoder["ls"]) > 0, "Decoder must contain ls branch."
    fetch_pretrained_weights("hovernetplus-oed", f"{tmp_path}/weigths.pth")
    pretrained = torch.load(f"{tmp_path}/weigths.pth")
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, on_gpu=False)
    assert len(output) == 4, "Must contain predictions for: np, hv, tp and ls branches."
    output = [v[0] for v in output]
    output = model.postproc(output)
    assert len(output[1]) > 0 and len(output[3]) > 0, "Must have some nuclei/layers."

    # Test functionality with both nuclei segmentation and classification alone
    model = HoVerNetPlus(num_types=6, num_layers=None, mode="fast")
    fetch_pretrained_weights("hovernet_fast-pannuke", f"{tmp_path}/weigths.pth")
    pretrained = torch.load(f"{tmp_path}/weigths.pth")
    model.load_state_dict(pretrained)
    # Test decoder as expected
    assert len(model.decoder["np"]) > 0, "Decoder must contain np branch."
    assert len(model.decoder["hv"]) > 0, "Decoder must contain hv branch."
    assert len(model.decoder["tp"]) > 0, "Decoder must contain tp branch."
    output = model.infer_batch(model, batch, on_gpu=False)
    assert len(output) == 3, "Must contain predictions for: np, hv and tp branches."
    output = [v[0] for v in output]
    output = model.postproc(output)
    assert len(output[1]) > 0, "Must have some nuclei."

    # Test functionality with both nuclei segmentation and classification alone
    model = HoVerNetPlus(num_types=6, num_layers=None, mode="original")
    fetch_pretrained_weights("hovernet_original-pannuke", f"{tmp_path}/weigths.pth")
    pretrained = torch.load(f"{tmp_path}/weigths.pth")
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, on_gpu=False)
    assert len(output) == 3, "Must contain predictions for: np, hv and tp branches."
    output = [v[0] for v in output]
    output = model.postproc(output)
    assert len(output[1]) > 0, "Must have some nuclei."

    # Test functionality with nuclei segmentation alone
    model = HoVerNetPlus(num_types=None, num_layers=None, mode="fast")
    fetch_pretrained_weights("hovernet_fast-pannuke", f"{tmp_path}/weigths.pth")
    pretrained = torch.load(f"{tmp_path}/weigths.pth")
    model.load_state_dict(pretrained)
    # Test decoder as expected
    assert len(model.decoder["np"]) > 0, "Decoder must contain np branch."
    assert len(model.decoder["hv"]) > 0, "Decoder must contain hv branch."
    output = model.infer_batch(model, batch, on_gpu=False)
    assert len(output) == 2, "Must contain predictions for: np and hv branches."
    output = [v[0] for v in output]
    output = model.postproc(output)
    assert len(output[1]) > 0, "Must have some nuclei."

    # Test functionality with layer segmentation alone
    model = HoVerNetPlus(num_types=None, num_layers=5, mode="fast")
    fetch_pretrained_weights("hovernet-oed", f"{tmp_path}/weigths.pth")
    pretrained = torch.load(f"{tmp_path}/weigths.pth")
    model.load_state_dict(pretrained)
    # Test decoder as expected
    assert len(model.decoder["ls"]) > 0, "Decoder must contain ls branch."
    output = model.infer_batch(model, batch, on_gpu=False)
    assert len(output) > 0, "Must contain predictions for: ls branch."
    output = output[0][np.newaxis, :, :]
    output = model.postproc(output)
    assert len(output[1]) > 0, "Must have some layers."

    # test crash when providing exotic mode
    with pytest.raises(ValueError, match=r".*Invalid mode.*"):
        model = HoVerNetPlus(num_types=None, num_layers=None, mode="super")
