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

import pathlib

import numpy as np
import pytest
import torch

from tiatoolbox import utils
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.architecture.micronet import MicroNet
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.utils import env_detection as toolbox_env
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
    output = model.postproc(output[0])
    assert np.max(np.unique(output)) == 33


def test_value_error():
    """Test to generate value error is num_classes < 2."""
    with pytest.raises(ValueError, match="Number of classes should be >=2"):
        _ = MicroNet(num_class=1)


@pytest.mark.skipif(
    toolbox_env.running_on_travis() or not toolbox_env.has_gpu(),
    reason="Local test on machine with GPU.",
)
def test_micronet_output(remote_sample, tmp_path):
    """Tests the output of MicroNet."""
    svs_1_small = pathlib.Path(remote_sample("svs-1-small"))
    micronet_output = pathlib.Path(remote_sample("micronet-output"))
    pretrained_model = "micronet_hovernet-consep"
    batch_size = 5
    num_loader_workers = 0
    num_postproc_workers = 0

    predictor = SemanticSegmentor(
        pretrained_model=pretrained_model,
        batch_size=batch_size,
        num_loader_workers=num_loader_workers,
        num_postproc_workers=num_postproc_workers,
    )

    output = predictor.predict(
        imgs=[
            svs_1_small,
        ],
        save_dir=tmp_path / "output",
    )

    output = np.load(output[0][1] + ".raw.0.npy")
    output_on_server = np.load(str(micronet_output))
    assert (
        np.count_nonzero(
            np.round(output_on_server, 3) == np.round(output[500:1000, 1000:1500, :], 3)
        )
        / np.size(output_on_server)
        > 0.999
    )
