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

"""Unit test package for HoVerNet+."""

import pathlib
import shutil

import numpy as np
import pytest
import torch

from tiatoolbox.models import IOSegmentorConfig, MultiTaskSegmentor, SemanticSegmentor

BATCH_SIZE = 1
ON_TRAVIS = True
ON_GPU = not ON_TRAVIS and torch.cuda.is_available()

# ----------------------------------------------------


def _rm_dir(path):
    """Helper func to remove directory."""
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.skip(reason="Local manual test, not applicable for travis.")
def test_functionality_local(remote_sample, tmp_path):
    """Local functionality test for multi task segmentor. Currently only
    testing HoVer-Net+ with semantic segmentor.
    """
    root_save_dir = pathlib.Path(tmp_path)
    mini_wsi_svs = pathlib.Path(remote_sample("CMU-1-Small-Region.svs"))

    save_dir = f"{root_save_dir}/semantic/"
    _rm_dir(save_dir)
    semantic_segmentor = SemanticSegmentor(
        pretrained_model="hovernetplus-oed",
        batch_size=BATCH_SIZE,
        num_postproc_workers=2,
    )
    output = semantic_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        on_gpu=True,
        crash_on_exception=True,
        save_dir=save_dir,
    )

    raw_maps = [np.load(f"{output[0][1]}.raw.{head_idx}.npy") for head_idx in range(4)]
    inst_map, inst_dict, layer_map, layer_dict = semantic_segmentor.model.postproc(
        raw_maps
    )
    assert len(inst_dict) > 0 and len(layer_dict) > 0, "Must have some nuclei/layers."
    assert (
        inst_map.shape == layer_map.shape
    ), "Output instance and layer maps must be same shape"
    _rm_dir(tmp_path)


def test_functionality_travis(remote_sample, tmp_path):
    """Functionality test for multi task segmentor."""
    root_save_dir = pathlib.Path(tmp_path)
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))
    required_dims = (258, 258)
    # above image is 512x512 at 0.252 mpp resolution. This is 258x258 at 0.500 mpp.

    resolution = 2
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": resolution}],
        output_resolutions=[
            {"units": "mpp", "resolution": resolution},
            {"units": "mpp", "resolution": resolution},
            {"units": "mpp", "resolution": resolution},
            {"units": "mpp", "resolution": resolution},
        ],
        margin=128,
        tile_shape=[512, 512],
        patch_input_shape=[256, 256],
        patch_output_shape=[164, 164],
        stride_shape=[164, 164],
    )

    save_dir = f"{root_save_dir}/multi/"
    _rm_dir(save_dir)

    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="hovernetplus-oed",
        batch_size=BATCH_SIZE,
        num_postproc_workers=2,
    )
    output = multi_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        on_gpu=True,
        ioconfig=ioconfig,
        crash_on_exception=True,
        save_dir=save_dir,
    )

    layer_map = np.load(f"{output[0][1]}.npy")

    assert (
        layer_map.shape == required_dims
    ), "Output layer map dimensions must be same as the expected output shape"
    _rm_dir(tmp_path)
