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
"""Tests for Semantic Segmentor."""

import os
import pathlib
import shutil

import numpy as np
import torch

from tiatoolbox.models.architecture.vanilla import CNNExtractor
from tiatoolbox.models.controller.semantic_segmentor import (
    FeatureExtractor,
    IOSegmentorConfig,
)
from tiatoolbox.wsicore.wsireader import get_wsireader

ON_GPU = False
BATCH_SIZE = 4
# ----------------------------------------------------


def _rm_dir(path):
    """Helper func to remove directory."""
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


# -------------------------------------------------------------------------------------
# Controller
# -------------------------------------------------------------------------------------


def test_functional_feature_extractor(remote_sample, tmp_path):
    """Test for feature extraction."""
    save_dir = pathlib.Path(tmp_path)
    # # convert to pathlib Path to prevent wsireader complaint
    mini_wsi_svs = pathlib.Path(remote_sample("wsi2_4k_4k_svs"))

    # pre-emptive clean up
    _rm_dir(save_dir)  # default output dir test

    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {"units": "mpp", "resolution": 0.25},
        ],
        output_resolutions=[
            {"units": "mpp", "resolution": 0.25},
        ],
        patch_input_shape=[512, 512],
        patch_output_shape=[512, 512],
        stride_shape=[256, 256],
        save_resolution={"units": "mpp", "resolution": 8.0},
    )

    model = CNNExtractor("resnet50")
    extractor = FeatureExtractor(batch_size=BATCH_SIZE, model=model)
    # should still run because we skip exception
    output_list = extractor.predict(
        [mini_wsi_svs],
        mode="wsi",
        ioconfig=ioconfig,
        on_gpu=ON_GPU,
        crash_on_exception=False,
        save_dir=save_dir,
    )
    wsi_0_root_path = output_list[0][1]
    positions = np.load(f"{wsi_0_root_path}.position.npy")
    features = np.load(f"{wsi_0_root_path}.features.0.npy")

    reader = get_wsireader(mini_wsi_svs)
    patches = [
        reader.read_bounds(
            positions[patch_idx],
            resolution=0.25,
            units="mpp",
            pad_constant_values=0,
            coord_space="resolution",
        )
        for patch_idx in range(4)
    ]
    patches = np.array(patches)
    patches = torch.from_numpy(patches)  # NHWC
    patches = patches.permute(0, 3, 1, 2)  # NCHW
    patches = patches.type(torch.float32)
    model = model.to("cpu")
    # Inference mode
    model.eval()
    with torch.inference_mode():
        _features = model(patches).numpy()
    # ! must maintain same batch size and likely same ordering
    # ! else the output values will not exactly be the same (still < 1.0e-5
    # ! of epsilon though)
    assert np.mean(np.abs(features[:BATCH_SIZE] - _features)) < 1.0e-6

    _rm_dir(save_dir)
