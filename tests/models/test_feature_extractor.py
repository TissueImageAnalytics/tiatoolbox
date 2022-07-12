"""Tests for feature extractor."""

import os
import pathlib
import shutil

import numpy as np
import torch

from tiatoolbox.models.architecture.vanilla import CNNBackbone
from tiatoolbox.models.engine.semantic_segmentor import (
    DeepFeatureExtractor,
    IOSegmentorConfig,
)
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.wsicore.wsireader import WSIReader

ON_GPU = not toolbox_env.running_on_ci() and toolbox_env.has_gpu()

# ----------------------------------------------------


def _rm_dir(path):
    """Helper func to remove directory."""
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


# -------------------------------------------------------------------------------------
# Engine
# -------------------------------------------------------------------------------------


def test_functional(remote_sample, tmp_path):
    """Test for feature extraction."""
    save_dir = pathlib.Path(f"{tmp_path}/output/")
    # # convert to pathlib Path to prevent wsireader complaint
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_1k_1k_svs"))

    # * test providing pretrained from torch vs pretrained_model.yaml
    _rm_dir(save_dir)  # default output dir test
    extractor = DeepFeatureExtractor(batch_size=1, pretrained_model="fcn-tissue_mask")
    output_list = extractor.predict(
        [mini_wsi_svs],
        mode="wsi",
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    wsi_0_root_path = output_list[0][1]
    positions = np.load(f"{wsi_0_root_path}.position.npy")
    features = np.load(f"{wsi_0_root_path}.features.0.npy")
    assert len(features.shape) == 4

    # * test same output between full infer and engine
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

    model = CNNBackbone("resnet50")
    extractor = DeepFeatureExtractor(batch_size=4, model=model)
    # should still run because we skip exception
    output_list = extractor.predict(
        [mini_wsi_svs],
        mode="wsi",
        ioconfig=ioconfig,
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    wsi_0_root_path = output_list[0][1]
    positions = np.load(f"{wsi_0_root_path}.position.npy")
    features = np.load(f"{wsi_0_root_path}.features.0.npy")

    reader = WSIReader.open(mini_wsi_svs)
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
    # ! else the output values will not exactly be the same (still < 1.0e-4
    # ! of epsilon though)
    assert np.mean(np.abs(features[:4] - _features)) < 1.0e-4

    _rm_dir(save_dir)
