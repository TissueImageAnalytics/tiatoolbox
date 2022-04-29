"""Unit test package for HoVerNet+."""

import pathlib
import shutil

import joblib
import numpy as np
import pytest

from tiatoolbox.models import IOSegmentorConfig, MultiTaskSegmentor
from tiatoolbox.utils import env_detection as toolbox_env

ON_GPU = ON_GPU = toolbox_env.has_gpu()
# The value is based on 2 TitanXP each with 12GB
BATCH_SIZE = 1 if not ON_GPU else 16
NUM_POSTPROC_WORKERS = 2 if not toolbox_env.running_on_travis() else 8

# ----------------------------------------------------


def _rm_dir(path):
    """Helper func to remove directory."""
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.skipif(
    toolbox_env.running_on_travis() or not toolbox_env.has_gpu(),
    reason="Local test on machine with GPU.",
)
def test_functionality_local(remote_sample, tmp_path):
    """Local functionality test for multi task segmentor."""
    root_save_dir = pathlib.Path(tmp_path)
    mini_wsi_svs = pathlib.Path(remote_sample("svs-1-small"))

    save_dir = f"{root_save_dir}/semantic/"
    _rm_dir(save_dir)
    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="hovernetplus-oed",
        batch_size=BATCH_SIZE,
        num_postproc_workers=NUM_POSTPROC_WORKERS,
    )
    output = multi_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        on_gpu=True,
        crash_on_exception=True,
        save_dir=save_dir,
    )

    inst_dict = joblib.load(f"{output[0][1]}.0.dat")
    layer_map = np.load(f"{output[0][1]}.1.npy")

    assert len(inst_dict) > 0 and layer_map is not None, "Must have some nuclei/layers."
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

    inst_dict = joblib.load(f"{output[0][1]}.0.dat")
    layer_map = np.load(f"{output[0][1]}.1.npy")

    assert len(inst_dict) > 0 and layer_map is not None, "Must have some nuclei/layers."
    assert (
        layer_map.shape == required_dims
    ), "Output layer map dimensions must be same as the expected output shape"
    _rm_dir(tmp_path)
