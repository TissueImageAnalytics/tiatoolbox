"""Unit test package for HoVerNet+."""

import multiprocessing
import pathlib
import shutil

import numpy as np
import pytest

from tiatoolbox.models import SemanticSegmentor
from tiatoolbox.utils import env_detection as toolbox_env

ON_GPU = toolbox_env.has_gpu()
# The batch size value here is based on two TitanXP, each with 12GB
BATCH_SIZE = 1 if not ON_GPU else 16
try:
    NUM_POSTPROC_WORKERS = multiprocessing.cpu_count()
except NotImplementedError:
    NUM_POSTPROC_WORKERS = 2

# ----------------------------------------------------


def _rm_dir(path):
    """Helper func to remove directory."""
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.skipif(
    toolbox_env.running_on_ci() or not toolbox_env.has_gpu(),
    reason="Local test on machine with GPU.",
)
def test_functionality_local(remote_sample, tmp_path):
    """Local functionality test for multi task segmentor. Currently only
    testing HoVerNet+ with semantic segmentor.
    """
    root_save_dir = pathlib.Path(tmp_path)
    mini_wsi_svs = pathlib.Path(remote_sample("svs-1-small"))

    save_dir = f"{root_save_dir}/semantic/"
    _rm_dir(save_dir)
    semantic_segmentor = SemanticSegmentor(
        pretrained_model="hovernetplus-oed",
        batch_size=BATCH_SIZE,
        num_postproc_workers=NUM_POSTPROC_WORKERS,
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
    assert len(inst_dict) > 0, "Must have some nuclei."
    assert len(layer_dict) > 0, "Must have some layers."
    assert (
        inst_map.shape == layer_map.shape
    ), "Output instance and layer maps must be same shape"
    _rm_dir(tmp_path)
