"""Unit test package for Prompt Segmentor."""

from __future__ import annotations

# ! The garbage collector
import multiprocessing
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from tiatoolbox.models import PromptSegmentor
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.architecture.sam import SAM
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
)
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils import imwrite
from tiatoolbox.utils.misc import select_device
from tiatoolbox.wsicore.wsireader import WSIReader

ON_GPU = toolbox_env.has_gpu()
# The value is based on 2 TitanXP each with 12GB
BATCH_SIZE = 1 if not ON_GPU else 16
try:
    NUM_LOADER_WORKERS = multiprocessing.cpu_count()
except NotImplementedError:
    NUM_LOADER_WORKERS = 2


def test_functional_segmentor(
    remote_sample: Callable,
    tmp_path: Path,
) -> None:
    """Functional test for segmentor."""
    save_dir = tmp_path / "dump"
    # # convert to pathlib Path to prevent wsireader complaint
    resolution = 2.0
    mini_wsi_svs = Path(remote_sample("wsi4_1k_1k_svs"))
    reader = WSIReader.open(mini_wsi_svs)
    thumb = reader.slide_thumbnail(resolution=resolution, units="mpp")
    mini_wsi_jpg = f"{tmp_path}/mini_svs.jpg"
    imwrite(mini_wsi_jpg, thumb)
    mini_wsi_msk = f"{tmp_path}/mini_mask.jpg"
    imwrite(mini_wsi_msk, (thumb > 0).astype(np.uint8))

    # preemptive clean up
    shutil.rmtree(save_dir, ignore_errors=True)

    model = SAM()
    pretrained = fetch_pretrained_weights("segment_anything-base")
    model.load_state_dict(torch.load(pretrained, map_location="cpu"))

    prompt_segmentor = PromptSegmentor(model, BATCH_SIZE, NUM_LOADER_WORKERS)

    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {"units": "baseline", "resolution": 1.0},
        ],
        output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        patch_input_shape=[512, 512],
        patch_output_shape=[512, 512],
    )

    points = np.array([[[64, 64], [100, 100], [64, 100], [100, 64]]])  # Random points

    # Run on tile mode with multi-prompt
    shutil.rmtree(save_dir, ignore_errors=True)
    output_list = prompt_segmentor.predict(
        [mini_wsi_jpg],
        mode="tile",
        multi_prompt=True,
        device=select_device(on_gpu=ON_GPU),
        point_coords=points,
        ioconfig=ioconfig,
        crash_on_exception=False,
        save_dir=save_dir,
    )

    assert len(output_list) == 1

    # Run on tile mode with single-prompt
    shutil.rmtree(save_dir, ignore_errors=True)
    output_list = prompt_segmentor.predict(
        [mini_wsi_jpg],
        mode="tile",
        multi_prompt=False,
        device=select_device(on_gpu=ON_GPU),
        point_coords=points,
        ioconfig=ioconfig,
        crash_on_exception=False,
        save_dir=save_dir,
    )

    pred_1 = np.load(output_list[0][1] + ".raw.0.npy")
    pred_2 = np.load(output_list[1][1] + ".raw.0.npy")
    assert len(output_list) == 4
    assert np.sum(pred_1 - pred_2) == 0
    # due to overlapping merge and division, will not be
    # exactly 1, but should be approximately so
    assert np.sum((pred_1 - 1) > 1.0e-6) == 0
    shutil.rmtree(save_dir, ignore_errors=True)

    # * test running with mask and svs
    # * also test merging prediction at designated resolution
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": resolution}],
        output_resolutions=[{"units": "mpp", "resolution": resolution}],
        save_resolution={"units": "mpp", "resolution": resolution},
        patch_input_shape=[512, 512],
        patch_output_shape=[256, 256],
        stride_shape=[512, 512],
    )
    shutil.rmtree(save_dir, ignore_errors=True)
    output_list = prompt_segmentor.predict(
        [mini_wsi_svs],
        masks=[mini_wsi_msk],
        mode="wsi",
        device=select_device(on_gpu=ON_GPU),
        ioconfig=ioconfig,
        crash_on_exception=True,
        save_dir=f"{save_dir}/raw/",
    )
    reader = WSIReader.open(mini_wsi_svs)
    expected_shape = reader.slide_dimensions(**ioconfig.save_resolution)
    expected_shape = np.array(expected_shape)[::-1]  # to YX
    pred_1 = np.load(output_list[0][1] + ".raw.0.npy")
    saved_shape = np.array(pred_1.shape[:2])
    assert np.sum(expected_shape - saved_shape) == 0
    assert np.sum((pred_1 - 1) > 1.0e-6) == 0
    shutil.rmtree(save_dir, ignore_errors=True)
