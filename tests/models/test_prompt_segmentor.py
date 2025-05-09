"""Unit test package for Prompt Segmentor."""

from __future__ import annotations

# ! The garbage collector
import multiprocessing
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from tiatoolbox.models import PromptSegmentor
from tiatoolbox.models.architecture.sam import SAM
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
)
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils import imwrite
from tiatoolbox.utils.misc import select_device
from tiatoolbox.wsicore.wsireader import WSIReader

ON_GPU = toolbox_env.has_gpu()
BATCH_SIZE = 1 if not ON_GPU else 2
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
    mini_wsi_svs = Path(remote_sample("patch-extraction-vf"))
    reader = WSIReader.open(mini_wsi_svs, resolution)
    thumb = reader.slide_thumbnail(resolution=resolution, units="mpp")
    thumb = thumb[63:191, 750:878, :]
    mini_wsi_jpg = f"{tmp_path}/mini_svs.jpg"
    imwrite(mini_wsi_jpg, thumb)

    # preemptive clean up
    shutil.rmtree(save_dir, ignore_errors=True)

    model = SAM()

    # test engine setup

    _ = PromptSegmentor(None, BATCH_SIZE, NUM_LOADER_WORKERS)

    prompt_segmentor = PromptSegmentor(model, BATCH_SIZE, NUM_LOADER_WORKERS)

    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {"units": "mpp", "resolution": 4.0},
        ],
        output_resolutions=[{"units": "mpp", "resolution": 4.0}],
        patch_input_shape=[512, 512],
        patch_output_shape=[512, 512],
        stride_shape=[512, 512],
    )

    # test inference

    points = np.array([[[64, 64]], [[64, 64]]])  # Point on nuclei

    # Run on tile mode with multi-prompt
    # Test running with multiple images
    shutil.rmtree(save_dir, ignore_errors=True)
    output_list = prompt_segmentor.predict(
        [mini_wsi_jpg, mini_wsi_jpg],
        mode="tile",
        multi_prompt=True,
        device=select_device(on_gpu=ON_GPU),
        point_coords=points,
        ioconfig=ioconfig,
        crash_on_exception=False,
        save_dir=save_dir,
    )

    pred_1 = np.load(output_list[0][1] + "/0.raw.0.npy")
    pred_2 = np.load(output_list[1][1] + "/0.raw.0.npy")
    assert len(output_list) == 2
    assert np.sum(pred_1 - pred_2) == 0

    points = np.array([[[64, 64], [100, 40], [100, 70]]])  # Points on nuclei
    boxes = np.array([[[10, 10, 50, 50], [80, 80, 110, 110]]])  # Boxes on nuclei

    # Run on tile mode with single-prompt
    # Also tests boxes
    shutil.rmtree(save_dir, ignore_errors=True)
    output_list = prompt_segmentor.predict(
        [mini_wsi_jpg],
        mode="tile",
        multi_prompt=False,
        device=select_device(on_gpu=ON_GPU),
        point_coords=points,
        box_coords=boxes,
        ioconfig=ioconfig,
        crash_on_exception=False,
        save_dir=save_dir,
    )

    total_prompts = points.shape[1] + boxes.shape[1]
    preds = [
        np.load(output_list[0][1] + f"/{i}.raw.0.npy") for i in range(total_prompts)
    ]

    assert len(output_list) == 1
    assert len(preds) == total_prompts

    # Generate mask
    mask = np.zeros((thumb.shape[0], thumb.shape[1]), dtype=np.uint8)
    mask[32:120, 32:120] = 1
    mini_wsi_msk = f"{tmp_path}/mini_svs_mask.jpg"
    imwrite(mini_wsi_msk, mask)

    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {"units": "baseline", "resolution": 1.0},
        ],
        output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        patch_input_shape=[512, 512],
        patch_output_shape=[512, 512],
        stride_shape=[512, 512],
        save_resolution={"units": "baseline", "resolution": 1.0},
    )

    # Only point within mask should generate a segmentation
    points = np.array([[[64, 64], [100, 40]]])
    save_dir = tmp_path / "dump"

    # Run on wsi mode with multi-prompt
    # Also tests masks
    shutil.rmtree(save_dir, ignore_errors=True)
    output_list = prompt_segmentor.predict(
        [mini_wsi_jpg],
        masks=[mini_wsi_msk],
        mode="wsi",
        multi_prompt=True,
        device=select_device(on_gpu=ON_GPU),
        point_coords=points,
        ioconfig=ioconfig,
        crash_on_exception=False,
        save_dir=save_dir,
    )

    # Check if db exists
    assert Path(output_list[0][1] + ".0.db").exists()

    # Run on wsi mode with single-prompt
    shutil.rmtree(save_dir, ignore_errors=True)
    output_list = prompt_segmentor.predict(
        [mini_wsi_jpg],
        mode="wsi",
        multi_prompt=False,
        device=select_device(on_gpu=ON_GPU),
        point_coords=points,
        ioconfig=ioconfig,
        crash_on_exception=False,
        save_dir=save_dir,
    )

    # Check if db exists
    assert Path(output_list[0][1] + ".0.db").exists()


def test_crash_segmentor(remote_sample: Callable, tmp_path: Path) -> None:
    """Functional crash tests for segmentor."""
    # # convert to pathlib Path to prevent wsireader complaint
    mini_wsi_svs = Path(remote_sample("wsi2_4k_4k_svs"))
    mini_wsi_msk = Path(remote_sample("wsi2_4k_4k_msk"))

    save_dir = tmp_path / "test_crash_segmentor"
    prompt_segmentor = PromptSegmentor(batch_size=BATCH_SIZE)

    # * test basic crash
    with pytest.raises(TypeError, match=r".*`mask_reader`.*"):
        prompt_segmentor.filter_coordinates(mini_wsi_msk, np.array(["a", "b", "c"]))
    with pytest.raises(TypeError, match=r".*`mask_reader`.*"):
        prompt_segmentor.get_mask_bounds(mini_wsi_msk)
    with pytest.raises(TypeError, match=r".*mask_reader.*"):
        prompt_segmentor.clip_coordinates(mini_wsi_msk, np.array(["a", "b", "c"]))

    with pytest.raises(ValueError, match=r".*ndarray.*integer.*"):
        prompt_segmentor.filter_coordinates(
            WSIReader.open(mini_wsi_msk),
            np.array([1.0, 2.0]),
        )
    with pytest.raises(ValueError, match=r".*ndarray.*integer.*"):
        prompt_segmentor.clip_coordinates(
            WSIReader.open(mini_wsi_msk),
            np.array([1.0, 2.0]),
        )
    prompt_segmentor.get_reader(mini_wsi_svs, None, "wsi", auto_get_mask=True)
    with pytest.raises(ValueError, match=r".*must be a valid file path.*"):
        prompt_segmentor.get_reader(
            mini_wsi_msk,
            "not_exist",
            "wsi",
            auto_get_mask=True,
        )

    shutil.rmtree(save_dir, ignore_errors=True)  # default output dir test
    with pytest.raises(ValueError, match=r".*valid mode.*"):
        prompt_segmentor.predict([], mode="abc")

    shutil.rmtree(save_dir, ignore_errors=True)
