"""Unit test package for Prompt Segmentor."""

from __future__ import annotations

# ! The garbage collector
import multiprocessing
import shutil
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from tiatoolbox.models import PromptSegmentor
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.architecture.sam import SAM
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
)
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils import imread, imwrite
from tiatoolbox.utils.misc import select_device
from tiatoolbox.wsicore.wsireader import WSIReader

ON_GPU = toolbox_env.has_gpu()
# The value is based on 2 TitanXP each with 12GB
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

    # Remove before commit
    visualize_masks([imread(mini_wsi_jpg)], [preds], None)

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


# ! Remove before commit
def visualize_masks(
    images: list, masks_list: list, metadata_list: list | None = None
) -> None:
    """Visualizes masks on the given image."""
    for i, image in enumerate(images):
        for j, mask in enumerate(masks_list[i]):
            nuclei_type = metadata_list[i][j] if metadata_list else "Nuclei"
            plt.imshow(image / 255.0)
            plt.savefig("image_{i}.png", bbox_inches="tight", dpi=300)
            plt.imshow(mask, alpha=0.5, cmap="jet")
            plt.axis("off")
            plt.title(f"Nuclei Type: {nuclei_type}" + f"Image: {i}")
            plt.show()
            # save the image
            plt.savefig(f"image_{i}_mask_{j}.png", bbox_inches="tight", dpi=300)
