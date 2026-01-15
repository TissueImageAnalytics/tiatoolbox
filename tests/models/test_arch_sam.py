"""Unit test package for SAM."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from tiatoolbox.models.architecture.sam import SAM
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils import imread
from tiatoolbox.utils.misc import select_device

ON_GPU = toolbox_env.has_gpu()

# Test pretrained Model =============================


def test_functional_sam(
    remote_sample: Callable,
) -> None:
    """Test for SAM."""
    # convert to pathlib Path to prevent wsireader complaint
    tile_path = Path(remote_sample("patch-extraction-vf"))
    img = imread(tile_path)

    # test creation

    model = SAM(device=select_device(on_gpu=ON_GPU))

    # create image patch and prompts
    patch = img[63:191, 750:878, :]

    points = np.array([[[64, 64]]])
    boxes = np.array([[[64, 64, 128, 128]]])

    # test preproc
    tensor = torch.from_numpy(img)
    patch = np.expand_dims(model.preproc(tensor), axis=0)
    patch = model.preproc(patch)

    # test inference

    mask_output, score_output = model.infer_batch(
        model, patch, points, device=select_device(on_gpu=ON_GPU)
    )

    assert mask_output is not None, "Output should not be None"
    assert len(mask_output) > 0, "Output should have at least one element"
    assert len(score_output) > 0, "Output should have at least one element"

    mask_output, score_output = model.infer_batch(
        model, patch, box_coords=boxes, device=select_device(on_gpu=ON_GPU)
    )

    assert len(mask_output) > 0, "Output should have at least one element"
    assert len(score_output) > 0, "Output should have at least one element"

    mask_output, score_output = model.infer_batch(
        model, patch, device=select_device(on_gpu=ON_GPU)
    )

    assert mask_output is not None, "Output should not be None"
    assert len(mask_output) > 0, "Output should have at least one element"
    assert len(score_output) > 0, "Output should have at least one element"
