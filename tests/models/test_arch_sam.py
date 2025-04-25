"""Unit test package for SAM."""

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from tiatoolbox.models.architecture import fetch_pretrained_weights
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

    weights_path = fetch_pretrained_weights("segment_anything-base")

    # test creation

    model = SAM()
    pretrained = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(pretrained)

    # test inference

    # create image patch and prompts
    patch = np.expand_dims(img[63:191, 750:878, :], axis=0)
    patch = model.preproc(patch)  # pre-process the image

    points = np.array([[[64, 64]]], dtype=np.int32)
    boxes = np.array([[64, 64, 128, 128]], dtype=np.int32)

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
