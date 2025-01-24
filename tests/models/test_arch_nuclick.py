"""Unit test package for NuClick."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch

from tiatoolbox.models import NuClick
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.utils import imread
from tiatoolbox.utils.misc import select_device

ON_GPU = False

# Test pretrained Model =============================


def test_functional_nuclick(
    remote_sample: Callable,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test for NuClick."""
    # convert to pathlib Path to prevent wsireader complaint
    tile_path = Path(remote_sample("patch-extraction-vf"))
    img = imread(tile_path)

    weights_path = fetch_pretrained_weights("nuclick_original-pannuke")

    # test creation
    _ = NuClick(num_input_channels=5, num_output_channels=1)

    # test inference
    # create image patch, inclusion and exclusion maps
    patch = img[63:191, 750:878, :]
    inclusion_map = np.zeros((128, 128))
    inclusion_map[64, 64] = 1

    exclusion_map = np.zeros((128, 128))
    exclusion_map[68, 82] = 1
    exclusion_map[72, 102] = 1
    exclusion_map[52, 48] = 1

    patch = np.float32(patch) / 255.0
    patch = np.moveaxis(patch, -1, 0)
    batch = np.concatenate(
        (patch, inclusion_map[np.newaxis, ...], exclusion_map[np.newaxis, ...]),
        axis=0,
    )

    batch = torch.from_numpy(batch[np.newaxis, ...])

    model = NuClick(num_input_channels=5, num_output_channels=1)
    pretrained = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, device=select_device(on_gpu=ON_GPU))
    postproc_masks = model.postproc(
        output,
        do_reconstruction=True,
        nuc_points=inclusion_map[np.newaxis, ...],
    )

    gt_path = Path(remote_sample("nuclick-output"))
    gt_mask = np.load(str(gt_path))

    assert (
        np.count_nonzero(postproc_masks * gt_mask) / np.count_nonzero(gt_mask) > 0.999
    )

    # test post-processing without reconstruction
    _ = model.postproc(output)

    # test failed reconstruction in post-processing
    inclusion_map = np.zeros((128, 128))
    inclusion_map[0, 0] = 1
    _ = model.postproc(
        output,
        do_reconstruction=True,
        nuc_points=inclusion_map[np.newaxis, ...],
    )
    assert "Nuclei reconstruction was not done" in caplog.text
