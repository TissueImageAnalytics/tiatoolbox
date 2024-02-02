"""Unit test package for SCCNN."""

from typing import Callable

import numpy as np
import torch

from tiatoolbox.models import SCCNN
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.utils import env_detection
from tiatoolbox.utils.misc import select_device
from tiatoolbox.wsicore.wsireader import WSIReader


def _load_sccnn(name: str) -> torch.nn.Module:
    """Loads SCCNN model with specified weights."""
    model = SCCNN()
    weights_path = fetch_pretrained_weights(name)
    map_location = select_device(on_gpu=env_detection.has_gpu())
    pretrained = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(pretrained)

    return model


def test_functionality(remote_sample: Callable) -> None:
    """Functionality test for SCCNN.

    Test the functionality of SCCNN model for inference at the patch level.

    """
    sample_wsi = str(remote_sample("wsi1_2k_2k_svs"))
    reader = WSIReader.open(sample_wsi)

    # * test fast mode (architecture used in PanNuke paper)
    patch = reader.read_bounds(
        (30, 30, 61, 61),
        resolution=0.25,
        units="mpp",
        coord_space="resolution",
    )
    batch = torch.from_numpy(patch)[None]
    model = _load_sccnn(name="sccnn-crchisto")
    output = model.infer_batch(
        model,
        batch,
        device=select_device(on_gpu=env_detection.has_gpu()),
    )
    output = model.postproc(output[0])
    assert np.all(output == [[8, 7]])

    model = _load_sccnn(name="sccnn-conic")
    output = model.infer_batch(
        model,
        batch,
        device=select_device(on_gpu=env_detection.has_gpu()),
    )
    output = model.postproc(output[0])
    assert np.all(output == [[7, 8]])
