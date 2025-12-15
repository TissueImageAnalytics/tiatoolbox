"""Unit test package for SCCNN."""

from collections.abc import Callable

import numpy as np
import torch

from tiatoolbox.models import SCCNN
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.utils import env_detection
from tiatoolbox.utils.misc import select_device
from tiatoolbox.wsicore.wsireader import WSIReader


def _load_sccnn(name: str) -> SCCNN:
    """Loads SCCNN model with specified weights."""
    model = SCCNN()
    weights_path = fetch_pretrained_weights(name)
    map_location = select_device(on_gpu=env_detection.has_gpu())
    pretrained = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(pretrained)
    model.to(map_location)
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
    model = _load_sccnn(name="sccnn-crchisto")
    patch = model.preproc(patch)
    batch = torch.from_numpy(patch)[None]
    output = model.infer_batch(
        model,
        batch,
        device=select_device(on_gpu=env_detection.has_gpu()),
    )
    output = model.postproc(output[0])
    ys, xs, _ = np.nonzero(output)

    np.testing.assert_array_equal(xs, np.array([8]))
    np.testing.assert_array_equal(ys, np.array([7]))

    model = _load_sccnn(name="sccnn-conic")
    output = model.infer_batch(
        model,
        batch,
        device=select_device(on_gpu=env_detection.has_gpu()),
    )
    block_info = {
        0: {
            "array-location": [[0, 31], [0, 31]],
        }
    }
    output = model.postproc(output[0], block_info=block_info)
    ys, xs, _ = np.nonzero(output)
    np.testing.assert_array_equal(xs, np.array([7]))
    np.testing.assert_array_equal(ys, np.array([8]))

    model = _load_sccnn(name="sccnn-conic")
    output = model.infer_batch(
        model,
        batch,
        device=select_device(on_gpu=env_detection.has_gpu()),
    )
    block_info = {
        0: {
            "array-location": [
                [0, 1],
                [0, 1],
            ],  # dummy block to test no valid detections
        }
    }
    output = model.postproc(output[0], block_info=block_info)
    ys, xs, _ = np.nonzero(output)
    np.testing.assert_array_equal(xs, np.array([]))
    np.testing.assert_array_equal(ys, np.array([]))
