"""Unit test package for SCCNN."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from tiatoolbox.models import MapDe
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.misc import select_device
from tiatoolbox.wsicore.wsireader import WSIReader

ON_GPU = toolbox_env.has_gpu()


def _load_mapde(name: str) -> tuple[MapDe, str]:
    """Loads MapDe model with specified weights."""
    model = MapDe()
    weights_path = fetch_pretrained_weights(name)
    map_location = select_device(on_gpu=ON_GPU)
    pretrained = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(pretrained)
    model.to(map_location)
    return model, weights_path


def test_functionality(remote_sample: Callable) -> None:
    """Functionality test for MapDe.

    Test the functionality of MapDe model for inference at the patch level.

    """
    sample_wsi = str(remote_sample("wsi1_2k_2k_svs"))
    reader = WSIReader.open(sample_wsi)

    # * test fast mode (architecture used in PanNuke paper)
    patch = reader.read_bounds(
        (0, 0, 252, 252),
        resolution=0.50,
        units="mpp",
        coord_space="resolution",
    )

    model, weights_path = _load_mapde(name="mapde-conic")
    patch = model.preproc(patch)
    batch = torch.from_numpy(patch)[None]
    output = model.infer_batch(model, batch, device=select_device(on_gpu=ON_GPU))
    output = model.postproc(output[0])
    xs, ys, _, _ = NucleusDetector._centroid_maps_to_detection_records(output, None)

    np.testing.assert_array_equal(xs[0:2], np.array([242, 192]))
    np.testing.assert_array_equal(ys[0:2], np.array([10, 13]))
    Path(weights_path).unlink()


def test_multiclass_output() -> None:
    """Test the architecture for multi-class output."""
    multiclass_model = MapDe(num_input_channels=3, num_classes=3)
    test_input = torch.rand((1, 3, 252, 252))

    output = multiclass_model(test_input)
    assert output.shape == (1, 3, 252, 252)
