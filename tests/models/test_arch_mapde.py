"""Unit test package for SCCNN."""
import numpy as np
import torch

from tiatoolbox import utils
from tiatoolbox.models import MapDe
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.wsicore.wsireader import WSIReader


def _load_mapde(name):
    """Loads MapDe model with specified weights."""
    model = MapDe()
    weights_path = fetch_pretrained_weights(name)
    map_location = utils.misc.select_device(utils.env_detection.has_gpu())
    pretrained = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(pretrained)

    return model


def test_functionality(remote_sample):
    """Functionality test for MapDe.

    Tests the functionality of MapDe model for inference at the patch level.

    """
    sample_wsi = str(remote_sample("wsi1_2k_2k_svs"))
    reader = WSIReader.open(sample_wsi)

    # * test fast mode (architecture used in PanNuke paper)
    patch = reader.read_bounds(
        (0, 0, 252, 252), resolution=0.50, units="mpp", coord_space="resolution"
    )

    model = _load_mapde(name="mapde-crchisto")
    patch = model.preproc(patch)
    batch = torch.from_numpy(patch)[None]
    output = model.infer_batch(model, batch, on_gpu=False)
    output = model.postproc(output[0])
    assert np.all(output[0:2] == [[99, 178], [64, 218]])

    model = _load_mapde(name="mapde-conic")
    output = model.infer_batch(model, batch, on_gpu=False)
    output = model.postproc(output[0])
    assert np.all(output[0:2] == [[19, 171], [53, 89]])
