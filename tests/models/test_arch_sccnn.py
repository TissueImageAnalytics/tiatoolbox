"""Unit test package for SCCNN."""
import numpy as np
import torch

from tiatoolbox import utils
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.architecture.sccnn import SCCNN
from tiatoolbox.wsicore.wsireader import WSIReader


def _load_sccnn(tmp_path, name):
    """Loads SCCNN model with specified weights."""
    model = SCCNN()
    fetch_pretrained_weights(name, f"{tmp_path}/weights.pth")
    map_location = utils.misc.select_device(utils.env_detection.has_gpu())
    pretrained = torch.load(f"{tmp_path}/weights.pth", map_location=map_location)
    model.load_state_dict(pretrained)

    return model


def test_functionality(remote_sample, tmp_path):
    """Functionality test for SCCNN.

    Tests the functionality of SCCNN model for inference at the patch level.

    """
    tmp_path = str(tmp_path)
    sample_wsi = str(remote_sample("wsi1_2k_2k_svs"))
    reader = WSIReader.open(sample_wsi)

    # * test fast mode (architecture used in PanNuke paper)
    patch = reader.read_bounds(
        (30, 30, 61, 61), resolution=0.25, units="mpp", coord_space="resolution"
    )
    batch = torch.from_numpy(patch)[None]
    model = _load_sccnn(tmp_path=tmp_path, name="sccnn-crchisto")
    output = model.infer_batch(model, batch, on_gpu=False)
    output = model.postproc(output[0])
    assert np.all(output == [[8, 7]])

    model = _load_sccnn(tmp_path=tmp_path, name="sccnn-conic")
    output = model.infer_batch(model, batch, on_gpu=False)
    output = model.postproc(output[0])
    assert np.all(output == [[7, 8]])
