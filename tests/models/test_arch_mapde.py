"""Unit test package for SCCNN."""
import numpy as np
import torch

from tiatoolbox.models import MapDe
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.misc import select_device
from tiatoolbox.wsicore.wsireader import WSIReader

ON_GPU = toolbox_env.has_gpu()


def _load_mapde(tmp_path, name):
    """Loads MapDe model with specified weights."""
    model = MapDe()
    fetch_pretrained_weights(name, f"{tmp_path}/weights.pth")
    map_location = select_device(ON_GPU)
    pretrained = torch.load(f"{tmp_path}/weights.pth", map_location=map_location)
    model.load_state_dict(pretrained)

    return model


def test_functionality(remote_sample, tmp_path):
    """Functionality test for MapDe.

    Test the functionality of MapDe model for inference at the patch level.

    """
    tmp_path = str(tmp_path)
    sample_wsi = str(remote_sample("wsi1_2k_2k_svs"))
    reader = WSIReader.open(sample_wsi)

    # * test fast mode (architecture used in PanNuke paper)
    patch = reader.read_bounds(
        (0, 0, 252, 252),
        resolution=0.50,
        units="mpp",
        coord_space="resolution",
    )

    model = _load_mapde(tmp_path=tmp_path, name="mapde-conic")
    patch = model.preproc(patch)
    batch = torch.from_numpy(patch)[None]
    model = model.to(select_device(ON_GPU))
    output = model.infer_batch(model, batch, on_gpu=ON_GPU)
    output = model.postproc(output[0])
    assert np.all(output[0:2] == [[19, 171], [53, 89]])
