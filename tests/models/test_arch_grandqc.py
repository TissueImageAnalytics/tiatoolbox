"""Unit test package for GrandQC Tissue Model."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from tiatoolbox.models.architecture import (
    fetch_pretrained_weights,
    get_pretrained_model,
)
from tiatoolbox.models.architecture.grandqc import TissueDetectionModel
from tiatoolbox.models.engine.io_config import IOSegmentorConfig
from tiatoolbox.utils.misc import select_device
from tiatoolbox.wsicore.wsireader import WSIReader

ON_GPU = False


def test_functional_grandqc(remote_sample: Callable) -> None:
    """Test for GrandQC model."""
    # test fetch pretrained weights
    pretrained_weights = fetch_pretrained_weights("grandqc_tissue_detection_mpp10")
    assert pretrained_weights is not None

    # test creation
    model = TissueDetectionModel(num_input_channels=3, num_output_channels=2)
    assert model is not None

    # load pretrained weights
    pretrained = torch.load(pretrained_weights, map_location="cpu")
    model.load_state_dict(pretrained)

    # test get pretrained model
    model, ioconfig = get_pretrained_model("grandqc_tissue_detection_mpp10")
    assert isinstance(model, TissueDetectionModel)
    assert isinstance(ioconfig, IOSegmentorConfig)
    assert model.num_input_channels == 3
    assert model.num_output_channels == 2

    # test inference
    mini_wsi_svs = Path(remote_sample("wsi2_4k_4k_svs"))
    reader = WSIReader.open(mini_wsi_svs)
    read_kwargs = {"resolution": 10.0, "units": "mpp", "coord_space": "resolution"}
    batch = np.array(
        [
            reader.read_bounds((0, 0, 512, 512), **read_kwargs),
            reader.read_bounds((512, 512, 1024, 1024), **read_kwargs),
        ],
    )
    batch = torch.from_numpy(batch)
    output = model.infer_batch(model, batch, device=select_device(on_gpu=ON_GPU))
    assert output.shape == (2, 512, 512, 2)
