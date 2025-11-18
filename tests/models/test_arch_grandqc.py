"""Unit test package for GrandQC Tissue Model."""

import numpy as np
import torch

from tiatoolbox.models.architecture import (
    fetch_pretrained_weights,
    get_pretrained_model,
)
from tiatoolbox.models.architecture.grandqc import GrandQCModel
from tiatoolbox.models.engine.io_config import IOSegmentorConfig
from tiatoolbox.utils.misc import select_device
from tiatoolbox.wsicore.wsireader import VirtualWSIReader

ON_GPU = False


def test_functional_grandqc() -> None:
    """Test for GrandQC model."""
    # test fetch pretrained weights
    pretrained_weights = fetch_pretrained_weights("grandqc_tissue_detection_mpp10")
    assert pretrained_weights is not None

    # test creation
    model = GrandQCModel(num_output_channels=2)
    assert model is not None

    # load pretrained weights
    pretrained = torch.load(pretrained_weights, map_location="cpu")
    model.load_state_dict(pretrained)

    # test get pretrained model
    model, ioconfig = get_pretrained_model("grandqc_tissue_detection_mpp10")
    assert isinstance(model, GrandQCModel)
    assert isinstance(ioconfig, IOSegmentorConfig)
    assert model.num_output_channels == 2
    assert model.decoder_channels == (256, 128, 64, 32, 16)

    # test inference
    generator = np.random.default_rng(1337)
    test_image = generator.integers(0, 256, size=(2048, 2048, 3), dtype=np.uint8)
    reader = VirtualWSIReader.open(test_image)
    read_kwargs = {"resolution": 0, "units": "level", "coord_space": "resolution"}
    batch = np.array(
        [
            reader.read_bounds((0, 0, 512, 512), **read_kwargs),
            reader.read_bounds((512, 512, 1024, 1024), **read_kwargs),
        ],
    )
    batch = torch.from_numpy(batch)
    output = model.infer_batch(model, batch, device=select_device(on_gpu=ON_GPU))
    assert output.shape == (2, 512, 512, 2)


def test_grandqc_preproc_postproc() -> None:
    """Test GrandQC preproc and postproc functions."""
    model = GrandQCModel(num_output_channels=2)

    generator = np.random.default_rng(1337)
    # test preproc
    dummy_image = generator.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    preproc_image = model.preproc(dummy_image)
    assert preproc_image.shape == dummy_image.shape
    assert preproc_image.dtype == np.float64

    # test postproc
    dummy_output = generator.random(size=(512, 512, 2), dtype=np.float32)
    postproc_image = model.postproc(dummy_output)
    assert postproc_image.shape == (512, 512)
    assert postproc_image.dtype == np.int64
