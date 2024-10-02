"""Unit test package for MicroNet."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch

from tiatoolbox.models import MicroNet, SemanticSegmentor
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.misc import select_device
from tiatoolbox.wsicore.wsireader import WSIReader

ON_GPU = toolbox_env.has_gpu()


def test_functionality(
    remote_sample: Callable,
) -> None:
    """Functionality test."""
    sample_wsi = remote_sample("wsi1_2k_2k_svs")
    reader = WSIReader.open(sample_wsi)

    # * test fast mode (architecture used in PanNuke paper)
    patch = reader.read_bounds(
        (0, 0, 252, 252),
        resolution=0.25,
        units="mpp",
        coord_space="resolution",
    )

    model = MicroNet()
    patch = model.preproc(patch)
    batch = torch.from_numpy(patch)[None]
    weights_path = fetch_pretrained_weights("micronet-consep")
    map_location = select_device(on_gpu=ON_GPU)
    model = model.to(map_location)
    pretrained = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, device=map_location)
    output, _ = model.postproc(output[0])
    assert np.max(np.unique(output)) == 46


def test_value_error() -> None:
    """Test to generate value error is num_output_channels < 2."""
    with pytest.raises(ValueError, match="Number of classes should be >=2"):
        _ = MicroNet(num_output_channels=1)


@pytest.mark.skipif(
    toolbox_env.running_on_ci() or not ON_GPU,
    reason="Local test on machine with GPU.",
)
def test_micronet_output(remote_sample: Callable, tmp_path: Path) -> None:
    """Test the output of MicroNet."""
    svs_1_small = Path(remote_sample("svs-1-small"))
    micronet_output = Path(remote_sample("micronet-output"))
    pretrained_model = "micronet-consep"
    batch_size = 5
    num_loader_workers = 0
    num_postproc_workers = 0

    predictor = SemanticSegmentor(
        pretrained_model=pretrained_model,
        batch_size=batch_size,
        num_loader_workers=num_loader_workers,
        num_postproc_workers=num_postproc_workers,
    )

    output = predictor.predict(
        imgs=[
            svs_1_small,
        ],
        save_dir=tmp_path / "output",
    )

    output = np.load(output[0][1] + ".raw.0.npy")
    output_on_server = np.load(str(micronet_output))
    output_on_server = np.round(output_on_server, decimals=3)
    new_output = np.round(output[500:1000, 1000:1500, :], decimals=3)
    diff = new_output - output_on_server
    assert diff.mean() < 1e-5
