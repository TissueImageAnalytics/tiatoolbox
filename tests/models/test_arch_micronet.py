"""Unit test package for MicroNet."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from tiatoolbox.models import MicroNet, NucleusInstanceSegmentor
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
    output_ = model.postproc(list(output[0]))
    assert output_[0]["task_type"] == "nuclei_segmentation"
    assert np.max(np.unique(output_[0]["predictions"])) == 46
    assert len(output_[0]["info_dict"]["centroid"]) == 27
    assert len(output_[0]["info_dict"]["contours"]) == 27

    # For test coverage pass probability map with
    # no cell segmentation instance
    output_ = model.postproc(np.zeros((1, 252, 252, 2)))
    assert output_[0]["task_type"] == "nuclei_segmentation"
    assert np.max(np.unique(output_[0]["predictions"])) == 0
    assert len(output_[0]["info_dict"]["centroid"]) == 0

    Path(weights_path).unlink()


def test_value_error() -> None:
    """Test to generate value error is num_output_channels < 2."""
    with pytest.raises(ValueError, match="Number of classes should be >=2"):
        _ = MicroNet(num_output_channels=1)


@pytest.mark.skipif(
    toolbox_env.running_on_ci() or not ON_GPU,
    reason="Local test on machine with GPU.",
)
def test_micronet_output(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Test the output of MicroNet."""
    svs_1_small = Path(remote_sample("svs-1-small"))
    micronet_output = Path(remote_sample("micronet-output"))
    model = "micronet-consep"
    batch_size = 64
    num_workers = 0

    ninst_seg = NucleusInstanceSegmentor(
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    output = ninst_seg.run(
        images=[
            svs_1_small,
        ],
        save_dir=track_tmp_path / "output",
        patch_mode=False,
        verbose=True,
        device=select_device(on_gpu=ON_GPU),
        return_predictions=(True,),
        return_probabilities=True,
        output_type="zarr",
    )

    output = zarr.open(output[svs_1_small], mode="r")
    output_on_server = np.load(str(micronet_output))
    output_on_server = np.round(output_on_server, decimals=3)
    new_output = np.round(
        output["probabilities"][0][1000:2000:2, 2000:3000:2, :], decimals=3
    )
    diff = new_output - output_on_server
    assert diff.mean() < 1e-5
