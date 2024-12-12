"""Unit test package for HoVerNet+."""

from typing import Callable

import torch

from tiatoolbox.models import HoVerNetPlus
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.utils import imread
from tiatoolbox.utils.misc import select_device
from tiatoolbox.utils.transforms import imresize


def test_functionality(remote_sample: Callable) -> None:
    """Functionality test."""
    sample_patch = str(remote_sample("stainnorm-source"))
    patch_pre = imread(sample_patch)
    patch_pre = imresize(patch_pre, scale_factor=0.5)
    patch = patch_pre[0:256, 0:256]
    batch = torch.from_numpy(patch)[None]

    # Test functionality with both nuclei and layer segmentation
    model = HoVerNetPlus(num_types=3, num_layers=5)
    # Test decoder as expected
    assert len(model.decoder["np"]) > 0, "Decoder must contain np branch."
    assert len(model.decoder["hv"]) > 0, "Decoder must contain hv branch."
    assert len(model.decoder["tp"]) > 0, "Decoder must contain tp branch."
    assert len(model.decoder["ls"]) > 0, "Decoder must contain ls branch."
    weights_path = fetch_pretrained_weights("hovernetplus-oed")
    pretrained = torch.load(weights_path)
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, device=select_device(on_gpu=False))
    assert len(output) == 4, "Must contain predictions for: np, hv, tp and ls branches."
    output = [v[0] for v in output]
    output = model.postproc(output)
    assert len(output[1]) > 0, "Must have some nuclei."
    assert len(output[3]) > 0, "Must have some layers."
