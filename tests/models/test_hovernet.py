"""Unit test package for HoVerNet."""

from typing import Callable

import numpy as np
import pytest
import torch
from torch import nn

from tiatoolbox.models import HoVerNet
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.architecture.hovernet import (
    DenseBlock,
    ResidualBlock,
    TFSamepaddingLayer,
)
from tiatoolbox.utils.misc import select_device
from tiatoolbox.wsicore.wsireader import WSIReader


def test_functionality(remote_sample: Callable) -> None:
    """Functionality test."""
    sample_wsi = str(remote_sample("wsi1_2k_2k_svs"))
    reader = WSIReader.open(sample_wsi)

    # * test fast mode (architecture used in PanNuke paper)
    patch = reader.read_bounds(
        (0, 0, 256, 256),
        resolution=0.25,
        units="mpp",
        coord_space="resolution",
    )
    batch = torch.from_numpy(patch)[None]
    model = HoVerNet(num_types=6, mode="fast")
    weights_path = fetch_pretrained_weights("hovernet_fast-pannuke")
    pretrained = torch.load(weights_path)
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, device=select_device(on_gpu=False))
    output = [v[0] for v in output]
    output = model.postproc(output)
    assert len(output[1]) > 0, "Must have some nuclei."

    # * test fast mode (architecture used for MoNuSAC data)
    patch = reader.read_bounds(
        (0, 0, 256, 256),
        resolution=0.25,
        units="mpp",
        coord_space="resolution",
    )
    batch = torch.from_numpy(patch)[None]
    model = HoVerNet(num_types=5, mode="fast")
    weights_path = fetch_pretrained_weights("hovernet_fast-monusac")
    pretrained = torch.load(weights_path)
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, device=select_device(on_gpu=False))
    output = [v[0] for v in output]
    output = model.postproc(output)
    assert len(output[1]) > 0, "Must have some nuclei."

    # * test original mode on CoNSeP dataset (architecture used in HoVerNet paper)
    patch = reader.read_bounds(
        (0, 0, 270, 270),
        resolution=0.25,
        units="mpp",
        coord_space="resolution",
    )
    batch = torch.from_numpy(patch)[None]
    model = HoVerNet(num_types=5, mode="original")
    weights_path = fetch_pretrained_weights("hovernet_original-consep")
    pretrained = torch.load(weights_path)
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, device=select_device(on_gpu=False))
    output = [v[0] for v in output]
    output = model.postproc(output)
    assert len(output[1]) > 0, "Must have some nuclei."

    # * test original mode on Kumar dataset (architecture used in HoVerNet paper)
    patch = reader.read_bounds(
        (0, 0, 270, 270),
        resolution=0.25,
        units="mpp",
        coord_space="resolution",
    )
    batch = torch.from_numpy(patch)[None]
    model = HoVerNet(num_types=None, mode="original")
    weights_path = fetch_pretrained_weights("hovernet_original-kumar")
    pretrained = torch.load(weights_path)
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, device=select_device(on_gpu=False))
    output = [v[0] for v in output]
    output = model.postproc(output)
    assert len(output[1]) > 0, "Must have some nuclei."

    # test crash when providing exotic mode
    with pytest.raises(ValueError, match=r".*Invalid mode.*"):
        model = HoVerNet(num_types=None, mode="super")


def test_unit_blocks() -> None:
    """Test for blocks within HoVerNet."""
    # padding
    model = nn.Sequential(TFSamepaddingLayer(7, 1), nn.Conv2d(3, 3, 7, 1, padding=0))
    sample = torch.rand((1, 3, 14, 14), dtype=torch.float32)
    output = model(sample)
    assert np.sum(output.shape - np.array([1, 3, 14, 14])) == 0, f"{output.shape}"

    # padding with stride and odd shape
    model = nn.Sequential(TFSamepaddingLayer(7, 2), nn.Conv2d(3, 3, 7, 2, padding=0))
    sample = torch.rand((1, 3, 15, 15), dtype=torch.float32)
    output = model(sample)
    assert np.sum(output.shape - np.array([1, 3, 8, 8])) == 0, f"{output.shape}"

    # *
    sample = torch.rand((1, 16, 15, 15), dtype=torch.float32)

    block = ResidualBlock(16, [1, 3, 1], [16, 16, 16], 3)

    assert block.shortcut is None
    output = block(sample)
    assert np.sum(output.shape - np.array([1, 16, 15, 15])) == 0, f"{output.shape}"

    block = ResidualBlock(16, [1, 3, 1], [16, 16, 32], 3)
    assert block.shortcut is not None
    output = block(sample)
    assert np.sum(output.shape - np.array([1, 32, 15, 15])) == 0, f"{output.shape}"

    block = DenseBlock(16, [1, 3], [16, 16], 3)
    output = block(sample)
    assert output.shape[1] == 16 * 4, f"{output.shape}"

    # test crash when providing exotic mode
    with pytest.raises(ValueError, match=r".*Unbalance Unit Info.*"):
        _ = DenseBlock(16, [1, 3, 1], [16, 16], 3)
    with pytest.raises(ValueError, match=r".*Unbalance Unit Info.*"):
        _ = ResidualBlock(16, [1, 3, 1], [16, 16], 3)
