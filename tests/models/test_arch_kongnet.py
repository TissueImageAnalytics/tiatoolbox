"""Unit test package for KongNet Model."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models.architecture import (
    fetch_pretrained_weights,
    get_pretrained_model,
)
from tiatoolbox.models.architecture.kongnet import KongNet
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.wsicore.wsireader import VirtualWSIReader

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def test_KongNet_Modeling() -> None:
    """Test for KongNet model."""

    # test creation
    model = KongNet(
        num_heads=6, 
        num_channels_per_head=[3,3,3,3,3,3],
        target_channels=[2, 5, 8, 11, 14, 17],
        min_distance=5,
        threshold_abs=0.5,
        wide_decoder=False,
    )

    # ckp_path = "/media/u1910100/data/Monkey/conic_models/efficientnetv2_l_eq_loss/KongNet_CoNIC_1.pth"
    # state_dict = torch.load(ckp_path, map_location=device)["model"]
    # model.load_state_dict(state_dict)
    model = model.to(device)
    # assert model is not None

    model.eval()
    with torch.no_grad():
        input_tensor = torch.randn(1, 3, 256, 256).to(device)
        output = model(input_tensor)
        assert output.shape == (1, 6, 256, 256)

        batch_tensor = torch.randn(4, 3, 256, 256).to(device)
        output = model(batch_tensor)
        assert output.shape == (4, 6, 256, 256)
