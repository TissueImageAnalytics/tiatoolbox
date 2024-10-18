"""Unit test package for vanilla CNN within toolbox."""

import numpy as np
import pytest
import torch

from tiatoolbox.models.architecture.vanilla import CNNModel, TimmModel
from tiatoolbox.utils.misc import model_to

ON_GPU = False
RNG = np.random.default_rng()  # Numpy Random Generator


def test_functional() -> None:
    """Test for creating backbone."""
    backbones = [
        "alexnet",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        "googlenet",
        "mobilenet_v2",
        "mobilenet_v3_large",
        "mobilenet_v3_small",
    ]
    assert CNNModel.postproc([1, 2]) == 1

    b = 4
    h = w = 512
    samples = torch.from_numpy(RNG.random((b, h, w, 3)))

    # Dummy entry, will generate ValueError if "try" fails without running the loop.
    backbone = "empty"
    try:
        for backbone in backbones:
            model = CNNModel(backbone, num_classes=1)
            model_ = model_to(on_gpu=ON_GPU, model=model)
            model.infer_batch(model_, samples, on_gpu=ON_GPU)
    except ValueError as exc:
        msg = f"Model {backbone} failed."
        raise AssertionError(msg) from exc

    # skipcq
    with pytest.raises(ValueError, match=r".*Backbone.*not supported.*"):
        CNNModel("shiny_model_to_crash", num_classes=2)


def test_timm_functional() -> None:
    """Test for creating backbone."""
    backbones = [
        "efficientnet_b0",
    ]
    assert TimmModel.postproc([1, 2]) == 1

    b = 4
    h = w = 224
    samples = torch.from_numpy(RNG.random((b, h, w, 3)))

    # Dummy entry, will generate ValueError if "try" fails without running the loop.
    backbone = "empty"
    try:
        for backbone in backbones:
            model = TimmModel(backbone=backbone, num_classes=1, pretrained=False)
            model_ = model_to(on_gpu=ON_GPU, model=model)
            model.infer_batch(model_, samples, on_gpu=ON_GPU)
    except ValueError as exc:
        msg = f"Model {backbone} failed."
        raise AssertionError(msg) from exc

    # skipcq
    with pytest.raises(ValueError, match=r".*Backbone.*not supported.*"):
        TimmModel(backbone="shiny_model_to_crash", num_classes=2, pretrained=False)
