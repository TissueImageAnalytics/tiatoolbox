"""Unit test package for vanilla CNN within toolbox."""

import numpy as np
import pytest
import torch

from tiatoolbox.models.architecture.vanilla import CNNModel, TimmModel
from tiatoolbox.models.models_abc import model_to

ON_GPU = False
RNG = np.random.default_rng()  # Numpy Random Generator
device = "cuda" if ON_GPU else "cpu"


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
    assert CNNModel.postproc(np.array([1, 2])) == 1

    b = 4
    h = w = 512
    samples = torch.from_numpy(RNG.random((b, h, w, 3)))

    # Dummy entry, will generate ValueError if "try" fails without running the loop.
    backbone = "empty"
    try:
        for backbone in backbones:
            model = CNNModel(backbone, num_classes=1)
            model_ = model_to(device=device, model=model)
            model.infer_batch(model_, samples, device=device)
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
    assert TimmModel.postproc(np.array([1, 2])) == 1

    b = 4
    h = w = 224
    samples = torch.from_numpy(RNG.random((b, h, w, 3)))

    # Dummy entry, will generate ValueError if "try" fails without running the loop.
    backbone = "empty"
    try:
        for backbone in backbones:
            model = TimmModel(backbone=backbone, num_classes=1, pretrained=False)
            model_ = model_to(device=device, model=model)
            model.infer_batch(model_, samples, device=device)
    except ValueError as exc:
        msg = f"Model {backbone} failed."
        raise AssertionError(msg) from exc

    # skipcq
    with pytest.raises(ValueError, match=r".*Backbone.*not supported.*"):
        TimmModel(backbone="shiny_model_to_crash", num_classes=2, pretrained=False)


def test_classification_models_forward_logits_infer_probabilities() -> None:
    """Forward should return logits while infer_batch returns probabilities."""
    cnn_model = CNNModel(backbone="resnet18", num_classes=3)
    cnn_logits = cnn_model(torch.rand((2, 3, 64, 64)))
    assert cnn_logits.shape == (2, 3)
    assert not torch.allclose(cnn_logits.sum(dim=1), torch.ones(2), atol=1e-4)

    cnn_probs = cnn_model.infer_batch(
        model=cnn_model,
        batch_data=torch.rand((2, 64, 64, 3)),
        device="cpu",
    )
    assert np.allclose(cnn_probs.sum(axis=1), np.ones(2), atol=1e-5)

    timm_model = TimmModel(backbone="efficientnet_b0", num_classes=3, pretrained=False)
    timm_logits = timm_model(torch.rand((2, 3, 224, 224)))
    assert timm_logits.shape == (2, 3)
    assert not torch.allclose(timm_logits.sum(dim=1), torch.ones(2), atol=1e-4)

    timm_probs = timm_model.infer_batch(
        model=timm_model,
        batch_data=torch.rand((2, 224, 224, 3)),
        device="cpu",
    )
    assert np.allclose(timm_probs.sum(axis=1), np.ones(2), atol=1e-5)
