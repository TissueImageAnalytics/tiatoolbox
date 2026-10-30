"""Unit test package for vanilla CNN within toolbox."""

import numpy as np
import pytest
import torch

from tiatoolbox.models.architecture.vanilla import CNNModel, TimmModel, _infer_batch
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


def _fix_head(model: CNNModel | TimmModel, logits: list[float]) -> None:
    """Make the classifier output the same logits for any input."""
    head = torch.nn.Linear(model.classifier.in_features, len(logits))
    torch.nn.init.zeros_(head.weight)
    with torch.no_grad():
        head.bias.copy_(torch.tensor(logits))
    model.classifier = head


def test_classifier_output_modes_pick_the_activation_and_labels() -> None:
    """Binary and multi-label heads use sigmoid and a threshold, not softmax."""
    batch = torch.rand((2, 32, 32, 3))

    binary = CNNModel("resnet18", num_classes=1)
    assert binary.output_mode == "binary"
    _fix_head(binary, [-2.0])
    probabilities = binary.infer_batch(binary, batch, device="cpu")
    assert probabilities.shape == (2, 1)
    assert np.allclose(probabilities, 1 / (1 + np.exp(2.0)))
    assert binary.postproc_func(probabilities).tolist() == [0, 0]
    # Engines hand infer_batch the model they run, which may be wrapped.
    wrapped = torch.nn.DataParallel(binary)
    assert np.allclose(binary.infer_batch(wrapped, batch, device="cpu"), probabilities)

    multi_label = CNNModel("resnet18", num_classes=2, output_mode="multi_label")
    _fix_head(multi_label, [2.0, 2.0])
    probabilities = multi_label.infer_batch(multi_label, batch, device="cpu")
    assert np.allclose(probabilities, 1 / (1 + np.exp(-2.0)))
    assert multi_label.postproc_func(probabilities).tolist() == [[1, 1], [1, 1]]

    single_label = CNNModel("resnet18", num_classes=2)
    assert single_label.output_mode == "single_label"
    _fix_head(single_label, [2.0, 2.0])
    probabilities = single_label.infer_batch(single_label, batch, device="cpu")
    assert np.allclose(probabilities, 0.5)
    assert single_label.postproc_func(probabilities).tolist() == [0, 0]

    timm_binary = TimmModel("efficientnet_b0", num_classes=1, pretrained=False)
    assert timm_binary.output_mode == "binary"
    _fix_head(timm_binary, [-2.0])
    probabilities = timm_binary.infer_batch(
        timm_binary, torch.rand((1, 224, 224, 3)), device="cpu"
    )
    assert np.allclose(probabilities, 1 / (1 + np.exp(2.0)))
    assert timm_binary.postproc_func(probabilities).tolist() == [0]


def test_classifier_output_modes_must_fit_the_outputs() -> None:
    """An explicit mode must agree with the number of outputs."""
    with pytest.raises(ValueError, match="needs one output"):
        CNNModel("resnet18", num_classes=2, output_mode="binary")
    with pytest.raises(ValueError, match="at least two outputs"):
        CNNModel("resnet18", num_classes=1, output_mode="single_label")
    with pytest.raises(ValueError, match="Unsupported output mode"):
        TimmModel(
            "efficientnet_b0", num_classes=2, pretrained=False, output_mode="sigmoid"
        )
    one_label = CNNModel("resnet18", num_classes=1, output_mode="multi_label")
    assert one_label.output_mode == "multi_label"


def test_inference_can_return_unactivated_outputs() -> None:
    """The shared inference adapter can retain logits for custom callers."""
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    batch = torch.full((2, 8, 8, 3), -2.0)
    actual = _infer_batch(model, batch, device="cpu", activation=None)
    np.testing.assert_array_equal(actual, np.full((2, 3), -2.0))
