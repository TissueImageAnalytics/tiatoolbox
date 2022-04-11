"""Functional unit test package for IDARS."""

import torch

from tiatoolbox.models.architecture.idars import IDaRS


def test_functional():
    """Functional test for architectures."""
    # test forward
    samples = torch.rand(4, 3, 224, 224, dtype=torch.float32)
    model = IDaRS("resnet18")
    model(samples)

    model = IDaRS("resnet34")
    model(samples)

    # test preproc function
    img = torch.rand(224, 224, 3, dtype=torch.float32)
    img_ = IDaRS.preproc(img.numpy())
    assert tuple(img_.shape) == (224, 224, 3)
    img_ = IDaRS.preproc(img.numpy())
    assert tuple(img_.shape) == (224, 224, 3)
    # dummy to make runtime crash
    img_ = IDaRS.preproc(img.numpy() / 0.0)
    assert tuple(img_.shape) == (224, 224, 3)
