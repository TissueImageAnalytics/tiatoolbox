"""Unit test package for ABC and __init__ ."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

import tiatoolbox.models
from tiatoolbox import rcParam, utils
from tiatoolbox.models.architecture import (
    fetch_pretrained_weights,
    get_pretrained_model,
)
from tiatoolbox.models.models_abc import ModelABC
from tiatoolbox.utils import env_detection as toolbox_env

if TYPE_CHECKING:
    import numpy as np


class ProtoRaisesTypeError(ModelABC):
    """Intentionally created to check for TypeError."""

    # skipcq
    def __init__(self: Proto) -> None:
        """Initialize ProtoRaisesTypeError."""
        super().__init__()

    @staticmethod
    # skipcq
    def infer_batch() -> None:
        """Define infer batch."""
        # base class definition pass


class ProtoNoPostProcess(ModelABC):
    """Intentionally created to check No Post Processing."""

    def forward(self: ProtoNoPostProcess) -> None:
        """Define forward function."""

    @staticmethod
    # skipcq
    def infer_batch() -> None:
        """Define infer batch."""


class Proto(ModelABC):
    """Intentionally created to check error."""

    def __init__(self: Proto) -> None:
        """Initialize Proto."""
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    @staticmethod
    # skipcq
    def postproc(image: np.ndarray) -> np.ndarray:
        """Define postproc function."""
        return image - 2

    # skipcq
    def forward(self: Proto) -> None:
        """Define forward function."""

    @staticmethod
    # skipcq
    def infer_batch() -> None:
        """Define infer batch."""
        pass  # base class definition pass  # noqa: PIE790


@pytest.mark.skipif(
    toolbox_env.running_on_ci() or not toolbox_env.has_gpu(),
    reason="Local test on machine with GPU.",
)
def test_get_pretrained_model() -> None:
    """Test for downloading and creating pretrained models."""
    pretrained_info = rcParam["pretrained_model_info"]
    for pretrained_name in pretrained_info:
        get_pretrained_model(pretrained_name, overwrite=True)


@pytest.mark.skipif(
    toolbox_env.running_on_ci() or not toolbox_env.has_gpu(),
    reason="Local test on CLI",
)
def test_model_to_cuda() -> None:
    """This Test should pass locally if GPU is available."""
    # Test on GPU
    # no GPU on Travis so this will crash
    model = Proto()  # skipcq
    assert model.dummy_param.device.type == "cpu"
    model = model.to(device="cuda")
    assert isinstance(model, nn.Module)
    assert model.dummy_param.device.type == "cuda"


def test_model_abc() -> None:
    """Test API in model ABC."""
    # test missing definition for abstract
    with pytest.raises(TypeError):
        # crash due to not defining forward, infer_batch, postproc
        ModelABC()  # skipcq

    # skipcq
    with pytest.raises(TypeError):
        # crash due to not defining forward and postproc
        ProtoRaisesTypeError()  # skipcq

    model = ProtoNoPostProcess()
    assert model.preproc(1) == 1, "Must be unchanged!"
    assert model.postproc(1) == 1, "Must be unchanged!"

    model = Proto()  # skipcq
    # test assign un-callable to preproc_func/postproc_func
    with pytest.raises(ValueError, match=r".*callable*"):
        model.postproc_func = 1  # skipcq: PYL-W0201
    with pytest.raises(ValueError, match=r".*callable*"):
        model.preproc_func = 1  # skipcq: PYL-W0201

    # test setter/getter/initial of preproc_func/postproc_func
    assert model.preproc_func(1) == 1
    model.preproc_func = lambda x: x - 1  # skipcq: PYL-W0201
    assert model.preproc_func(1) == 0
    assert model.preproc(1) == 1, "Must be unchanged!"
    assert model.postproc(1) == -1, "Must be unchanged!"
    model.preproc_func = None  # skipcq: PYL-W0201
    assert model.preproc_func(2) == 2

    # repeat the setter test for postproc
    assert model.postproc_func(2) == 0
    model.postproc_func = lambda x: x - 1  # skipcq: PYL-W0201
    assert model.postproc_func(1) == 0
    assert model.preproc(1) == 1, "Must be unchanged!"
    assert model.postproc(2) == 0, "Must be unchanged!"
    # coverage setter check
    model.postproc_func = None  # skipcq: PYL-W0201
    assert model.postproc_func(2) == 0

    # Test load_weights_from_file() method
    weights_path = fetch_pretrained_weights("alexnet-kather100k")
    with pytest.raises(RuntimeError, match=r".*loading state_dict*"):
        _ = model.load_weights_from_file(weights_path)

    # Test on CPU
    model = model.to(device="cpu")
    assert isinstance(model, nn.Module)
    assert model.dummy_param.device.type == "cpu"


def test_model_to() -> None:
    """Test for placing model on device."""
    import torchvision.models as torch_models
    from torch import nn

    # Test on GPU
    # no GPU on Travis so this will crash
    if not utils.env_detection.has_gpu():
        model = torch_models.resnet18()
        with pytest.raises((AssertionError, RuntimeError)):
            _ = tiatoolbox.models.models_abc.model_to(device="cuda", model=model)

    # Test on CPU
    model = torch_models.resnet18()
    model = tiatoolbox.models.models_abc.model_to(device="cpu", model=model)
    assert isinstance(model, nn.Module)


def test_get_pretrained_model_not_str() -> None:
    """Test TypeError is raised if input is not str."""
    with pytest.raises(TypeError, match="pretrained_model must be a string."):
        _ = get_pretrained_model(1)


def test_get_pretrained_model_not_in_info() -> None:
    """Test ValueError is raised if input is not in info."""
    with pytest.raises(ValueError, match="Pretrained model `alexnet` does not exist."):
        _ = get_pretrained_model("alexnet")
