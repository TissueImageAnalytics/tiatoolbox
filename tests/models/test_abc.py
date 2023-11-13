"""Unit test package for ABC and __init__ ."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from torch import nn

from tiatoolbox import rcParam
from tiatoolbox.models.architecture import (
    fetch_pretrained_weights,
    get_pretrained_model,
)
from tiatoolbox.models.models_abc import ModelABC
from tiatoolbox.utils import env_detection as toolbox_env

if TYPE_CHECKING:
    import numpy as np


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
    toolbox_env.running_on_ci(),
    reason="Local test on CLI",
)
def test_model_abc() -> None:
    """Test API in model ABC."""
    # test missing definition for abstract
    with pytest.raises(TypeError):
        # crash due to not defining forward, infer_batch, postproc
        ModelABC()  # skipcq

    # intentionally created to check error
    # skipcq
    class Proto(ModelABC):
        # skipcq
        def __init__(self: Proto) -> None:
            super().__init__()

        @staticmethod
        # skipcq
        def infer_batch() -> None:
            pass  # base class definition pass

    # skipcq
    with pytest.raises(TypeError):
        # crash due to not defining forward and postproc
        Proto()  # skipcq

    # intentionally create to check inheritance
    # skipcq
    class Proto(ModelABC):
        # skipcq
        def forward(self: Proto) -> None:
            pass  # base class definition pass

        @staticmethod
        # skipcq
        def infer_batch() -> None:
            pass  # base class definition pass

    model = Proto()
    assert model.preproc(1) == 1, "Must be unchanged!"
    assert model.postproc(1) == 1, "Must be unchanged!"

    # intentionally created to check error
    # skipcq
    class Proto(ModelABC):
        # skipcq
        def __init__(self: Proto) -> None:
            super().__init__()

        @staticmethod
        # skipcq
        def postproc(image: np.ndarray) -> None:
            return image - 2

        # skipcq
        def forward(self: Proto) -> None:
            pass  # base class definition pass

        @staticmethod
        # skipcq
        def infer_batch() -> None:
            pass  # base class definition pass

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

    # Test on CPU
    model_on_device = model.to(device="cpu")
    assert isinstance(model_on_device, nn.Module)

    # Test on GPU
    # no GPU on Travis so this will crash
    if not toolbox_env.has_gpu():
        model_on_device = model.to(device="cuda")
        assert isinstance(model_on_device, nn.Module)

    # Test load_weights_from_path() method
    weights_path = fetch_pretrained_weights("alexnet-kather100k")
    with pytest.raises(RuntimeError, match=r".*loading state_dict*"):
        _ = model.load_weights_from_path(weights_path)
