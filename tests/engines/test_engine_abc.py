"""Test tiatoolbox.models.engine.engine_abc."""
from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import pytest

from tiatoolbox.models.engine.engine_abc import EngineABC

if TYPE_CHECKING:
    import torch.nn


class TestEngineABC(EngineABC):
    """Test EngineABC."""

    def __init__(self: TestEngineABC, model: str | torch.nn.Module) -> NoReturn:
        """Test EngineABC init."""
        super().__init__(model=model)

    def infer_patch(self: EngineABC) -> NoReturn:
        """Test infer_patch."""
        ...  # dummy function for tests.

    def infer_wsi(self: EngineABC) -> NoReturn:
        """Test infer_wsi."""
        ...  # dummy function for tests.

    def post_process_patch(self: EngineABC) -> NoReturn:
        """Test post_process_patch."""
        ...  # dummy function for tests.

    def post_process_wsi(self: EngineABC) -> NoReturn:
        """Test post_process_wsi."""
        ...  # dummy function for tests.

    def pre_process_patch(self: EngineABC) -> NoReturn:
        """Test pre_process_patch."""
        ...  # dummy function for tests.

    def pre_process_wsi(self: EngineABC) -> NoReturn:
        """Test pre_process_wsi."""
        ...  # dummy function for tests.


def test_engine_abc():
    """Test EngineABC initialization."""
    with pytest.raises(
        TypeError,
        match=r".*Can't instantiate abstract class EngineABC with abstract methods*",
    ):
        # Can't instantiate abstract class with abstract methods
        EngineABC()  # skipcq


def test_engine_abc_incorrect_model_type():
    """Test EngineABC initialization with incorrect model type."""
    with pytest.raises(
        TypeError,
        match=r".*missing 1 required positional argument: 'model'",
    ):
        TestEngineABC()  # skipcq

    with pytest.raises(
        TypeError,
        match="Input model must be a string or 'torch.nn.Module'.",
    ):
        # Can't instantiate abstract class with abstract methods
        TestEngineABC(model=1)


def test_incorrect_ioconfig():
    """Test EngineABC initialization with incorrect ioconfig."""
    import torchvision.models as torch_models

    model = torch_models.resnet18()
    engine = TestEngineABC(model=model)
    with pytest.raises(
        ValueError,
        match=r".*provide a valid ModelIOConfigABC.*",
    ):
        engine.run(images=[], masks=[], ioconfig=None)
