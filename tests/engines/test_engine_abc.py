"""Test tiatoolbox.models.engine.engine_abc."""
from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import pytest

from tiatoolbox.models.engine.engine_abc import EngineABC, prepare_engines_save_dir

if TYPE_CHECKING:
    import numpy as np
    import torch.nn
    from torch.utils.data import DataLoader


class TestEngineABC(EngineABC):
    """Test EngineABC."""

    def __init__(self: TestEngineABC, model: str | torch.nn.Module) -> NoReturn:
        """Test EngineABC init."""
        super().__init__(model=model)

    def infer_patches(self: EngineABC, data_loader: DataLoader) -> NoReturn:
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

    def pre_process_patches(
        self: EngineABC,
        images: np.ndarray,
        labels: list,
    ) -> NoReturn:
        """Test pre_process_patch."""
        ...  # dummy function for tests.

    def pre_process_wsi(self: EngineABC) -> NoReturn:
        """Test pre_process_wsi."""
        ...  # dummy function for tests.


def test_engine_abc() -> NoReturn:
    """Test EngineABC initialization."""
    with pytest.raises(
        TypeError,
        match=r".*Can't instantiate abstract class EngineABC with abstract methods*",
    ):
        # Can't instantiate abstract class with abstract methods
        EngineABC()  # skipcq


def test_engine_abc_incorrect_model_type() -> NoReturn:
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


def test_incorrect_ioconfig() -> NoReturn:
    """Test EngineABC initialization with incorrect ioconfig."""
    import torchvision.models as torch_models

    model = torch_models.resnet18()
    engine = TestEngineABC(model=model)
    with pytest.raises(
        ValueError,
        match=r".*provide a valid ModelIOConfigABC.*",
    ):
        engine.run(images=[], masks=[], ioconfig=None)


def test_prepare_engines_save_dir(
    tmp_path: pytest.TempPathFactory,
    caplog: pytest.LogCaptureFixture,
) -> NoReturn:
    """Test prepare save directory for engines."""
    out_dir = prepare_engines_save_dir(
        save_dir=tmp_path / "patch_output",
        patch_mode=True,
        len_images=1,
    )

    assert out_dir == tmp_path / "patch_output"
    assert out_dir.exists()

    with pytest.raises(
        OSError,
        match=r".*More than 1 WSIs detected but there is no save directory provided.*",
    ):
        _ = prepare_engines_save_dir(
            save_dir=None,
            patch_mode=False,
            len_images=2,
        )

    out_dir = prepare_engines_save_dir(
        save_dir=tmp_path / "wsi_single_output",
        patch_mode=False,
        len_images=1,
    )

    assert out_dir == tmp_path / "wsi_single_output"
    assert out_dir.exists()
    assert r"When providing multiple whole-slide images / tiles" not in caplog.text

    out_dir = prepare_engines_save_dir(
        save_dir=tmp_path / "wsi_multiple_output",
        patch_mode=False,
        len_images=2,
    )

    assert out_dir == tmp_path / "wsi_multiple_output"
    assert out_dir.exists()
    assert r"When providing multiple whole-slide images / tiles" in caplog.text
