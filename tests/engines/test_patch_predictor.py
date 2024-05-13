"""Test tiatoolbox.models.engine.engine_abc."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import pytest

from tiatoolbox.models.engine.engine_abc import EngineABC

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import torch.nn

    from tiatoolbox.models.engine.io_config import ModelIOConfigABC


class TestEngineABC(EngineABC):
    """Test EngineABC."""

    def __init__(
        self: TestEngineABC,
        model: str | torch.nn.Module,
        weights: str | Path | None = None,
        verbose: bool | None = None,
    ) -> NoReturn:
        """Test EngineABC init."""
        super().__init__(model=model, weights=weights, verbose=verbose)

    def get_dataloader(
        self: EngineABC,
        images: Path,
        masks: Path | None = None,
        labels: list | None = None,
        ioconfig: ModelIOConfigABC | None = None,
        *,
        patch_mode: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Test pre process images."""
        return super().get_dataloader(
            images,
            masks,
            labels,
            ioconfig,
            patch_mode=patch_mode,
        )

    def save_wsi_output(
        self: EngineABC,
        raw_output: dict,
        save_dir: Path,
        **kwargs: dict,
    ) -> Path:
        """Test post_process_wsi."""
        return super().save_wsi_output(
            raw_output,
            save_dir=save_dir,
            **kwargs,
        )

    def infer_wsi(
        self: EngineABC,
        dataloader: torch.utils.data.DataLoader,
        save_path: Path,
        **kwargs: dict,
    ) -> dict | np.ndarray:
        """Test infer_wsi."""
        return super().infer_wsi(
            dataloader,
            save_path,
            **kwargs,
        )


def test_engine_abc() -> NoReturn:
    """Test EngineABC initialization."""
    with pytest.raises(
        TypeError,
        match=r".*Can't instantiate abstract class EngineABC*",
    ):
        # Can't instantiate abstract class with abstract methods
        EngineABC()  # skipcq
