"""Test for Patch Predictor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiatoolbox.models.engine.patch_predictor import PatchPredictor

if TYPE_CHECKING:
    import pytest


# -------------------------------------------------------------------------------------
# Engine
# -------------------------------------------------------------------------------------


def test_patch_predictor_api(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Helper function to get the model output using API 1."""
    _ = PatchPredictor(model="resnet18-kather100k", batch_size=1)
    assert "PatchPredictor will be deprecated in v2.1" in caplog.text
