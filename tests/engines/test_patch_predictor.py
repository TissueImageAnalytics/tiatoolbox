"""Test tiatoolbox.models.engine.engine_abc."""

from __future__ import annotations

from typing import NoReturn

import pytest

from tiatoolbox.models.engine.engine_abc import EngineABC


def test_engine_abc() -> NoReturn:
    """Test EngineABC initialization."""
    with pytest.raises(
        TypeError,
        match=r".*Can't instantiate abstract class EngineABC*",
    ):
        # Can't instantiate abstract class with abstract methods
        EngineABC()  # skipcq
