"""Test tiatoolbox.models.engine.engine_abc."""
import pytest

from tiatoolbox.models.engine.engine_abc import EngineABC


def test_engine_abc():
    """Test EngineABC initialization."""
    with pytest.raises(TypeError):
        # Can't instantiate abstract class with abstract methods
        EngineABC()  # skipcq
