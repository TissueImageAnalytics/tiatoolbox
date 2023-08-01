"""Tests for IOconfig."""

import pytest

from tiatoolbox.models import ModelIOConfigABC


def test_validation_error_io_config() -> None:
    """Test Validation Error for ModelIOConfigABC."""
    with pytest.raises(ValueError, match=r".*Multiple resolution units found.*"):
        ModelIOConfigABC(
            input_resolutions=[
                {"units": "baseline", "resolution": 1.0},
                {"units": "mpp", "resolution": 0.25},
            ],
            patch_input_shape=(224, 224),
        )

    with pytest.raises(ValueError, match=r"Invalid resolution units.*"):
        ModelIOConfigABC(
            input_resolutions=[{"units": "level", "resolution": 1.0}],
            patch_input_shape=(224, 224),
        )
