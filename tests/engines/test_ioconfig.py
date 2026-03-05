"""Tests for IOconfig."""

import numpy as np
import pytest

from tiatoolbox.models import IOSegmentorConfig, ModelIOConfigABC


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


def test_scale_to_highest_mpp() -> None:
    """Mpp → min(old_vals) / old_vals."""
    resolutions = [
        {"units": "mpp", "resolution": 0.25},
        {"units": "mpp", "resolution": 0.5},
    ]
    result = ModelIOConfigABC.scale_to_highest(resolutions, units="mpp")

    expected = np.array([1.0, 0.5])  # 0.25 / [0.25, 0.5]
    np.testing.assert_allclose(result, expected)


def test_scale_to_highest_mpp_reversed_order() -> None:
    """Ensure order is preserved even when resolutions are reversed."""
    resolutions = [
        {"units": "mpp", "resolution": 0.5},
        {"units": "mpp", "resolution": 0.25},
    ]
    result = ModelIOConfigABC.scale_to_highest(resolutions, units="mpp")

    expected = np.array([0.5, 1.0])  # 0.25 / [0.5, 0.25]
    np.testing.assert_allclose(result, expected)


def test_scale_to_highest_baseline() -> None:
    """Baseline → identity."""
    resolutions = [
        {"units": "baseline", "resolution": 2.0},
        {"units": "baseline", "resolution": 4.0},
    ]
    result = ModelIOConfigABC.scale_to_highest(resolutions, units="baseline")

    expected = [2.0, 4.0]
    assert result == expected


def test_scale_to_highest_power() -> None:
    """Power → old_vals / max(old_vals)."""
    resolutions = [
        {"units": "power", "resolution": 10},
        {"units": "power", "resolution": 5},
    ]
    result = ModelIOConfigABC.scale_to_highest(resolutions, units="power")

    expected = np.array([1.0, 0.5])  # [10, 5] / 10
    np.testing.assert_allclose(result, expected)


def test_scale_to_highest_invalid_units() -> None:
    """Test ModelIOConfigABC for unknown units."""
    resolutions = [{"units": "mpp", "resolution": 1.0}]
    with pytest.raises(ValueError, match="Unknown units"):
        ModelIOConfigABC.scale_to_highest(resolutions, units="unknown")


def test_modelio_to_baseline_without_save_resolution() -> None:
    """Test ModelIOConfigABC when save_resolution is None.

    Ensure ModelIOConfigABC.to_baseline does NOT add or convert
    save_resolution when it is None.

    """
    cfg = ModelIOConfigABC(
        input_resolutions=[{"units": "mpp", "resolution": 0.5}],
        output_resolutions=[{"units": "mpp", "resolution": 1.0}],
        patch_input_shape=(224, 224),
        stride_shape=(224, 224),
    )

    new_cfg = cfg.to_baseline()

    # save_resolution should not appear in the new config
    assert not hasattr(new_cfg, "save_resolution") or new_cfg.save_resolution is None


def test_ios_to_baseline_without_save_resolution() -> None:
    """Test IOSegmentorConfig when save_resolution is None.

    Ensure IOSegmentorConfig.to_baseline leaves save_resolution=None
    when no save_resolution is provided.

    """
    cfg = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 0.5}],
        output_resolutions=[{"units": "mpp", "resolution": 1.0}],
        patch_input_shape=(224, 224),
        patch_output_shape=(112, 112),
        stride_shape=(224, 224),
        save_resolution=None,
    )

    new_cfg = cfg.to_baseline()

    # save_resolution should remain None after conversion
    assert new_cfg.save_resolution is None
