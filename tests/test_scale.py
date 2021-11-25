"""Tests for scaling methods."""

import numpy as np
import pytest

from tiatoolbox.tools.scale import PlattScaling


def test_platt_scaler():
    """Test for Platt scaler."""
    np.random.seed(5)
    sample_size = 1000
    logit = np.random.rand(sample_size)
    # binary class
    label = np.concatenate(
        [np.full(int(0.9 * sample_size), -1), np.full(int(0.1 * sample_size), 1)]
    )
    scaler = PlattScaling(num_iters=1)
    scaler._fixer_a = 0.0
    scaler._fixer_b = 0.0
    _ = scaler.fit_transform(logit * 0.01, label)

    scaler = PlattScaling(num_iters=1)
    scaler._fixer_a = 0.0
    scaler._fixer_b = 1.0
    _ = scaler.fit_transform(logit * 0.01, label)

    scaler = PlattScaling(num_iters=10)
    _ = scaler.fit_transform(logit * 100, label)

    label = np.concatenate([np.full(int(sample_size), -1)])
    scaler = PlattScaling(num_iters=1)
    _ = scaler.fit_transform(logit * 0.01, label)

    with pytest.raises(ValueError, match=r".*same shape.*"):
        scaler.fit_transform(logit, label[:2])
    print(scaler)
