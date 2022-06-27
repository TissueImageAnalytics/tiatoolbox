"""Tests for scaling methods."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as PlattScaling


def test_platt_scaler():
    """Test for Platt scaler."""
    np.random.seed(5)
    sample_size = 1000
    logit = np.random.rand(sample_size)
    # binary class
    label = np.concatenate(
        [np.full(int(0.9 * sample_size), -1), np.full(int(0.1 * sample_size), 1)]
    )
    scaler = PlattScaling(max_iter=1)
    scaler._fixer_a = 0.0
    scaler._fixer_b = 0.0
    scaler.fit(np.array(logit * 0.01, ndmin=2).T, label)
    _ = scaler.predict_proba(np.array(logit * 0.01, ndmin=2).T)

    scaler = PlattScaling(max_iter=1)
    scaler._fixer_a = 0.0
    scaler._fixer_b = 1.0
    scaler.fit(np.array(logit * 0.01, ndmin=2).T, label)
    _ = scaler.predict_proba(np.array(logit * 0.01, ndmin=2).T)

    scaler = PlattScaling(max_iter=10)
    scaler.fit(np.array(logit * 100, ndmin=2).T, label)
    _ = scaler.predict_proba(np.array(logit * 0.01, ndmin=2).T)

    label = np.concatenate([np.full(int(sample_size), -1)])
    scaler = PlattScaling(max_iter=1)
    with pytest.raises(ValueError, match="needs samples of at least 2 classes"):
        scaler.fit(np.array(logit * 0.01, ndmin=2).T, label)

    with pytest.raises(ValueError, match="inconsistent"):
        scaler.fit(np.array(logit, ndmin=2).T, label[:2])
    print(scaler)
