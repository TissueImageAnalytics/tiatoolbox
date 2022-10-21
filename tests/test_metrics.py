"""Tests for metrics package in the toolbox."""

import numpy as np
import pytest

from tiatoolbox.utils.metrics import dice, f1_detection, pair_coordinates


def test_pair_coordinates():
    """Test for unique coordinates matching."""
    set_a = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    set_b = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1], [3.1, 3.1], [4.2, 4.2]])
    # 4 in set_a and 4, 5 in set B should be unpaired
    paired, unpaired_a, unpaired_b = pair_coordinates(set_a, set_b, 0.15)
    assert len(paired) == 4
    assert len(unpaired_a) == 1
    assert np.all(set_a[unpaired_a[0]] == np.array([4, 4]))
    assert len(unpaired_b) == 1
    assert np.all(set_b[unpaired_b[0]] == np.array([4.2, 4.2]))


def test_f1_detection():
    """Test for calculate F1 detection."""
    set_a = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    set_b = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1], [3.1, 3.1], [4.2, 4.2]])
    score = f1_detection(set_a, set_b, 0.2)
    assert score - 1.0 < 1.0e-6


def test_dice():
    """Test to calculate DICE."""
    gt_mask = np.random.randint(2, size=(256, 256))
    pred_mask = np.random.randint(2, size=(256, 256))
    dice_val = dice(gt_mask, pred_mask)
    assert dice_val >= 0
    assert dice_val <= 1.0

    gt_mask = np.ones(shape=(256, 256))
    pred_mask = np.ones(shape=(256, 256))

    dice_val = dice(gt_mask, pred_mask)
    assert dice_val == 1.0

    gt_mask = np.ones(shape=(256, 256))
    pred_mask = np.zeros(shape=(256, 256))

    dice_val = dice(gt_mask, pred_mask)
    assert dice_val == 0.0

    gt_mask = np.zeros(shape=(256, 256))
    pred_mask = np.zeros(shape=(256, 256))

    dice_val = dice(gt_mask, pred_mask)
    assert np.isnan(dice_val)


def test_dice_shape_mismatch_error():
    """Tests if the shape of inputs does not match."""
    gt_mask = np.random.randint(2, size=(256, 256, 1))
    pred_mask = np.random.randint(2, size=(256, 256, 3))
    with pytest.raises(ValueError, match=r".*Shape mismatch between the two masks.*"):
        _ = dice(gt_mask, pred_mask)
