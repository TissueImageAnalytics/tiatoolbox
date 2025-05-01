"""This module defines several metrics used in computational pathology."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance


def pair_coordinates(
    set_a: np.ndarray,
    set_b: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find optimal unique pairing between two sets of coordinates.

    This function uses the Munkres or Kuhn-Munkres algorithm behind the
    scene to find the most optimal unique pairing when pairing points in
    set B against points in set A, using Euclidean distance as the cost
    function.

    Args:
        set_a (np.ndarray):
            An array of shape Nx2 contains the of XY coordinate of N
            different points.
        set_b (np.ndarray):
            An array of shape Mx2 contains the of XY coordinate of M
            different points.
        radius (float):
            Valid area around a point in set A to consider a given
            coordinate in set B a candidate for matching.

    Returns:
        tuple:
            - :class:`numpy.ndarray` - Pairing:
                    An array of shape Kx2, each item in K contains indices
                    where point at index [0] in set A paired with point in
                    set B at index [1].
            - :class:`numpy.ndarray` - Unpaired A:
                    Indices of unpaired points in set A.
            - :class:`numpy.ndarray` - Unpaired B:
                    Indices of unpaired points in set B.


    Examples:
        >>> from tiatoolbox.utils.metrics import pair_coordinates
        >>> # Generate two random example sets; replace with your own data
        >>> import numpy as np
        >>> np.random.seed(6)
        >>> set_a_num_points = np.random.randint(low=10, high=30)
        >>> set_b_num_points = np.random.randint(low=10, high=30)
        >>> set_a = np.random.randint(low=0, high=25, size=(set_a_num_points, 2))
        >>> set_b = np.random.randint(low=0, high=25, size=(set_b_num_points, 2))
        >>> radius = 2.0
        >>> # Example usage of pair_coordinates
        >>> pairing, unpaired_a, unpaired_b = pair_coordinates(set_a, set_b, radius)

    """
    # * Euclidean distance as the cost matrix
    pair_distance = distance.cdist(set_a, set_b, metric="euclidean")

    # * Munkres pairing with scipy library
    # The algorithm return (row indices, matched column indices) if
    # there is multiple same cost in a row, index of first occurrence is
    # return, thus the unique pairing is ensured.
    indices_a, paired_indices_b = linear_sum_assignment(pair_distance)

    # Extract the paired cost and remove instances outside designated
    # radius.
    pair_cost = pair_distance[indices_a, paired_indices_b]

    paired_a = indices_a[pair_cost <= radius]
    paired_b = paired_indices_b[pair_cost <= radius]

    pairing = np.concatenate([paired_a[:, None], paired_b[:, None]], axis=-1)
    unpaired_a = np.delete(np.arange(set_a.shape[0]), paired_a)
    unpaired_b = np.delete(np.arange(set_b.shape[0]), paired_b)
    return pairing, unpaired_a, unpaired_b


def f1_detection(true: np.ndarray, pred: np.ndarray, radius: float) -> float:
    """Calculate the F1-score for predicted set of coordinates.

    Examples:
        >>> from tiatoolbox.utils.metrics import f1_detection
        >>> # Generate two random example sets; replace with your own data
        >>> import numpy as np
        >>> np.random.seed(6)
        >>> true_num_points = np.random.randint(low=10, high=30)
        >>> pred_num_points = np.random.randint(low=10, high=30)
        >>> true = np.random.randint(low=0, high=25, size=(true_num_points, 2))
        >>> pred = np.random.randint(low=0, high=25, size=(pred_num_points, 2))
        >>> radius = 2.0
        >>> # Example usage of f1_detection
        >>> f1_score = f1_detection(true, pred, radius)

    """
    (paired_true, unpaired_true, unpaired_pred) = pair_coordinates(true, pred, radius)

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    return tp / (tp + 0.5 * fp + 0.5 * fn)


def dice(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    r"""Compute the Sørensen-Dice coefficient.

    This function computes `Sørensen-Dice coefficient
    <https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient>`_,
    between the two masks.

    .. math::
        DSC = 2 * |X ∩ Y| / |X| + |Y|

    Args:
        gt_mask (:class:`np.ndarray`):
            A binary ground truth mask
        pred_mask (:class:`np.ndarray`):
            A binary predicted mask

    Returns:
        :class:`float`:
            An estimate of Sørensen-Dice coefficient value.

    Examples:
        >>> from tiatoolbox.utils.metrics import dice
        >>> # Generate two random example masks; replace with your own data
        >>> import numpy as np
        >>> np.random.seed(6)
        >>> gt_mask = (np.random.rand(256, 256) > 0.8).astype(np.uint8)
        >>> pred_mask = (np.random.rand(256, 256) > 0.8).astype(np.uint8)
        >>> # Example usage of dice
        >>> dice_score = dice(gt_mask, pred_mask)

    """
    if gt_mask.shape != pred_mask.shape:
        msg = f"{'Shape mismatch between the two masks.'}"
        raise ValueError(msg)

    gt_mask = gt_mask.astype(np.bool_)
    pred_mask = pred_mask.astype(np.bool_)
    sum_masks = gt_mask.sum() + pred_mask.sum()
    if sum_masks == 0:
        return np.nan
    return 2 * np.logical_and(gt_mask, pred_mask).sum() / sum_masks
