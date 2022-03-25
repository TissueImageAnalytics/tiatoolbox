"""This module defines several metrics used in computational pathology."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance


def pair_coordinates(setA, setB, radius):
    """Find optimal unique pairing between two sets of coordinates.

    This function uses the Munkres or Kuhn-Munkres algorithm behind the
    scene to find the most optimal unique pairing when pairing points
    in set B against points in set A, using euclidean distance as
    the cost function.

    Args:
        setA (ndarray): an array of shape Nx2 contains the of XY coordinate
            of N different points.
        setB (ndarray): an array of shape Mx2 contains the of XY coordinate
            of M different points.
        radius: valid area around a point in setA to consider a given
            coordinate in setB a candidate for matching.

    Returns:
        pairing (ndarray): an array of shape Kx2, each item in K contains
            indices where point at index [0] in set A paired with
            point in set B at index [1].

        unpairedA (ndarray): indices of unpaired points in set A.
        unpairedB (ndarray): indices of unpaired points in set A.

    """
    # * Euclidean distance as the cost matrix
    pair_distance = distance.cdist(setA, setB, metric="euclidean")

    # * Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence
    # is return, thus the unique pairing is ensured
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances
    # outside of designated radius
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    pairing = np.concatenate([pairedA[:, None], pairedB[:, None]], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
    return pairing, unpairedA, unpairedB


def f1_detection(true, pred, radius):
    """Calculate the F1-score for predicted set of coordinates."""
    (paired_true, unpaired_true, unpaired_pred) = pair_coordinates(true, pred, radius)

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    return tp / (tp + 0.5 * fp + 0.5 * fn)
