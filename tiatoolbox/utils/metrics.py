# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

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
