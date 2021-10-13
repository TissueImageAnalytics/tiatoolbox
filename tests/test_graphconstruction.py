"""Tests for graph construction tools."""

import numpy as np
import pytest
import torch

from tiatoolbox.tools.graphconstruction import (
    affinity_to_edge_index,
    build_graph_dict,
    connect_points,
)


def test_connect_points_dthresh_type():
    """Test empty input raises a TypeError if dthresh is not a Number."""
    with pytest.raises(TypeError, match="number"):
        connect_points(points=[[0, 0]], dthresh=None)


def test_connect_points_empty():
    """Test empty input raises a ValueError."""
    points = np.array([])
    with pytest.raises(ValueError, match="Points must have length >= 4"):
        connect_points(points)


def test_connect_points_invalid_shape():
    """Test points with invalid shape (not NxM) raises a ValueError."""
    points = np.random.rand(4, 4, 4)
    with pytest.raises(ValueError, match="NxM"):
        connect_points(points)


def test_connect_points_nothing_connected():
    """Test connect_points does not connect points further than dthresh.

    Nothing should connect for this case as all points are further
    apart than dthresh.
    """
    # Simple convex hull with the minimum of 4 points
    points = np.array(
        [
            [0, 0],
            [1, 1],
            [1, 3],
            [1, 6],
        ]
    )
    adjacency_matrix = connect_points(points=points, dthresh=0.5)
    assert np.sum(adjacency_matrix) == 0


def test_connect_points_connected():
    """Test connect_points connects expects points in handcrafted input."""
    # Simple convex hull with the minimum of 4 points
    points = np.array(
        [
            [0, 0],
            [1, 1],
            [1, 3],
            [1, 6],
        ]
    )
    adjacency_matrix = connect_points(points=points, dthresh=1.5)
    # Expect 1 connection, symmetrical so dividing by 2
    assert np.sum(adjacency_matrix) / 2 == 1


def test_affinity_to_edge_fuzz_output_shape():
    """Fuzz test that output shape is 2xM for affinity_to_edge.

    Output is 2xM, where M is the number of edges in the graph, i.e.
    the number of connections between nodes with a value > threshold.
    """
    np.random.seed(123)
    for _ in range(1000):
        # Generate some random square inputs
        input_shape = [np.random.randint(2, 10)] * 2
        affinity_matrix = np.random.sample(input_shape)
        threshold = np.random.rand()
        # Convert to torch randomly
        if np.random.rand() > 0.5:
            affinity_matrix = torch.tensor(affinity_matrix)
        edge_index = affinity_to_edge_index(affinity_matrix, threshold=threshold)
        # noqa Check the output has shape (2, M)
        assert len(edge_index.shape) == 2
        n = len(affinity_matrix)
        two, m = edge_index.shape
        assert two == 2
        assert 0 <= m <= n ** 2


def test_build_graph_dict():
    points = np.concatenate([np.random.rand(25, 2) + offset for offset in range(4)])
    features = np.concatenate([np.random.rand(25, 100) + offset for offset in range(4)])
    graph = build_graph_dict(points, features, lambda_h=0.001)
    x = graph["x"]
    assert len(x) > 0
    assert len(x) <= len(points)

    edge_index = graph["edge_index"]
    two, m = edge_index.shape
    n = len(x)
    assert two == 2
    assert 0 <= m <= n ** 2
