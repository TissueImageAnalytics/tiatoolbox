"""Tests for graph construction tools."""

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt

from tiatoolbox.tools.graph import (
    SlideGraphConstructor,
    affinity_to_edge_index,
    delaunay_adjacency,
    edge_index_to_triangles,
    triangle_signed_area,
)


def test_delaunay_adjacency_dthresh_type():
    """Test empty input raises a TypeError if dthresh is not a Number."""
    with pytest.raises(TypeError, match="number"):
        delaunay_adjacency(points=[[0, 0]], dthresh=None)


def test_delaunay_adjacency_empty():
    """Test empty input raises a ValueError."""
    points = np.array([])
    with pytest.raises(ValueError, match="Points must have length >= 4"):
        delaunay_adjacency(points, 10)


def test_delaunay_adjacency_invalid_shape():
    """Test points with invalid shape (not NxM) raises a ValueError."""
    points = np.random.rand(4, 4, 4)
    with pytest.raises(ValueError, match="NxM"):
        delaunay_adjacency(points, 10)


def test_delaunay_adjacency_nothing_connected():
    """Test delaunay_adjacency does not connect points further than dthresh.

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
    adjacency_matrix = delaunay_adjacency(points=points, dthresh=0.5)
    assert np.sum(adjacency_matrix) == 0


def test_delaunay_adjacency_connected():
    """Test delaunay_adjacency connects expects points in handcrafted input."""
    # Simple convex hull with the minimum of 4 points
    points = np.array(
        [
            [0, 0],
            [1, 1],
            [1, 3],
            [1, 6],
        ]
    )
    adjacency_matrix = delaunay_adjacency(points=points, dthresh=1.5)
    # Expect 1 connection, symmetrical so dividing by 2
    assert np.sum(adjacency_matrix) / 2 == 1


def test_affinity_to_edge_index_fuzz_output_shape():
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
        assert 0 <= m <= n**2


def test_affinity_to_edge_index_invalid_fuzz_input_shape():
    """Test that affinity_to_edge fails with non-square input."""
    # Generate some random square inputs
    np.random.seed(123)
    for _ in range(100):
        input_shape = [np.random.randint(2, 10)] * 2
        input_shape[1] -= 1
        affinity_matrix = np.random.sample(input_shape)
        threshold = np.random.rand()
        # Convert to torch randomly
        if np.random.rand() > 0.5:
            affinity_matrix = torch.tensor(affinity_matrix)
        with pytest.raises(ValueError, match="square"):
            _ = affinity_to_edge_index(affinity_matrix, threshold=threshold)


def test_edge_index_to_triangles_invalid_input():
    """Test edge_index_to_triangles fails with invalid input."""
    edge_index = torch.tensor([[0, 1], [0, 2], [1, 2]])
    with pytest.raises(ValueError, match="must be a 2xM"):
        edge_index_to_triangles(edge_index)


def test_triangle_signed_area():
    """Test that the signed area of a triangle is correct."""
    # Triangle with positive area
    points = np.array([[0, 0], [1, 0], [0, 1]])
    area = triangle_signed_area(points)
    assert area == 0.5

    # Triangle with negative area
    points = np.array([[0, 0], [1, 0], [0, -1]])
    area = triangle_signed_area(points)
    assert area == -0.5

    # Triangle with co-linear points
    points = np.array([[0, 0], [1, 1], [2, 2]])
    area = triangle_signed_area(points)
    assert area == 0

    # Triangle with larger area
    points = np.array([[0, 0], [2, 0], [0, 2]])
    area = triangle_signed_area(points)
    assert area == 2


def test_triangle_signed_area_invalid_input():
    """Test that the signed area of a triangle with invalid input fails."""
    points = np.random.rand(3, 3)
    with pytest.raises(ValueError, match="3x2"):
        triangle_signed_area(points)


def test_edge_index_to_trainangles_single():
    """Test edge_index_to_triangles with a simple 2XM input matrix.

    Basic test case for a single triangle.

    0 -- 1
    |   /
    | /
    2
    """
    edge_index = np.array([[0, 1], [0, 2], [1, 2]]).T
    triangles = edge_index_to_triangles(edge_index)
    assert triangles.shape == (1, 3)
    assert np.array_equal(triangles, np.array([[0, 1, 2]]))


def test_edge_index_to_trainangles_many():
    """Test edge_index_to_triangles with a simple 2XM input matrix.

    Moderate test case for a few trainangles.

    4 -- 3
    |  / |
    |/   |
    0 -- 1
    |   /
    | /
    2
    """
    edge_index = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [4, 3]]).T
    triangles = edge_index_to_triangles(edge_index)
    assert triangles.shape == (3, 3)


def test_slidegraph_build_feature_range_thresh_none():
    """Test SlideGraphConstructor builds a graph without removing features."""
    # Generate random points and features
    np.random.seed(0)
    points = np.random.rand(100, 2)
    features = np.random.rand(100, 100) / 1e-5
    # Build the graph
    graph = SlideGraphConstructor.build(
        points=points, features=features, feature_range_thresh=None
    )
    assert graph["x"].shape[1] == 100


class TestConstructor:
    scenarios = [
        ("SlideGraph", {"graph_constructor": SlideGraphConstructor}),
    ]

    @staticmethod
    def test_build(graph_constructor):
        """Test that build outputs are in an expected format.

        Check the lengths and ranges of outputs with random data as input.

        """
        np.random.seed(123)
        points = np.concatenate(
            [np.random.rand(25, 2) * 100 + (offset * 1000) for offset in range(10)]
        )
        features = np.concatenate(
            [np.random.rand(25, 100) * 100 + (offset * 1000) for offset in range(10)]
        )
        graph = graph_constructor.build(points, features)
        x = graph["x"]
        assert len(x) > 0
        assert len(x) <= len(points)

        edge_index = graph["edge_index"]
        two, m = edge_index.shape
        n = len(x)
        assert two == 2
        assert 0 <= m <= n**2

    @staticmethod
    def test_visualise(graph_constructor):
        """Test visualising a graph."""
        np.random.seed(123)
        points = np.concatenate(
            [np.random.rand(25, 2) * 100 + (offset * 1000) for offset in range(10)]
        )
        features = np.concatenate(
            [np.random.rand(25, 100) * 100 + (offset * 1000) for offset in range(10)]
        )
        graph = graph_constructor.build(points, features)
        graph_constructor.visualise(graph)
        plt.close()

    @staticmethod
    def test_visualise_ax(graph_constructor):
        """Test visualising a graph on a given axis."""
        np.random.seed(123)
        points = np.concatenate(
            [np.random.rand(25, 2) * 100 + (offset * 1000) for offset in range(10)]
        )
        features = np.concatenate(
            [np.random.rand(25, 100) * 100 + (offset * 1000) for offset in range(10)]
        )
        _, ax = plt.subplots()
        graph = graph_constructor.build(points, features)
        graph_constructor.visualise(graph, ax=ax)
        plt.close()

    @staticmethod
    def test_visualise_custom_color_function(graph_constructor):
        """Test visualising a graph with a custom color function."""
        np.random.seed(123)
        points = np.concatenate(
            [np.random.rand(25, 2) * 100 + (offset * 1000) for offset in range(10)]
        )
        features = np.concatenate(
            [np.random.rand(25, 100) * 100 + (offset * 1000) for offset in range(10)]
        )
        graph = graph_constructor.build(points, features)
        cmap = plt.get_cmap("viridis")
        graph_constructor.visualise(
            graph, color=lambda g: cmap(np.mean(g["x"], axis=1))
        )
        plt.close()

    @staticmethod
    def test_visualise_static_color(graph_constructor):
        """Test visualising a graph with a custom color function."""
        np.random.seed(123)
        points = np.concatenate(
            [np.random.rand(25, 2) * 100 + (offset * 1000) for offset in range(10)]
        )
        features = np.concatenate(
            [np.random.rand(25, 100) * 100 + (offset * 1000) for offset in range(10)]
        )
        graph = graph_constructor.build(points, features)
        graph_constructor.visualise(graph, color="orange")
        plt.close()

    @staticmethod
    def test_visualise_invalid_input(graph_constructor):
        """Test visualising a graph with invalid input."""
        with pytest.raises(ValueError, match="must contain key `x`"):
            graph_constructor.visualise({})
        with pytest.raises(ValueError, match="must contain key `edge_index`"):
            graph_constructor.visualise({"x": []})
        with pytest.raises(ValueError, match="must contain key `coordinates`"):
            graph_constructor.visualise({"x": [], "edge_index": []})
