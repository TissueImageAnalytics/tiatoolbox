"""Construction and visualisation of graphs for WSI prediction."""

from __future__ import annotations

from collections import defaultdict
from numbers import Number
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
import umap
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from scipy.cluster import hierarchy
from scipy.spatial import Delaunay, cKDTree


def delaunay_adjacency(points: ArrayLike, dthresh: Number) -> list:
    """Create an adjacency matrix via Delaunay triangulation from a list of coordinates.

    Points which are further apart than dthresh will not be connected.

    See https://en.wikipedia.org/wiki/Adjacency_matrix.

    Args:
        points (ArrayLike):
            An nxm list of coordinates.
        dthresh (int):
            Distance threshold for triangulation.

    Returns:
        ArrayLike:
            Adjacency matrix of shape NxN where 1 indicates connected
            and 0 indicates unconnected.

    Example:
        >>> points = np.random.rand(100, 2)
        >>> adjacency = delaunay_adjacency(points)

    """
    # Validate inputs
    if not isinstance(dthresh, Number):
        raise TypeError("dthresh must be a number.")
    if len(points) < 4:
        raise ValueError("Points must have length >= 4.")
    if len(np.shape(points)) != 2:
        raise ValueError("Points must have an NxM shape.")
    # Apply Delaunay triangulation to the coordinates to get a
    # tessellation of triangles.
    tessellation = Delaunay(points)
    # Find all connected neighbours for each point in the set of
    # triangles. Starting with an empty dictionary.
    triangle_neighbours = defaultdict(set)
    # Iterate over each triplet of point indexes which denotes a
    # triangle within the tessellation.
    for index_triplet in tessellation.simplices:
        for index in index_triplet:
            connected = set(index_triplet)
            connected.remove(index)  # Do not allow connection to itself.
            triangle_neighbours[index] = triangle_neighbours[index].union(connected)
    # Initialise the nxn adjacency matrix with zeros.
    adjacency = np.zeros((len(points), len(points)))
    # Fill the adjacency matrix:
    for index in triangle_neighbours:
        neighbours = triangle_neighbours[index]
        neighbours = np.array(list(neighbours), dtype=int)
        kdtree = cKDTree(points[neighbours, :])
        nearby_neighbours = kdtree.query_ball_point(
            x=points[index],
            r=dthresh,
        )
        neighbours = neighbours[nearby_neighbours]
        adjacency[index, neighbours] = 1.0
        adjacency[neighbours, index] = 1.0
    # Return neighbours of each coordinate as an affinity (adjacency
    # in this case) matrix.
    return adjacency


def triangle_signed_area(triangle: ArrayLike) -> int:
    """Determine the signed area of a triangle.

    Args:
        triangle (ArrayLike):
            A 3x2 list of coordinates.

    Returns:
        int:
            The signed area of the triangle. It will be negative if the
            triangle has a clockwise winding, negative if the triangle
            has a counter-clockwise winding, and zero if the triangles
            points are collinear.

    """
    # Validate inputs
    triangle = np.asarray(triangle)
    if triangle.shape != (3, 2):
        raise ValueError("Input triangle must be a 3x2 array.")
    # Calculate the area of the triangle
    return 0.5 * (  # noqa: ECE001
        triangle[0, 0] * (triangle[1, 1] - triangle[2, 1])
        + triangle[1, 0] * (triangle[2, 1] - triangle[0, 1])
        + triangle[2, 0] * (triangle[0, 1] - triangle[1, 1])
    )


def edge_index_to_triangles(edge_index: ArrayLike) -> ArrayLike:
    """Convert an edged index to triangle simplices (triplets of coordinate indices).

    Args:
        edge_index (ArrayLike):
            An Nx2 array of edges.

    Returns:
        ArrayLike:
            An Nx3 array of triangles.

    Example:
        >>> points = np.random.rand(100, 2)
        >>> adjacency = delaunay_adjacency(points)
        >>> edge_index = affinity_to_edge_index(adjacency)
        >>> triangles = edge_index_to_triangles(edge_index)

    """
    # Validate inputs
    edge_index_shape = np.shape(edge_index)
    if edge_index_shape[0] != 2 or len(edge_index_shape) != 2:
        raise ValueError("Input edge_index must be a 2xM matrix.")
    nodes = np.unique(edge_index).tolist()
    neighbours = defaultdict(set)
    edges = edge_index.T.tolist()
    # Find the neighbours of each node
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    # Remove any nodes with less than two neighbours
    nodes = [node for node in nodes if len(neighbours[node]) >= 2]
    # Find the triangles
    triangles = set()
    for node in nodes:
        for neighbour in neighbours[node]:
            overlap = neighbours[node].intersection(neighbours[neighbour])
            while overlap:
                triangles.add(frozenset({node, neighbour, overlap.pop()}))
    return np.array([list(tri) for tri in triangles], dtype=np.int32, order="C")


def affinity_to_edge_index(
    affinity_matrix: Union[torch.Tensor, ArrayLike],
    threshold: Number = 0.5,
) -> Union[torch.tensor, ArrayLike]:
    """Convert an affinity matrix (similarity matrix) to an edge index.

    Converts an NxN affinity matrix to a 2xM edge index, where M is the
    number of node pairs with a similarity greater than the threshold
    value (defaults to 0.5).

    Args:
        affinity_matrix:
            An NxN matrix of affinities between nodes.
        threshold (Number):
            Threshold above which to be considered connected. Defaults
            to 0.5.

    Returns:
        ArrayLike or torch.Tensor:
            The edge index of shape (2, M).

    Example:
        >>> points = np.random.rand(100, 2)
        >>> adjacency = delaunay_adjacency(points)
        >>> edge_index = affinity_to_edge_index(adjacency)

    """
    # Validate inputs
    input_shape = np.shape(affinity_matrix)
    if len(input_shape) != 2 or len(np.unique(input_shape)) != 1:
        raise ValueError("Input affinity_matrix must be square (NxN).")
    # Handle cases for pytorch and numpy inputs
    if isinstance(affinity_matrix, torch.Tensor):
        return (affinity_matrix > threshold).nonzero().t().contiguous()
    return np.ascontiguousarray(
        np.stack((affinity_matrix > threshold).nonzero(), axis=1).T
    )


class SlideGraphConstructor:  # noqa: PIE798
    """Construct a graph using the SlideGraph+ (Liu et al. 2021) method.

    This uses a hybrid agglomerative clustering which uses a weighted
    combination of spatial distance (within the WSI) and feature-space
    distance to group patches into nodes. See the `build` function for
    more details on the graph construction method.

    """

    @staticmethod
    def _umap_reducer(graph: Dict[str, ArrayLike]) -> ArrayLike:
        """Default reduction which reduces `graph["x"]` to 3D values.

        Reduces graph features to 3D values using UMAP which are suitable
        for plotting as RGB values.

        Args:
            graph (dict):
                A graph with keys "x", "edge_index", and optionally
                "coordinates".
        Returns:
            ArrayLike:
                A UMAP embedding of `graph["x"]` with shape (N, 3) and
                values ranging from 0 to 1.
        """
        reducer = umap.UMAP(n_components=3)
        reduced = reducer.fit_transform(graph["x"])
        reduced -= reduced.min(axis=0)
        reduced /= reduced.max(axis=0)
        return reduced

    @staticmethod
    def build(
        points: ArrayLike,
        features: ArrayLike,
        lambda_d: Number = 3.0e-3,
        lambda_f: Number = 1.0e-3,
        lambda_h: Number = 0.8,
        connectivity_distance: Number = 4000,
        neighbour_search_radius: Number = 2000,
        feature_range_thresh: Optional[Number] = 1e-4,
    ) -> Dict[str, ArrayLike]:
        """Build a graph via hybrid clustering in spatial and feature space.

        The graph is constructed via hybrid hierarchical clustering
        followed by Delaunay triangulation of these cluster centroids.
        This is part of the SlideGraph pipeline but may be used to
        construct a graph in general from point coordinates and
        features.

        The clustering uses a distance kernel, ranging between 0 and 1,
        which is a weighted product of spatial distance (distance
        between coordinates in `points`, e.g. WSI location) and
        feature-space distance (e.g. ResNet features).

        Points which are spatially further apart than
        `neighbour_search_radius` are given a similarity of 1 (most
        dissimilar). This significantly speeds up computation. This
        distance metric is then used to form clusters via
        hierarchical/agglomerative clustering.

        Next, a Delaunay triangulation is applied to the clusters to
        connect the neighouring clusters. Only clusters which are closer
        than `connectivity_distance` in the spatial domain will be
        connected.

        Args:
            points (ArrayLike):
                A list of (x, y) spatial coordinates, e.g. pixel
                locations within a WSI.
            features (ArrayLike):
                A list of features associated with each coordinate in
                `points`. Must be the same length as `points`.
            lambda_d (Number):
                Spatial distance (d) weighting.
            lambda_f (Number):
                Feature distance (f) weighting.
            lambda_h (Number):
                Clustering distance threshold. Applied to the similarity
                kernel (1-fd). Ranges between 0 and 1. Defaults to 0.8.
                A good value for this parameter will depend on the
                intra-cluster variance.
            connectivity_distance (Number):
                Spatial distance threshold to consider points as
                connected during the Delaunay triangulation step.
            neighbour_search_radius (Number):
                Search radius (L2 norm) threshold for points to be
                considered as similar for clustering. Points with a
                spatial distance above this are not compared and have a
                similarity set to 1 (most dissimilar).
            feature_range_thresh (Number):
                Minimal range for which a feature is considered
                significant. Features which have a range less than this
                are ignored. Defaults to 1e-4. If falsy (None, False, 0,
                etc.), then no features are removed.

        Returns:
            dict:
                A dictionary defining a graph for serialisation (e.g.
                JSON or msgpack) or converting into a torch-geometric
                Data object where each node is the centroid (mean) of
                the features in a cluster.

                The dictionary has the following entries:

                - :class:`numpy.ndarray` - x:
                    Features of each node (mean of features in a
                    cluster). Required for torch-geometric Data.
                - :class:`numpy.ndarray` - edge_index:
                    Edge index matrix defining connectivity. Required
                    for torch-geometric Data.
                - :py:obj:`numpy.ndarray` - coords:
                    Coordinates of each node within the WSI (mean of
                    point in a cluster). Useful for visualisation over
                    the WSI.

        Example:
            >>> points = np.random.rand(99, 2) * 1000
            >>> features = np.array([
            ...     np.random.rand(11) * n
            ...     for n, _ in enumerate(points)
            ... ])
            >>> graph_dict = SlideGraphConstructor.build(points, features)

        """
        # Remove features which do not change significantly between patches
        if feature_range_thresh:
            feature_ranges = np.max(features, axis=0) - np.min(features, axis=0)
            where_significant = feature_ranges > feature_range_thresh
            features = features[:, where_significant]

        # Build a kd-tree and rank neighbours according to the euclidean
        # distance (nearest -> farthest).
        kd_tree = cKDTree(points)
        neighbour_distances_ckd, neighbour_indexes_ckd = kd_tree.query(
            x=points, k=len(points)
        )

        # Initialise an empty 1-D condensed distance matrix.
        # For information on condensed distance matrices see:
        # noqa - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist
        # noqa - https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        condensed_distance_matrix = np.zeros(int(len(points) * (len(points) - 1) / 2))

        # Find the similarity between pairs of patches
        index = 0
        for i in range(len(points) - 1):
            # Only consider neighbours which are inside the radius
            # (neighbour_search_radius).
            neighbour_distances_single_point = neighbour_distances_ckd[i][
                neighbour_distances_ckd[i] < neighbour_search_radius
            ]
            neighbour_indexes_single_point = neighbour_indexes_ckd[i][
                : len(neighbour_distances_single_point)
            ]

            # Called f in the paper
            neighbour_feature_similarities = np.exp(
                -lambda_f
                * np.linalg.norm(
                    features[i] - features[neighbour_indexes_single_point], axis=1
                )
            )
            # Called d in paper
            neighbour_distance_similarities = np.exp(
                -lambda_d * neighbour_distances_single_point
            )
            # 1 - product of similarities (1 - fd)
            # (1 = most un-similar 0 = most similar)
            neighbour_similarities = (
                1 - neighbour_feature_similarities * neighbour_distance_similarities
            )
            # Initialise similarity of coordinate i vs all coordinates to 1
            # (most un-similar).
            i_vs_all_similarities = np.ones(len(points))
            # Set the neighbours similarity to calculated values (similarity/fd)
            i_vs_all_similarities[
                neighbour_indexes_single_point
            ] = neighbour_similarities
            i_vs_all_similarities = i_vs_all_similarities[i + 1 :]
            condensed_distance_matrix[
                index : index + len(i_vs_all_similarities)
            ] = i_vs_all_similarities
            index = index + len(i_vs_all_similarities)

        # Perform hierarchical clustering (using similarity as distance)
        linkage_matrix = hierarchy.linkage(condensed_distance_matrix, method="average")
        clusters = hierarchy.fcluster(linkage_matrix, lambda_h, criterion="distance")

        # Finding the xy centroid and average features for each cluster
        unique_clusters = list(set(clusters))
        point_centroids = []
        feature_centroids = []
        for c in unique_clusters:
            (idx,) = np.where(clusters == c)
            # Find the xy and feature space averages of the cluster
            point_centroids.append(np.round(points[idx, :].mean(axis=0)))
            feature_centroids.append(features[idx, :].mean(axis=0))
        point_centroids = np.array(point_centroids)
        feature_centroids = np.array(feature_centroids)

        adjacency_matrix = delaunay_adjacency(
            points=point_centroids,
            dthresh=connectivity_distance,
        )
        edge_index = affinity_to_edge_index(adjacency_matrix)

        return {
            "x": feature_centroids,
            "edge_index": edge_index,
            "coordinates": point_centroids,
        }

    @classmethod
    def visualise(
        cls,
        graph: Dict[str, ArrayLike],
        color: Union[ArrayLike, str, Callable] = None,
        node_size: Union[Number, ArrayLike, Callable] = 25,
        edge_color: Union[str, ArrayLike] = (0, 0, 0, 0.33),
        ax: Axes = None,
    ) -> Axes:
        """Visualise a graph.

        The visualisation is a scatter plot of the graph nodes and the
        connections between them. By default, nodes are coloured
        according to the features of the graph via a UMAP embedding to
        the sRGB color space. This can be customised by passing a color
        argument which can be a single color, a list of colors, or a
        function which takes the graph and returns a list of colors for
        each node. The edge color(s) can be customised in the same way.

        Args:
            graph (dict):
                The graph to visualise as a dictionary with the following entries:

                - :class:`numpy.ndarray` - x:
                      Features of each node (mean of features
                      in a cluster). Required
                - :class:`numpy.ndarray` - edge_index:
                      Edge index matrix defining connectivity. Required
                - :class:`numpy.ndarray` - coordinates:
                      Coordinates of each node within the WSI (mean of point in a
                      cluster). Required
            color (np.array or str or callable):
                Colours of the nodes in the plot. If it is a callable,
                it should take a graph as input and return a numpy array
                of matplotlib colours. If `None` then a default function
                is used (UMAP on `graph["x"]`).
            node_size (int or np.ndarray or callable):
                Size of the nodes in the plot. If it is a function then
                it is called with the graph as an argument.
            edge_color (str):
                Colour of edges in the graph plot.
            ax (:class:`matplotlib.axes.Axes`):
                The axes which were plotted on.


        Returns:
            matplotlib.axes.Axes:
                The axes object to plot the graph on.

        Example:
            >>> points = np.random.rand(99, 2) * 1000
            >>> features = np.array([
            ...     np.random.rand(11) * n
            ...     for n, _ in enumerate(points)
            ... ])
            >>> graph_dict = SlideGraphConstructor.build(points, features)
            >>> fig, ax = plt.subplots()
            >>> slide_dims = wsi.info.slide_dimensions
            >>> ax.imshow(wsi.get_thumbnail(), extent=(0, *slide_dims, 0))
            >>> SlideGraphConstructor.visualise(graph_dict, ax=ax)
            >>> plt.show()

        """
        from matplotlib import collections as mc

        # Check that the graph is valid
        if "x" not in graph:
            raise ValueError("Graph must contain key `x`.")
        if "edge_index" not in graph:
            raise ValueError("Graph must contain key `edge_index`.")
        if "coordinates" not in graph:
            raise ValueError("Graph must contain key `coordinates`")
        if ax is None:
            _, ax = plt.subplots()
        if color is None:
            color = cls._umap_reducer

        nodes = graph["coordinates"]
        edges = graph["edge_index"]

        # Plot the edges
        line_segments = nodes[edges.T]
        edge_collection = mc.LineCollection(
            line_segments, colors=edge_color, linewidths=1
        )
        ax.add_collection(edge_collection)

        # Plot the nodes
        plt.scatter(
            *nodes.T,
            c=color(graph) if isinstance(color, Callable) else color,
            s=node_size(graph) if isinstance(node_size, Callable) else node_size,
            zorder=2,
        )

        return ax
