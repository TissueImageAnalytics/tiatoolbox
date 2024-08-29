"""Construction and visualisation of graphs for WSI prediction."""

from __future__ import annotations

from collections import defaultdict
from numbers import Number
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
import torch
import umap
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import Delaunay, cKDTree

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike


def delaunay_adjacency(points: np.ndarray, dthresh: float) -> np.ndarray:
    """Create an adjacency matrix via Delaunay triangulation from a list of coordinates.

    Points which are further apart than dthresh will not be connected.

    See https://en.wikipedia.org/wiki/Adjacency_matrix.

    Args:
        points (np.ndarray):
            An nxm list of coordinates.
        dthresh (float):
            Distance threshold for triangulation.

    Returns:
        ArrayLike:
            Adjacency matrix of shape NxN where 1 indicates connected
            and 0 indicates unconnected.

    Example:
        >>> rng = np.random.default_rng()
        >>> points = rng.random((100, 2))
        >>> adjacency = delaunay_adjacency(points)

    """
    # Validate inputs
    if not isinstance(dthresh, Number):
        msg = "dthresh must be a number."
        raise TypeError(msg)
    if len(points) < 4:  # noqa: PLR2004
        msg = "Points must have length >= 4."
        raise ValueError(msg)
    if len(np.shape(points)) != 2:  # noqa: PLR2004
        msg = "Points must have an NxM shape."
        raise ValueError(msg)
    # Apply Delaunay triangulation to the coordinates to get a
    # tessellation of triangles.
    tessellation = Delaunay(points)
    # Find all connected neighbours for each point in the set of
    # triangles. Starting with an empty dictionary.
    triangle_neighbours: defaultdict
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
        msg = "Input triangle must be a 3x2 array."
        raise ValueError(msg)
    # Calculate the area of the triangle
    return 0.5 * (
        triangle[0, 0] * (triangle[1, 1] - triangle[2, 1])
        + triangle[1, 0] * (triangle[2, 1] - triangle[0, 1])
        + triangle[2, 0] * (triangle[0, 1] - triangle[1, 1])
    )


def edge_index_to_triangles(edge_index: np.ndarray) -> np.ndarray:
    """Convert an edged index to triangle simplices (triplets of coordinate indices).

    Args:
        edge_index (np.ndarray):
            An Nx2 array of edges.

    Returns:
        ArrayLike:
            An Nx3 array of triangles.

    Example:
        >>> rng = np.random.default_rng()
        >>> points = rng.random((100, 2))
        >>> adjacency = delaunay_adjacency(points)
        >>> edge_index = affinity_to_edge_index(adjacency)
        >>> triangles = edge_index_to_triangles(edge_index)

    """
    # Validate inputs
    edge_index_shape = np.shape(edge_index)
    if edge_index_shape[0] != 2 or len(edge_index_shape) != 2:  # noqa: PLR2004
        msg = "Input edge_index must be a 2xM matrix."
        raise ValueError(msg)
    nodes = np.unique(edge_index).tolist()
    neighbours = defaultdict(set)
    edges = edge_index.T.tolist()
    # Find the neighbours of each node
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    # Remove any nodes with less than two neighbours
    nodes = [node for node in nodes if len(neighbours[node]) >= 2]  # noqa: PLR2004
    # Find the triangles
    triangles = set()
    for node in nodes:
        for neighbour in neighbours[node]:
            overlap = neighbours[node].intersection(neighbours[neighbour])
            while overlap:
                triangles.add(frozenset({node, neighbour, overlap.pop()}))
    return np.array([list(tri) for tri in triangles], dtype=np.int32, order="C")


def affinity_to_edge_index(
    affinity_matrix: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> torch.Tensor | np.ndarray:
    """Convert an affinity matrix (similarity matrix) to an edge index.

    Converts an NxN affinity matrix to a 2xM edge index, where M is the
    number of node pairs with a similarity greater than the threshold
    value (defaults to 0.5).

    Args:
        affinity_matrix (torch.Tensor | np.ndarray):
            An NxN matrix of affinities between nodes.
        threshold (Number):
            Threshold above which to be considered connected. Defaults
            to 0.5.

    Returns:
        torch.Tensor | np.ndarray:
            The edge index of shape (2, M).

    Example:
        >>> rng = np.random.default_rng()
        >>> points = rng.random((100, 2))
        >>> adjacency = delaunay_adjacency(points)
        >>> edge_index = affinity_to_edge_index(adjacency)

    """
    # Validate inputs
    input_shape = np.shape(affinity_matrix)
    if len(input_shape) != 2 or len(np.unique(input_shape)) != 1:  # noqa: PLR2004
        msg = "Input affinity_matrix must be square (NxN)."
        raise ValueError(msg)
    # Handle cases for pytorch and numpy inputs
    if isinstance(affinity_matrix, torch.Tensor):
        return (affinity_matrix > threshold).nonzero().t().contiguous().to(torch.int64)
    return np.ascontiguousarray(
        np.stack((affinity_matrix > threshold).nonzero(), axis=1).T.astype(np.int64),
    )


class SlideGraphConstructor:
    """Construct a graph using the SlideGraph+ (Liu et al. 2021) method.

    This uses a hybrid agglomerative clustering which uses a weighted
    combination of spatial distance (within the WSI) and feature-space
    distance to group patches into nodes. See the `build` function for
    more details on the graph construction method.

    """

    @staticmethod
    def _umap_reducer(graph: dict[str, np.ndarray]) -> np.ndarray:
        """Default reduction which reduces `graph["x"]` to 3D values.

        Reduces graph features to 3D values using UMAP which are suitable
        for plotting as RGB values.

        Args:
            graph (dict):
                A graph with keys "x", "edge_index", and optionally
                "coordinates".

        Returns:
            np.ndarray:
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
        points: np.ndarray,
        features: np.ndarray,
        lambda_d: float = 3.0e-3,
        lambda_f: float = 1.0e-3,
        lambda_h: float = 0.8,
        connectivity_distance: int = 4000,
        neighbour_search_radius: int = 2000,
        feature_range_thresh: float | None = 1e-4,
    ) -> dict[str, np.ndarray]:
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
            points (np.ndarray):
                A list of (x, y) spatial coordinates, e.g. pixel
                locations within a WSI.
            features (np.ndarray):
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
            >>> rng = np.random.default_rng()
            >>> points = rng.random((99, 2)) * 1000
            >>> features = np.array([
            ...     rng.random(11) * n
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
            x=points,
            k=len(points),
        )

        # Initialise an empty 1-D condensed distance matrix.
        # For information on condensed distance matrices see:
        # - scipy.spatial.distance.pdist
        # - scipy.cluster.hierarchy.linkage
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
                    features[i] - features[neighbour_indexes_single_point],
                    axis=1,
                ),
            )
            # Called d in paper
            neighbour_distance_similarities = np.exp(
                -lambda_d * neighbour_distances_single_point,
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
            i_vs_all_similarities[neighbour_indexes_single_point] = (
                neighbour_similarities
            )
            i_vs_all_similarities = i_vs_all_similarities[i + 1 :]
            condensed_distance_matrix[index : index + len(i_vs_all_similarities)] = (
                i_vs_all_similarities
            )
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
        point_centroids_arr = np.array(point_centroids)
        feature_centroids_arr = np.array(feature_centroids)

        adjacency_matrix = delaunay_adjacency(
            points=point_centroids_arr,
            dthresh=connectivity_distance,
        )
        edge_index = affinity_to_edge_index(adjacency_matrix)
        edge_index = cast(np.ndarray, edge_index)
        return {
            "x": feature_centroids_arr,
            "edge_index": edge_index,
            "coordinates": point_centroids_arr,
        }

    @classmethod
    def visualise(
        cls: type[SlideGraphConstructor],
        graph: dict[str, np.ndarray],
        color: np.ndarray | str | Callable | None = None,
        node_size: int | np.ndarray | Callable = 25,
        edge_color: str | ArrayLike = (0, 0, 0, 0.33),
        ax: Axes | None = None,
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
            >>> rng = np.random.default_rng()
            >>> points = rng.random((99, 2)) * 1000
            >>> features = np.array([
            ...     rng.random(11) * n
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
            msg = "Graph must contain key `x`."
            raise ValueError(msg)
        if "edge_index" not in graph:
            msg = "Graph must contain key `edge_index`."
            raise ValueError(msg)
        if "coordinates" not in graph:
            msg = "Graph must contain key `coordinates`"
            raise ValueError(msg)
        if ax is None:
            _, ax = plt.subplots()
        if color is None:
            color = cls._umap_reducer

        nodes = graph["coordinates"]
        edges = graph["edge_index"]

        # Plot the edges
        line_segments = nodes[edges.T]
        edge_collection = mc.LineCollection(
            line_segments,
            colors=edge_color,
            linewidths=1,
        )
        ax.add_collection(edge_collection)

        # Plot the nodes
        plt.scatter(
            x=nodes.T[0],
            y=nodes.T[1],
            c=color(graph) if callable(color) else color,
            s=node_size(graph) if callable(node_size) else node_size,
            zorder=2,
        )

        return ax
