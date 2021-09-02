from pathlib import Path
from typing import Tuple, Union, List
from numbers import Number

import pytest
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely import affinity

from tiatoolbox.annotation.storage import (
    AnnotationStoreABC,
    PyTablesStore,
    SQLite3RTreeStore,
)

GRID_SIZE = (10, 10)
FILLED_LEN = GRID_SIZE[0] * GRID_SIZE[1]


def cell_polygon(
    xy: Tuple[Number, Number],
    n_points: int = 20,
    radius: Number = 10,
    noise: Number = 1,
    eccentricity: Tuple[Number, Number] = (1, 3),
    repeat_first: bool = True,
    direction: str = "CCW",
) -> Polygon:
    """Generate a fake cell boundary polygon.

    Cell boundaries are generated an ellipsoids with randomised eccentricity,
    added noise, and a random rotation.

    Args:
        xy (tuple(int)): The x,y centre point to generate the cell boundary around.
        n_points (int): Number of points in the boundary. Defaults to 20.
        radius (float): Radius of the points from the centre. Defaults to 10.
        noise (float): Noise to add to the point locations. Defaults to 1.
        eccentricity (tuple(float)): Range of values (low, high) to use for
            randomised eccentricity. Defaults to (1, 3).
        repeat_first (bool): Enforce that the last point is equal to the first.
        direction (str): Ordering of the points. Defaults to "CCW". Valid options
            are: counter-clockwise "CCW", and clockwise "CW".
    """
    if repeat_first:
        n_points -= 1

    # Generate points about an ellipse with random eccentricity
    x, y = xy
    alpha = np.linspace(0, 2 * np.pi - (2 * np.pi / n_points), n_points)
    rx = radius * (np.random.rand() + 0.5)
    ry = np.random.uniform(*eccentricity) * radius - rx
    x = rx * np.cos(alpha) + x + (np.random.rand(n_points) - 0.5) * noise
    y = ry * np.sin(alpha) + y + (np.random.rand(n_points) - 0.5) * noise
    boundary_coords = np.stack([x, y], axis=1).astype(int).tolist()

    # Copy first coordinate to the end if required
    if repeat_first:
        boundary_coords = boundary_coords + [boundary_coords[0]]

    # Swap direction
    if direction.strip().lower() == "cw":
        boundary_coords = boundary_coords[::-1]

    polygon = Polygon(boundary_coords)

    # Add random rotation
    angle = np.random.rand() * 360
    polygon = affinity.rotate(polygon, angle, origin="centroid")
    return polygon


@pytest.fixture(scope="session")
def cell_grid() -> List[Polygon]:
    """Generate a 100x100 grid of fake cell boundary polygon annotations."""
    return [cell_polygon((i * 25, j * 25)) for i, j in np.ndindex(*GRID_SIZE)]


@pytest.fixture(scope="session")
def sample_triangle() -> Polygon:
    """Simple traingle polygon used for testing."""
    return Polygon([(0, 0), (1, 1), (2, 0)])


@pytest.fixture
def fill_store(cell_grid):
    """Factory fixture to cache and return filled stores."""

    def _fill_store(
        store_class: AnnotationStoreABC,
        path: Union[str, Path],
    ):
        store = store_class(path)
        indexes = store.append_many(cell_grid)
        return indexes, store

    return _fill_store


def test_SQLite3RTreeStore_append_many(cell_grid, tmp_path):
    store = SQLite3RTreeStore(tmp_path / "polygons.db")
    indexes = store.append_many(
        cell_grid, ({"class": x} for x in np.random.randint(0, 7, len(cell_grid)))
    )
    assert len(indexes) == len(cell_grid)


def test_PyTablesStore_append_many(cell_grid, tmp_path):
    store = PyTablesStore(tmp_path / "polygons.h5")
    indexes = store.append_many(
        cell_grid, ({"class": x} for x in np.random.randint(0, 7, len(cell_grid)))
    )
    assert len(indexes) == len(cell_grid)


def test_SQLite3RTreeStore_update(fill_store, tmp_path):
    indexes, store = fill_store(SQLite3RTreeStore, tmp_path / "polygon.db")
    index = indexes[0]
    new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
    # Geometry update
    store.update(index, {"geometry": new_geometry})
    assert store[index][0] == new_geometry
    # Properties update
    store.update(index, {"abc": 123})
    assert store[index][1]["abc"] == 123


def test_SQLite3RTreeStore_remove(fill_store, tmp_path):
    indexes, store = fill_store(SQLite3RTreeStore, tmp_path / "polygon.db")
    index = indexes[0]
    store.remove(index)
    assert len(store) == FILLED_LEN - 1


def test_SQLite3RTreeStore_remove_many(fill_store, tmp_path):
    indexes, store = fill_store(SQLite3RTreeStore, tmp_path / "polygon.db")
    store.remove_many(indexes)
    assert len(store) == 0


def test_PytablesStore_len(fill_store, tmp_path):
    _, store = fill_store(PyTablesStore, tmp_path / "polygon.h5")
    assert len(store) == FILLED_LEN


def test_SQLite3RTreeStore_len(fill_store, tmp_path):
    _, store = fill_store(SQLite3RTreeStore, tmp_path / "polygon.db")
    assert len(store) == FILLED_LEN


def test_PytablesStore_getitem(fill_store, tmp_path, sample_triangle):
    _, store = fill_store(PyTablesStore, tmp_path / "polygon.h5")
    index = store.append(sample_triangle)
    geometry, properties = store[index]
    assert geometry == sample_triangle
    assert properties == {"class": -1}


def test_SQLite3RTreeStore_getitem(fill_store, tmp_path, sample_triangle):
    _, store = fill_store(SQLite3RTreeStore, tmp_path / "polygon.db")
    index = store.append(sample_triangle)
    geometry, properties = store[index]
    assert geometry == sample_triangle
    assert properties == {}


def test_SQLite3RTreeStore_setitem(fill_store, tmp_path, sample_triangle):
    _, store = fill_store(SQLite3RTreeStore, tmp_path / "polygon.db")
    index = store.append(sample_triangle)
    new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
    new_properties = {"abc": 123}
    store[index] = (new_geometry, new_properties)
    assert store[index] == (new_geometry, new_properties)


def test_SQLite3RTreeStore_delitem(fill_store, tmp_path, sample_triangle):
    _, store = fill_store(SQLite3RTreeStore, tmp_path / "polygon.db")
    index = store.append(sample_triangle)
    del store[index]
    assert len(store) == FILLED_LEN


def test_SQLite3RTreeStore_getitem_setitem_cycle(fill_store, tmp_path, sample_triangle):
    _, store = fill_store(SQLite3RTreeStore, tmp_path / "polygon.db")
    index = store.append(sample_triangle, {"class": 0})
    geometry, properties = store[index]
    store[index] = (geometry, properties)
    assert store[index] == (geometry, properties)


def test_SQLite3RTreeStore_to_dataframe(fill_store, tmp_path):
    _, store = fill_store(SQLite3RTreeStore, tmp_path / "polygon.db")
    df = store.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == FILLED_LEN
    assert "geometry" in df.columns
    assert df.index.name == "index"
    assert isinstance(df.geometry.iloc[0], Polygon)


def test_SQLite3RTreeStore_to_features(fill_store, tmp_path):
    _, store = fill_store(SQLite3RTreeStore, tmp_path / "polygon.db")
    features = store.to_features()
    assert isinstance(features, list)
    assert len(features) == FILLED_LEN
    assert isinstance(features[0], dict)
    assert all({"type", "geometry", "properties"} == set(f.keys()) for f in features)


def test_SQLite3RTreeStore_to_geodict(fill_store, tmp_path):
    _, store = fill_store(SQLite3RTreeStore, tmp_path / "polygon.db")
    geodict = store.to_geodict()
    assert isinstance(geodict, dict)
    assert "features" in geodict
    assert "type" in geodict
    assert geodict["type"] == "FeatureCollection"
    assert geodict["features"] == store.to_features()


def test_SQLite3RTreeStore_compile_options():
    options = SQLite3RTreeStore.compile_options()
    assert all(isinstance(x, str) for x in options)
