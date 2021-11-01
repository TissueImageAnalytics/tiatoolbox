import json
import pickle
import random
import sqlite3
from itertools import repeat
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from shapely import affinity
from shapely.geometry import Polygon

from tiatoolbox.annotation.storage import (
    AnnotationStoreABC,
    DictionaryStore,
    SQLiteStore,
)

sqlite3.enable_callback_tracebacks(True)

# Constants

GRID_SIZE = (10, 10)
FILLED_LEN = GRID_SIZE[0] * GRID_SIZE[1]

# Helper Functions


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
    return affinity.rotate(polygon, angle, origin="centroid")


def sample_predicate(props: Dict[str, Any]) -> bool:
    """Simple example predicate function for tests."""
    return props.get("class") == 123


# Fixtures


@pytest.fixture(scope="session")
def cell_grid() -> List[Polygon]:
    """Generate a 100x100 grid of fake cell boundary polygon annotations."""
    return [cell_polygon((i * 25, j * 25)) for i, j in np.ndindex(*GRID_SIZE)]


@pytest.fixture(scope="session")
def sample_triangle() -> Polygon:
    """Simple traingle polygon used for testing."""
    return Polygon([(0, 0), (1, 1), (2, 0)])


@pytest.fixture()
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


# Generate Parameterized Tests


def pytest_generate_tests(metafunc):
    """Generate (parameterize) test scenarios.

    Adapted from pytest documentation. For more information on
    parameterized tests see:
    https://docs.pytest.org/en/6.2.x/example/parametrize.html#a-quick-port-of-testscenarios
    """
    # Return if the test is not part of a class
    if metafunc.cls is None:
        return
    idlist = []
    argvalues = []
    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append([x[1] for x in items])
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


# Class Specific Tests


def test_sqlite_compile_options():
    options = SQLiteStore.compile_options()
    assert all(isinstance(x, str) for x in options)


def test_sqlite_compile_options_exception(monkeypatch):
    monkeypatch.setattr(SQLiteStore, "compile_options", lambda x: [], raising=True)
    with pytest.raises(Exception, match="RTREE and JSON1"):
        SQLiteStore()


def test_sqlite_multiple_connection(tmp_path):
    store = SQLiteStore(tmp_path / "annotations.db")
    store2 = SQLiteStore(tmp_path / "annotations.db")
    assert len(store) == len(store2)


# Annotation Store Interface Tests (AnnotationStoreABC)


class TestStore:
    scenarios = [
        ("Dictionary", {"store": DictionaryStore}),
        ("SQLite", {"store": SQLiteStore}),
    ]

    @staticmethod
    def test_open_close(fill_store, tmp_path, store):
        path = tmp_path / "polygons"
        indexes, store = fill_store(store, path)
        store.close()
        store2 = store.open(path)
        assert len(store2) == len(indexes)

    @staticmethod
    def test_append_many(cell_grid, tmp_path, store):
        store = store(tmp_path / "polygons")
        indexes = store.append_many(
            cell_grid,
            ({"class": random.randint(0, 6)} for _ in cell_grid),
        )
        assert len(indexes) == len(cell_grid)

    @staticmethod
    def test_query_bbox(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        results = store.query((0, 0, 25, 25))
        assert len(results) == 4

    @staticmethod
    def test_iquery_bbox(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        results = store.iquery((0, 0, 25, 25))
        assert len(results) == 4
        assert all(isinstance(index, str) for index in results)

    @staticmethod
    def test_iquery_polygon(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        results = store.iquery(Polygon([(0, 0), (0, 25), (1, 1), (25, 0)]))
        assert len(results) == 3
        assert all(isinstance(index, str) for index in results)

    @staticmethod
    def test_update(fill_store, tmp_path, store):
        indexes, store = fill_store(store, tmp_path / "polygon.db")
        index = indexes[0]
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        # Geometry update
        store.update(index, {"geometry": new_geometry})
        assert store[index][0] == new_geometry
        # Properties update
        store.update(index, {"abc": 123})
        assert store[index][1]["abc"] == 123

    @staticmethod
    def test_update_many(fill_store, tmp_path, store):
        indexes, store = fill_store(store, tmp_path / "polygon.db")
        # Geometry update
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        store.update_many(indexes, repeat({"geometry": new_geometry}))
        # Properties update
        store.update_many(indexes, repeat({"abc": 123}))

        for _, index in enumerate(indexes[:10]):
            assert store[index][0] == new_geometry
            assert store[index][1]["abc"] == 123

    @staticmethod
    def test_keys(fill_store, tmp_path, store):
        indexes, store = fill_store(store, tmp_path / "polygon.db")
        indexes = list(indexes)
        assert len(list(store.keys())) == len(indexes)
        assert isinstance(list(store.keys())[0], type(indexes[0]))

    @staticmethod
    def test_remove(fill_store, tmp_path, store):
        indexes, store = fill_store(store, tmp_path / "polygon.db")
        index = indexes[0]
        store.remove(index)
        assert len(store) == FILLED_LEN - 1

    @staticmethod
    def test_delitem(fill_store, tmp_path, store):
        indexes, store = fill_store(store, tmp_path / "polygon.db")
        index = indexes[0]
        del store[index]
        assert len(store) == FILLED_LEN - 1

    @staticmethod
    def test_remove_many(fill_store, tmp_path, store):
        indexes, store = fill_store(store, tmp_path / "polygon.db")
        store.remove_many(indexes)
        assert len(store) == 0

    @staticmethod
    def test_len(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        assert len(store) == FILLED_LEN

    @staticmethod
    def test_contains(fill_store, tmp_path, store):
        indexes, store = fill_store(store, tmp_path / "polygon.db")
        for index in indexes:
            assert index in store

    @staticmethod
    def test_iter(fill_store, tmp_path, store):
        indexes, store = fill_store(store, tmp_path / "polygon.db")
        for index in store:
            assert index in indexes

    @staticmethod
    def test_getitem(fill_store, tmp_path, sample_triangle, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        index = store.append(sample_triangle)
        geometry, properties = store[index]
        assert geometry == sample_triangle
        assert properties == {}

    @staticmethod
    def test_setitem(fill_store, tmp_path, sample_triangle, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        index = store.append(sample_triangle)
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        new_properties = {"abc": 123}
        store[index] = (new_geometry, new_properties)
        assert store[index] == (new_geometry, new_properties)

    @staticmethod
    def test_getitem_setitem_cycle(fill_store, tmp_path, sample_triangle, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        index = store.append(sample_triangle, {"class": 0})
        geometry, properties = store[index]
        store[index] = (geometry, properties)
        assert store[index] == (geometry, properties)

    @staticmethod
    def test_from_dataframe(cell_grid, store):
        df = pd.DataFrame.from_records(
            [
                {
                    "geometry": cell,
                    "row_id": n,
                }
                for n, cell in enumerate(cell_grid)
            ]
        )
        store = store.from_dataframe(df)
        keys = list(store.keys())
        _, properties = store[keys[0]]
        assert "row_id" in properties

    @staticmethod
    def test_to_dataframe(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        df = store.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == FILLED_LEN
        assert "geometry" in df.columns
        assert df.index.name == "key"
        assert isinstance(df.geometry.iloc[0], Polygon)

    @staticmethod
    def test_features(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        features = store.features()
        assert isinstance(features, Generator)
        features = list(features)
        assert len(features) == FILLED_LEN
        assert isinstance(features[0], dict)
        assert all(
            {"type", "geometry", "properties"} == set(f.keys()) for f in features
        )

    @staticmethod
    def test_to_geodict(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        geodict = store.to_geodict()
        assert isinstance(geodict, dict)
        assert "features" in geodict
        assert "type" in geodict
        assert geodict["type"] == "FeatureCollection"
        assert geodict["features"] == list(store.features())

    @staticmethod
    def test_from_geojson_str(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        geojson = store.to_geojson()
        store2 = store.from_geojson(geojson)
        assert len(store) == len(store2)

    @staticmethod
    def test_from_geojson_file(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        store.to_geojson(tmp_path / "polygon.json")
        with open(tmp_path / "polygon.json", "r") as file_handle:
            store2 = store.from_geojson(file_handle)
        assert len(store) == len(store2)

    @staticmethod
    def test_from_geojson_path(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        store.to_geojson(tmp_path / "polygon.json")
        store2 = store.from_geojson(tmp_path / "polygon.json")
        assert len(store) == len(store2)

    @staticmethod
    def test_to_geojson_str(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        geojson = store.to_geojson()
        assert isinstance(geojson, str)
        geodict = json.loads(geojson)
        assert "features" in geodict
        assert "type" in geodict
        assert geodict["type"] == "FeatureCollection"
        assert len(geodict["features"]) == len(list(store.features()))

    @staticmethod
    def test_to_geojson_file(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        with open(tmp_path / "polygon.json", "w") as fh:
            geojson = store.to_geojson(fp=fh)
        assert geojson is None
        with open(tmp_path / "polygon.json", "r") as fh:
            geodict = json.load(fh)
        assert "features" in geodict
        assert "type" in geodict
        assert geodict["type"] == "FeatureCollection"
        assert len(geodict["features"]) == len(list(store.features()))

    @staticmethod
    def test_to_geojson_path(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        geojson = store.to_geojson(fp=tmp_path / "polygon.json")
        assert geojson is None
        with open(tmp_path / "polygon.json", "r") as fh:
            geodict = json.load(fh)
        assert "features" in geodict
        assert "type" in geodict
        assert geodict["type"] == "FeatureCollection"
        assert len(geodict["features"]) == len(list(store.features()))

    @staticmethod
    def test_to_ldjson_str(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        ldjson = store.to_ldjson()
        for line in ldjson.split():
            assert isinstance(line, str)
            feature = json.loads(line)
            assert "type" in feature
            assert "geometry" in feature
            assert "properties" in feature

    @staticmethod
    def test_to_ldjson_file(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        with open(tmp_path / "polygon.ldjson", "w") as fh:
            ldjson = store.to_ldjson(fp=fh)
        assert ldjson is None
        with open(tmp_path / "polygon.ldjson", "r") as fh:
            for line in fh.readlines():
                assert isinstance(line, str)
                feature = json.loads(line)
                assert "type" in feature
                assert "geometry" in feature
                assert "properties" in feature

    @staticmethod
    def test_to_ldjson_path(fill_store, tmp_path, store):
        _, store = fill_store(store, tmp_path / "polygon.db")
        ldjson = store.to_ldjson(fp=tmp_path / "polygon.ldjson")
        assert ldjson is None
        with open(tmp_path / "polygon.ldjson", "r") as fh:
            for line in fh.readlines():
                assert isinstance(line, str)
                feature = json.loads(line)
                assert "type" in feature
                assert "geometry" in feature
                assert "properties" in feature

    @staticmethod
    def test_dump(fill_store, tmp_path, store):
        _, store = fill_store(store, ":memory:")
        store.dump(tmp_path / "dump_test.db")
        assert (tmp_path / "dump_test.db").stat().st_size > 0

    @staticmethod
    def test_dump_file_handle(fill_store, tmp_path, store):
        _, store = fill_store(store, ":memory:")
        with open(tmp_path / "dump_test.db", "w") as fh:
            store.dump(fh)
        assert (tmp_path / "dump_test.db").stat().st_size > 0

    @staticmethod
    def test_dumps(fill_store, store):
        _, store = fill_store(store, ":memory:")
        string = store.dumps()
        assert isinstance(string, str)

    @staticmethod
    def test_iquery_predicate_str(fill_store, store):
        keys, store = fill_store(store, ":memory:")
        store.update(keys[0], {"class": 123})
        results = store.iquery(
            (0, 0, 1024, 1024), properties_predicate="props.get('class') == 123"
        )
        assert len(results) == 1
        assert results[0] == keys[0]

    @staticmethod
    def test_iquery_predicate_callable(fill_store, store):
        keys, store = fill_store(store, ":memory:")
        store.update(keys[0], {"class": 123})
        results = store.iquery(
            (0, 0, 1024, 1024),
            properties_predicate=lambda props: props.get("class") == 123,
        )
        assert len(results) == 1
        assert results[0] == keys[0]

    @staticmethod
    def test_iquery_predicate_pickle(fill_store, store):
        keys, store = fill_store(store, ":memory:")
        store.update(keys[0], {"class": 123})

        results = store.query(
            (0, 0, 1024, 1024), properties_predicate=pickle.dumps(sample_predicate)
        )
        assert len(results) == 1

    @staticmethod
    def test_query_predicate_str(fill_store, store):
        keys, store = fill_store(store, ":memory:")
        store.update(keys[0], {"class": 123})
        results = store.query(
            (0, 0, 1024, 1024), properties_predicate="props.get('class') == 123"
        )
        assert len(results) == 1

    @staticmethod
    def test_query_predicate_callable(fill_store, store):
        keys, store = fill_store(store, ":memory:")
        store.update(keys[0], {"class": 123})
        results = store.iquery(
            (0, 0, 1024, 1024),
            properties_predicate=lambda props: props.get("class") == 123,
        )
        assert len(results) == 1

    @staticmethod
    def test_query_predicate_pickle(fill_store, store):
        keys, store = fill_store(store, ":memory:")
        store.update(keys[0], {"class": 123})

        results = store.query(
            (0, 0, 1024, 1024), properties_predicate=pickle.dumps(sample_predicate)
        )
        assert len(results) == 1
