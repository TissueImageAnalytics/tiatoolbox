import json
import pickle
import random
import sqlite3
from itertools import repeat
from numbers import Number
from pathlib import Path
from timeit import timeit
from typing import Any, Dict, Generator, List, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from tiatoolbox.annotation.storage import (
    Annotation,
    AnnotationStore,
    DictionaryStore,
    SQLiteStore,
)

sqlite3.enable_callback_tracebacks(True)

# Constants

GRID_SIZE = (10, 10)
FILLED_LEN = 2 * (GRID_SIZE[0] * GRID_SIZE[1])

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
    """Generate a grid of fake cell boundary polygon annotations."""
    return [cell_polygon((i * 25, j * 25)) for i, j in np.ndindex(*GRID_SIZE)]


@pytest.fixture(scope="session")
def points_grid() -> List[Polygon]:
    """Generate a grid of fake point annotations."""
    return [Point((i * 25, j * 25)) for i, j in np.ndindex(*GRID_SIZE)]


@pytest.fixture(scope="session")
def sample_triangle() -> Polygon:
    """Simple traingle polygon used for testing."""
    return Polygon([(0, 0), (1, 1), (2, 0)])


@pytest.fixture()
def fill_store(cell_grid, points_grid):
    """Factory fixture to fill stores with test data."""

    def _fill_store(
        store_class: AnnotationStore,
        path: Union[str, Path],
    ):
        store = store_class(path)
        annotations = [Annotation(cell) for cell in cell_grid] + [
            Annotation(point) for point in points_grid
        ]
        keys = store.append_many(annotations)
        return keys, store

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


def test_sqlitestore_compile_options():
    options = SQLiteStore.compile_options()
    assert all(isinstance(x, str) for x in options)


def test_sqlitestore_compile_options_exception(monkeypatch):
    monkeypatch.setattr(SQLiteStore, "compile_options", lambda x: [], raising=True)
    with pytest.raises(Exception, match="RTREE and JSON1"):
        SQLiteStore()


def test_sqlitestore_multiple_connection(tmp_path):
    store = SQLiteStore(tmp_path / "annotations.db")
    store2 = SQLiteStore(tmp_path / "annotations.db")
    assert len(store) == len(store2)


def test_sqlitestore_index_error():
    """Test adding an index of invalid type."""
    store = SQLiteStore()
    with pytest.raises(TypeError, match="where"):
        store.create_index("foo", lambda g, p: "foo" in p)


def test_sqlitestore_index_str(fill_store, tmp_path):
    """Test that adding an index improves performance."""
    _, store = fill_store(SQLiteStore, tmp_path / "polygon.db")

    def query():
        """Test query."""
        return store.iquery((0, 0, 1e4, 1e4), where="props['class'] == 0")

    def alt_query():
        """Alternative query to avoid temporary caches giving unfair advantage."""
        return store.iquery((0, 0, 1e4, 1e4), where="has_key(props, 'foo')")

    # Do an initial bunch of queries to initialise any internal state or
    # caches.
    _ = timeit(query, number=10)
    # Time query without index
    t1 = timeit(query, number=50)
    # Create the index
    properties_predicate = "props['class']"
    store.create_index("test_index", properties_predicate)
    # Do another unrelated query
    _ = timeit(alt_query, number=10)
    # Time the original query with the index
    t2 = timeit(query, number=50)
    assert t2 < t1


def test_sqlitestore_unsupported_compression(sample_triangle):
    """Test that using an unsupported compression str raises error."""
    store = SQLiteStore(compression="foo")
    with pytest.raises(Exception, match="Unsupported"):
        _ = store.serialise_geometry(sample_triangle)


def test_sqlitestore_unsupported_decompression():
    """Test that using an unsupported decompression str raises error."""
    store = SQLiteStore(compression="foo")
    with pytest.raises(Exception, match="Unsupported"):
        _ = store.deserialise_geometry(bytes())


def test_sqlitestore_wkt_deserialisation(sample_triangle):
    """Test WKT deserialisation."""
    store = SQLiteStore(compression=None)
    wkt = sample_triangle.wkt
    geom = store.deserialise_geometry(wkt)
    assert geom == sample_triangle


def test_sqlitestore_metadata_get_keyerror():
    """Test getting a metadata entry that does not exists."""
    store = SQLiteStore()
    with pytest.raises(KeyError):
        store.metadata["foo"]


def test_sqlitestore_metadata_delete_keyerror():
    """Test deleting a metadata entry that does not exists."""
    store = SQLiteStore(compression=None)
    with pytest.raises(KeyError):
        del store.metadata["foo"]


def test_sqlitestore_metadata_delete():
    """Test adding and deleting a metadata entry."""
    store = SQLiteStore(compression=None)
    store.metadata["foo"] = 1
    assert "foo" in store.metadata
    del store.metadata["foo"]
    assert "foo" not in store.metadata


# Annotation Store Interface Tests (AnnotationStoreABC)


class TestStore:
    scenarios = [
        ("Dictionary", {"store": DictionaryStore}),
        ("SQLite", {"store": SQLiteStore}),
    ]

    @staticmethod
    def test_open_close(fill_store, tmp_path, store):
        """Test opening and closing a store."""
        path = tmp_path / "polygons"
        keys, store = fill_store(store, path)
        store.close()
        store2 = store.open(path)
        assert len(store2) == len(keys)

    @staticmethod
    def test_append_many(cell_grid, tmp_path, store):
        """Test bulk append of annotations."""
        store = store(tmp_path / "polygons")
        annotations = [
            Annotation(cell, {"class": random.randint(0, 6)}) for cell in cell_grid
        ]
        keys = store.append_many(annotations)
        assert len(keys) == len(cell_grid)

    @staticmethod
    def test_query_bbox(fill_store, tmp_path, store):
        """Test query with a bounding box."""
        _, store = fill_store(store, tmp_path / "polygon.db")
        results = store.query((0, 0, 25, 25))
        assert len(results) == 8

    @staticmethod
    def test_iquery_bbox(fill_store, tmp_path, store):
        """Test iquery with a bounding box."""
        _, store = fill_store(store, tmp_path / "polygon.db")
        results = store.iquery((0, 0, 25, 25))
        assert len(results) == 8
        assert all(isinstance(key, str) for key in results)

    @staticmethod
    def test_iquery_polygon(fill_store, tmp_path, store):
        """Test iquery with a non-rectangular geometry."""
        _, store = fill_store(store, tmp_path / "polygon.db")
        results = store.iquery(Polygon([(0, 0), (0, 25), (1, 1), (25, 0)]))
        assert len(results) == 6
        assert all(isinstance(key, str) for key in results)

    @staticmethod
    def test_pach(fill_store, tmp_path, store):
        """Test patching an annotation."""
        keys, store = fill_store(store, tmp_path / "polygon.db")
        key = keys[0]
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        # Geometry update
        store.patch(key, new_geometry)
        assert store[key].geometry == new_geometry
        # Properties update
        store.patch(key, properties={"abc": 123})
        assert store[key].properties["abc"] == 123

    @staticmethod
    def test_update_many(fill_store, tmp_path, store):
        """Test bulk update."""
        keys, store = fill_store(store, tmp_path / "polygon.db")
        # Geometry update
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        store.patch_many(keys, repeat(new_geometry, 10))
        # Properties update
        store.patch_many(keys, properties_iter=repeat({"abc": 123}, 10))

        for _, key in enumerate(keys[:10]):
            assert store[key].geometry == new_geometry
            assert store[key].properties["abc"] == 123

    @staticmethod
    def test_keys(fill_store, tmp_path, store):
        """Test getting an keys iterator."""
        keys, store = fill_store(store, tmp_path / "polygon.db")
        keys = list(keys)
        assert len(list(store.keys())) == len(keys)
        assert isinstance(list(store.keys())[0], type(keys[0]))

    @staticmethod
    def test_remove(fill_store, tmp_path, store):
        """Test removing an annotation."""
        keys, store = fill_store(store, tmp_path / "polygon.db")
        key = keys[0]
        store.remove(key)
        assert len(store) == FILLED_LEN - 1

    @staticmethod
    def test_delitem(fill_store, tmp_path, store):
        """Test using the delitem syntax."""
        keys, store = fill_store(store, tmp_path / "polygon.db")
        key = keys[0]
        del store[key]
        assert len(store) == FILLED_LEN - 1

    @staticmethod
    def test_remove_many(fill_store, tmp_path, store):
        """Test bulk deletion."""
        keys, store = fill_store(store, tmp_path / "polygon.db")
        store.remove_many(keys)
        assert len(store) == 0

    @staticmethod
    def test_len(fill_store, tmp_path, store):
        """Test finding the number of annotation via the len operator."""
        _, store = fill_store(store, tmp_path / "polygon.db")
        assert len(store) == FILLED_LEN

    @staticmethod
    def test_contains(fill_store, tmp_path, store):
        """Test using the contains (in) operator."""
        keys, store = fill_store(store, tmp_path / "polygon.db")
        for key in keys:
            assert key in store

    @staticmethod
    def test_iter(fill_store, tmp_path, store):
        """Test iterating over the store."""
        keys, store = fill_store(store, tmp_path / "polygon.db")
        for key in store:
            assert key in keys

    @staticmethod
    def test_getitem(fill_store, tmp_path, sample_triangle, store):
        """Test the getitem syntax."""
        _, store = fill_store(store, tmp_path / "polygon.db")
        key = store.append(Annotation(sample_triangle))
        annotation = store[key]
        assert annotation.geometry == sample_triangle
        assert annotation.properties == {}

    @staticmethod
    def test_setitem(fill_store, tmp_path, sample_triangle, store):
        """Test the setitem syntax."""
        _, store = fill_store(store, tmp_path / "polygon.db")
        key = store.append(Annotation(sample_triangle))
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        new_properties = {"abc": 123}
        store[key] = Annotation(new_geometry, new_properties)
        assert store[key] == Annotation(new_geometry, new_properties)

    @staticmethod
    def test_getitem_setitem_cycle(fill_store, tmp_path, sample_triangle, store):
        """Test getting an setting an annotation."""
        _, store = fill_store(store, tmp_path / "polygon.db")
        key = store.append(Annotation(sample_triangle, {"class": 0}))
        annotation = store[key]
        store[key] = annotation
        assert store[key] == annotation

    @staticmethod
    def test_from_dataframe(cell_grid, store):
        """Test loading from to a pandas dataframe."""
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
        annotation = store[keys[0]]
        assert "row_id" in annotation.properties

    @staticmethod
    def test_to_dataframe(fill_store, tmp_path, store):
        """Test converting to a pandas dataframe."""
        _, store = fill_store(store, tmp_path / "polygon.db")
        df = store.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == FILLED_LEN
        assert "geometry" in df.columns
        assert df.index.name == "key"
        assert isinstance(df.geometry.iloc[0], Polygon)

    @staticmethod
    def test_features(fill_store, tmp_path, store):
        """Test converting to a features dictionaries."""
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
        """Test converting to a geodict."""
        _, store = fill_store(store, tmp_path / "polygon.db")
        geodict = store.to_geodict()
        assert isinstance(geodict, dict)
        assert "features" in geodict
        assert "type" in geodict
        assert geodict["type"] == "FeatureCollection"
        assert geodict["features"] == list(store.features())

    @staticmethod
    def test_from_geojson_str(fill_store, tmp_path, store):
        """Test loading from geojson with a file path string."""
        _, store = fill_store(store, tmp_path / "polygon.db")
        geojson = store.to_geojson()
        store2 = store.from_geojson(geojson)
        assert len(store) == len(store2)

    @staticmethod
    def test_from_geojson_file(fill_store, tmp_path, store):
        """Test loading from geojson with a file handle."""
        _, store = fill_store(store, tmp_path / "polygon.db")
        store.to_geojson(tmp_path / "polygon.json")
        with open(tmp_path / "polygon.json", "r") as file_handle:
            store2 = store.from_geojson(file_handle)
        assert len(store) == len(store2)

    @staticmethod
    def test_from_geojson_path(fill_store, tmp_path, store):
        """Test loading from geojson with a file path."""
        _, store = fill_store(store, tmp_path / "polygon.db")
        store.to_geojson(tmp_path / "polygon.json")
        store2 = store.from_geojson(tmp_path / "polygon.json")
        assert len(store) == len(store2)

    @staticmethod
    def test_to_geojson_str(fill_store, tmp_path, store):
        """Test exporting to ldjson with a file path string."""
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
        """Test exporting to ldjson with a file handle."""
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
        """Test exporting to geojson with a file path."""
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
        """Test exporting to ldjson with a file path string."""
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
        """Test exporting to ldjson with a file handle."""
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
        """Test exporting to ldjson with a file path."""
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
        """Test dumping to a file path."""
        _, store = fill_store(store, ":memory:")
        store.dump(tmp_path / "dump_test.db")
        assert (tmp_path / "dump_test.db").stat().st_size > 0

    @staticmethod
    def test_dump_file_handle(fill_store, tmp_path, store):
        """Test dumping to a file handle."""
        _, store = fill_store(store, ":memory:")
        with open(tmp_path / "dump_test.db", "w") as fh:
            store.dump(fh)
        assert (tmp_path / "dump_test.db").stat().st_size > 0

    @staticmethod
    def test_dumps(fill_store, store):
        """Test dumping by returning a string."""
        _, store = fill_store(store, ":memory:")
        string = store.dumps()
        assert isinstance(string, str)

    @staticmethod
    def test_iquery_predicate_str(fill_store, store):
        """Test iquering with a predicate string."""
        keys, store = fill_store(store, ":memory:")
        store.patch(keys[0], properties={"class": 123})
        results = store.iquery((0, 0, 1024, 1024), where="props.get('class') == 123")
        assert len(results) == 1
        assert results[0] == keys[0]

    @staticmethod
    def test_iquery_predicate_callable(fill_store, store):
        """Test iquering with a predicate function."""
        keys, store = fill_store(store, ":memory:")
        store.patch(keys[0], properties={"class": 123})
        results = store.iquery(
            (0, 0, 1024, 1024),
            where=lambda props: props.get("class") == 123,
        )
        assert len(results) == 1
        assert results[0] == keys[0]

    @staticmethod
    def test_iquery_predicate_pickle(fill_store, store):
        """Test iquering with a pickled predicate function."""
        keys, store = fill_store(store, ":memory:")
        store.patch(keys[0], properties={"class": 123})

        results = store.query((0, 0, 1024, 1024), where=pickle.dumps(sample_predicate))
        assert len(results) == 1

    @staticmethod
    def test_query_predicate_str(fill_store, store):
        """Test quering with a predicate string."""
        keys, store = fill_store(store, ":memory:")
        store.patch(keys[0], properties={"class": 123})
        results = store.query((0, 0, 1024, 1024), where="props.get('class') == 123")
        assert len(results) == 1

    @staticmethod
    def test_query_predicate_callable(fill_store, store):
        """Test quering with a predicate function."""
        keys, store = fill_store(store, ":memory:")
        store.patch(keys[0], properties={"class": 123})
        results = store.iquery(
            (0, 0, 1024, 1024),
            where=lambda props: props.get("class") == 123,
        )
        assert len(results) == 1

    @staticmethod
    def test_query_predicate_pickle(fill_store, store):
        """Test quering with a pickled predicate function."""
        keys, store = fill_store(store, ":memory:")
        store.patch(keys[0], properties={"class": 123})

        results = store.query((0, 0, 1024, 1024), where=pickle.dumps(sample_predicate))
        assert len(results) == 1

    @staticmethod
    def test_append_invalid_geometry(fill_store, store):
        """Test that appending invalid geometry raises an exception."""
        store = store()
        with pytest.raises((TypeError, AttributeError)):
            store.append("point", {})

    @staticmethod
    def test_update_invalid_geometry(fill_store, store):
        """Test that updating  a new key and None geometry raises an exception."""
        store = store()
        key = "foo"
        with pytest.raises((TypeError, AttributeError)):
            store.patch(key, geometry=None, properties={"class": 123})

    @staticmethod
    def test_pop_key(fill_store, store):
        """Test popping an annotation by key."""
        keys, store = fill_store(store, ":memory:")
        assert keys[0] in store
        annotation = store.pop(keys[0])
        assert keys[0] not in store
        assert annotation not in store.values()

    @staticmethod
    def test_pop_key_error(fill_store, store):
        """Test that popping a key that is not in the store raises an exception."""
        store = store()
        with pytest.raises(KeyError):
            store.pop("foo")

    @staticmethod
    def test_popitem(fill_store, store):
        """Test popping a key and annotation by key."""
        keys, store = fill_store(store, ":memory:")
        key, annotation = store.popitem()
        assert key in keys
        assert key not in store
        assert annotation not in store.values()

    @staticmethod
    def test_setdefault(fill_store, store, sample_triangle):
        """Test setting a default value for a key."""
        store = store()
        default = Annotation(sample_triangle)
        assert "foo" not in store
        assert store.setdefault("foo", default) == default
        assert "foo" in store
        assert default in store.values()

    @staticmethod
    def test_setdefault_error(fill_store, store, sample_triangle):
        """Test setting a default value for a key with invalid type."""
        store = store()
        with pytest.raises(TypeError):
            store.setdefault("foo", {})

    @staticmethod
    def test_get(fill_store, store):
        """Test getting an annotation by key."""
        keys, store = fill_store(store, ":memory:")
        assert keys[0] in store
        assert store.get(keys[0]) == store[keys[0]]

    @staticmethod
    def test_get_default(fill_store, store):
        """Test getting a default value for a key."""
        store = store()
        assert "foo" not in store
        assert store.get("foo") is None

    @staticmethod
    def test_eq(fill_store, store):
        """Test comparing two stores for equality."""
        store1 = store()
        store2 = store()
        assert store1 == store2

        store1 = fill_store(store, ":memory:")[1]
        assert store1 != store2

        for key, value in store1.items():
            store2[key] = value
        assert store1 == store2

    @staticmethod
    def test_ne(fill_store, store):
        """Test comparing two stores for inequality."""
        store1 = fill_store(store, ":memory:")[1]
        store2 = fill_store(store, ":memory:")[1]
        assert store1 != store2

    @staticmethod
    def test_clear(fill_store, store):
        """Test clearing a store."""
        keys, store = fill_store(store, ":memory:")
        store.clear()
        assert keys[0] not in store
        assert len(store) == 0
        assert not store

    @staticmethod
    def test_update(fill_store, store, sample_triangle):
        """Test updating a store with a dictionary."""
        _, store = fill_store(store, ":memory:")
        annotations = {
            "foo": Annotation(sample_triangle),
        }
        assert "foo" not in store
        store.update(annotations)
        assert "foo" in store
        assert store["foo"] == annotations["foo"]
