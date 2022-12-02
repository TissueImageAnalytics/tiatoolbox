"""Tests for annotation store classes."""
import json
import pickle
import random
import sqlite3
import sys
from itertools import repeat
from numbers import Number
from pathlib import Path
from timeit import timeit
from typing import Any, Dict, Generator, List, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from shapely import affinity
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.point import Point

from tiatoolbox.annotation.storage import (
    Annotation,
    AnnotationStore,
    DictionaryStore,
    SQLiteMetadata,
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
    noise: Number = 0.01,
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
    ry = np.random.uniform(*eccentricity) * radius - 0.5 * rx
    x = rx * np.cos(alpha) + x + (np.random.rand(n_points) - 0.5) * noise
    y = ry * np.sin(alpha) + y + (np.random.rand(n_points) - 0.5) * noise
    boundary_coords = np.stack([x, y], axis=1).tolist()

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


def sample_where_1(props: Dict[str, Any]) -> bool:
    """Simple example predicate function for tests.

    Checks for a class = 1.

    """
    return props.get("class") == 1


def sample_where_123(props: Dict[str, Any]) -> bool:
    """Simple example predicate function for tests.

    Checks for a class = 123.

    """
    return props.get("class") == 123


def sample_select(props: Dict[str, Any]) -> Tuple[Any]:
    """Simple example select expression for tests.

    Gets the class value.

    """
    return props.get("class")


def sample_multi_select(props: Dict[str, Any]) -> Tuple[Any]:
    """Simple example select expression for tests.

    Gets the class value and the class mod 2.

    """
    return (props.get("class"), props.get("class") % 2)


def annotations_center_of_mass(annotations):
    """Compute the mean of the annotation centroids."""
    centroids = [annotation.geometry.centroid for annotation in annotations]
    return MultiPoint(centroids).centroid


# Fixtures


@pytest.fixture(scope="session")
def cell_grid() -> List[Polygon]:
    """Generate a grid of fake cell boundary polygon annotations."""
    np.random.seed(0)
    return [cell_polygon((i * 25, j * 25)) for i, j in np.ndindex(*GRID_SIZE)]


@pytest.fixture(scope="session")
def points_grid() -> List[Polygon]:
    """Generate a grid of fake point annotations."""
    np.random.seed(0)
    return [Point((i * 25, j * 25)) for i, j in np.ndindex(*GRID_SIZE)]


@pytest.fixture(scope="session")
def sample_triangle() -> Polygon:
    """Simple triangle polygon used for testing."""
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
            Annotation(point, properties={"class": random.randint(0, 4)})
            for point in points_grid
        ]
        keys = store.append_many(annotations)
        return keys, store

    return _fill_store


# Class Specific Tests


def test_sqlite_store_compile_options():
    options = SQLiteStore.compile_options()
    assert all(isinstance(x, str) for x in options)


def test_sqlite_store_compile_options_exception(monkeypatch):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 37, 0))
    monkeypatch.setattr(
        SQLiteStore,
        "compile_options",
        lambda x: ["ENABLE_RTREE", "ENABLE_JSON1"],
        raising=True,
    )
    SQLiteStore()
    monkeypatch.setattr(SQLiteStore, "compile_options", lambda x: [], raising=True)
    with pytest.raises(Exception, match="RTREE and JSON1"):
        SQLiteStore()


def test_sqlite_store_compile_options_exception_v3_38(monkeypatch):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 38, 0))
    monkeypatch.setattr(
        SQLiteStore, "compile_options", lambda x: ["OMIT_JSON"], raising=True
    )
    with pytest.raises(Exception, match="JSON must not"):
        SQLiteStore()


def test_sqlite_store_compile_options_missing_math(monkeypatch, caplog):
    """Test that a warning is shown if the sqlite math module is missing."""
    monkeypatch.setattr(
        SQLiteStore,
        "compile_options",
        lambda x: ["ENABLE_JSON1", "ENABLE_RTREE"],
        raising=True,
    )
    SQLiteStore()
    assert "SQLite math functions are not enabled" in caplog.text


def test_sqlite_store_multiple_connection(tmp_path):
    store = SQLiteStore(tmp_path / "annotations.db")
    store2 = SQLiteStore(tmp_path / "annotations.db")
    assert len(store) == len(store2)


def test_sqlite_store_index_type_error():
    """Test adding an index of invalid type."""
    store = SQLiteStore()
    with pytest.raises(TypeError, match="where"):
        store.create_index("foo", lambda g, p: "foo" in p)


def test_sqlite_store_index_version_error(monkeypatch):
    """Test adding an index with SQlite <3.9."""
    store = SQLiteStore()
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 8, 0))
    with pytest.raises(Exception, match="Requires sqlite version 3.9.0"):
        store.create_index("foo", lambda _, p: "foo" in p)


def test_sqlite_store_index_str(fill_store, tmp_path):
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


def test_sqlite_create_index_no_analyze(fill_store, tmp_path):
    """Test that creating an index without ANALYZE."""
    _, store = fill_store(SQLiteStore, tmp_path / "polygon.db")
    properties_predicate = "props['class']"
    store.create_index("test_index", properties_predicate, analyze=False)
    assert "test_index" in store.indexes()


def test_sqlite_pquery_warn_no_index(fill_store):
    """Test that querying without an index warns."""
    _, store = fill_store(SQLiteStore, ":memory:")
    with pytest.warns(UserWarning, match="index"):
        store.pquery("*", unique=False)
    # Check that there is no warning after creating the index
    store.create_index("test_index", "props['class']")
    with pytest.warns(None) as record:
        store.pquery("props['class']")
        assert len(record) == 0


def test_sqlite_store_indexes(fill_store, tmp_path):
    """Test getting a list of index names."""
    _, store = fill_store(SQLiteStore, tmp_path / "polygon.db")
    store.create_index("test_index", "props['class']")
    assert "test_index" in store.indexes()


def test_sqlite_drop_index_error(fill_store, tmp_path):
    """Test dropping an index that does not exist."""
    _, store = fill_store(SQLiteStore, tmp_path / "polygon.db")
    with pytest.raises(sqlite3.OperationalError, match="no such index"):
        store.drop_index("test_index")


def test_sqlite_store_unsupported_compression(sample_triangle):
    """Test that using an unsupported compression str raises error."""
    store = SQLiteStore(compression="foo")
    with pytest.raises(Exception, match="Unsupported"):
        _ = store.serialise_geometry(sample_triangle)


def test_sqlite_optimize(fill_store, tmp_path):
    """Test optimizing the database."""
    _, store = fill_store(SQLiteStore, tmp_path / "polygon.db")
    store.optimize()


def test_sqlite_store_no_compression(sample_triangle):
    """Test that using no compression raises no error."""
    store = SQLiteStore(compression=None)
    serialised = store.serialise_geometry(sample_triangle)
    deserialised = store.deserialize_geometry(serialised)
    assert deserialised.wkb == sample_triangle.wkb


def test_sqlite_store_unsupported_decompression():
    """Test that using an unsupported decompression str raises error."""
    store = SQLiteStore(compression="foo")
    with pytest.raises(Exception, match="Unsupported"):
        _ = store.deserialize_geometry(bytes())


def test_sqlite_store_wkt_deserialisation(sample_triangle):
    """Test WKT deserialisation."""
    store = SQLiteStore(compression=None)
    wkt = sample_triangle.wkt
    geom = store.deserialize_geometry(wkt)
    assert geom == sample_triangle


def test_sqlite_store_wkb_deserialisation(sample_triangle):
    """Test WKB deserialisation.

    Test the default stattic method in the ABC.

    """
    wkb = sample_triangle.wkb
    geom = AnnotationStore.deserialize_geometry(wkb)
    assert geom == sample_triangle


def test_sqlite_store_metadata_get_key():
    """Test getting a metadata entry."""
    store = SQLiteStore()
    assert store.metadata["compression"] == "zlib"


def test_sqlite_store_metadata_get_keyerror():
    """Test getting a metadata entry that does not exists."""
    store = SQLiteStore()
    with pytest.raises(KeyError):
        store.metadata["foo"]


def test_sqlite_store_metadata_delete_keyerror():
    """Test deleting a metadata entry that does not exists."""
    store = SQLiteStore(compression=None)
    with pytest.raises(KeyError):
        del store.metadata["foo"]


def test_sqlite_store_metadata_delete():
    """Test adding and deleting a metadata entry."""
    store = SQLiteStore(compression=None)
    store.metadata["foo"] = 1
    assert "foo" in store.metadata
    del store.metadata["foo"]
    assert "foo" not in store.metadata


def test_sqlite_store_metadata_iter():
    """Test iterating over metadata entries."""
    conn = sqlite3.Connection(":memory:")
    metadata = SQLiteMetadata(conn)
    metadata["foo"] = 1
    metadata["bar"] = 2
    assert set(metadata.keys()) == {"foo", "bar"}


def test_sqlite_store_metadata_len():
    """Test len of metadata entries."""
    conn = sqlite3.Connection(":memory:")
    metadata = SQLiteMetadata(conn)
    metadata["foo"] = 1
    metadata["bar"] = 2
    assert len(metadata) == 2


def test_sqlite_drop_index():
    """Test creating and dropping an index."""
    store = SQLiteStore()
    store.create_index("foo", "props['class']")
    assert "foo" in store.indexes()
    store.drop_index("foo")
    assert "foo" not in store.indexes()


def test_sqlite_drop_index_fail():
    """Test dropping an index that does not exist."""
    store = SQLiteStore()
    with pytest.raises(sqlite3.OperationalError):
        store.drop_index("foo")


def test_sqlite_optimize_no_vacuum(fill_store, tmp_path):
    """Test running the optimize function on an SQLiteStore without VACUUM."""
    _, store = fill_store(SQLiteStore, tmp_path / "polygon.db")
    store.optimize(limit=0, vacuum=False)


def test_annotation_to_geojson():
    """Test converting an annotation to geojson."""
    annotation = Annotation(
        geometry=Point(0, 0),
        properties={"foo": "bar", "baz": "qux"},
    )
    geojson = json.loads(annotation.to_geojson())
    assert geojson["type"] == "Feature"
    assert geojson["geometry"]["type"] == "Point"
    assert geojson["properties"] == {"foo": "bar", "baz": "qux"}


def test_remove_area_column(fill_store):
    """Test removing an area column."""
    _, store = fill_store(SQLiteStore, ":memory:")
    store.remove_area_column()
    assert "area" not in store._get_table_columns()
    result = store.query((0, 0, 1000, 1000))
    assert len(result) == 200

    store.add_area_column()
    assert "area" in store.indexes()
    store.remove_area_column()
    # Check that the index is removed if its there
    assert "area" not in store.indexes()


def test_remove_area_column_indexed(fill_store):
    """Test removing an area column if theres an index on it."""
    _, store = fill_store(SQLiteStore, ":memory:")
    store.create_index("area", '"area"')
    store.remove_area_column()
    assert "area" not in store._get_table_columns()
    result = store.query((0, 0, 1000, 1000))
    assert len(result) == 200


def test_add_area_column(fill_store):
    """Test adding an area column."""
    _, store = fill_store(SQLiteStore, ":memory:")
    store.remove_area_column()
    store.add_area_column()
    assert "area" in store.indexes()
    assert "area" in store._get_table_columns()

    # check the results are properly sorted by area
    result = store.query((0, 0, 100, 100))
    areas = [ann.geometry.area for ann in result.values()]
    assert areas == sorted(areas, reverse=True)

    # check add index option of add_area_column
    _, store = fill_store(SQLiteStore, ":memory:")
    store.remove_area_column()
    assert "area" not in store.indexes()
    store.add_area_column(False)
    assert "area" not in store.indexes()


def test_query_min_area(fill_store):
    """Test querying with a minimum area."""
    _, store = fill_store(SQLiteStore, ":memory:")
    result = store.query((0, 0, 1000, 1000), min_area=1)
    assert len(result) == 100  # should only get cells, pts are too small


def test_query_min_area_no_area_column(fill_store):
    """Test querying with a minimum area when there is no area column."""
    _, store = fill_store(SQLiteStore, ":memory:")
    store.remove_area_column()
    with pytest.raises(ValueError, match="without an area column"):
        store.query((0, 0, 1000, 1000), min_area=1)


def test_auto_commit(fill_store, tmp_path):
    """Test auto commit.

    Check that if auto-commit is False, the changes are not committed until
    commit() is called.

    """
    _, store = fill_store(SQLiteStore, tmp_path / "polygon.db")
    store.close()
    store = SQLiteStore(tmp_path / "polygon.db", auto_commit=False)
    keys = list(store.keys())
    store.patch(keys[0], Point(-500, -500))
    store.append_many([Annotation(Point(10, 10), {}), Annotation(Point(20, 20), {})])
    store.remove_many(keys[5:10])
    store.clear()
    store.close()
    store = SQLiteStore(tmp_path / "polygon.db")
    result = store.query((0, 0, 1000, 1000))
    assert len(result) == 200  # check none of the changes were committed
    store = SQLiteStore(tmp_path / "polygon2.db", auto_commit=False)
    store.append_many([Annotation(Point(10, 10), {}), Annotation(Point(20, 20), {})])
    store.commit()
    store.close()
    store = SQLiteStore(tmp_path / "polygon2.db")
    assert len(store) == 2  # check explicitly committing works


# Annotation Store Interface Tests (AnnotationStoreABC)


class TestStore:
    scenarios = [
        ("Dictionary", {"store_cls": DictionaryStore}),
        ("SQLite", {"store_cls": SQLiteStore}),
    ]

    @staticmethod
    def test_open_close(fill_store, tmp_path, store_cls):
        """Test opening and closing a store."""
        path = tmp_path / "polygons"
        keys, store = fill_store(store_cls, path)
        store.close()
        store2 = store.open(path)
        assert len(store2) == len(keys)

    @staticmethod
    def test_append_many(cell_grid, tmp_path, store_cls):
        """Test bulk append of annotations."""
        store = store_cls(tmp_path / "polygons")
        annotations = [
            Annotation(cell, {"class": random.randint(0, 6)}) for cell in cell_grid
        ]
        keys = store.append_many(annotations)
        assert len(keys) == len(cell_grid)

    @staticmethod
    def test_append_many_with_keys(cell_grid, tmp_path, store_cls):
        """Test bulk append of annotations with keys."""
        store = store_cls(tmp_path / "polygons")
        annotations = [
            Annotation(cell, {"class": random.randint(0, 6)}) for cell in cell_grid
        ]
        keys = [chr(n) for n, _ in enumerate(annotations)]
        returned_keys = store.append_many(annotations, keys=keys)
        assert len(returned_keys) == len(cell_grid)
        assert keys == returned_keys

    @staticmethod
    def test_append_many_with_keys_len_mismatch(cell_grid, tmp_path, store_cls):
        """Test bulk append of annotations with keys of wrong length."""
        store = store_cls(tmp_path / "polygons")
        annotations = [
            Annotation(cell, {"class": random.randint(0, 6)}) for cell in cell_grid
        ]
        keys = ["foo"]
        with pytest.raises(ValueError, match="equal"):
            store.append_many(annotations, keys=keys)

    @staticmethod
    def test_query_bbox(fill_store, tmp_path, store_cls):
        """Test query with a bounding box."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        results = store.query((0, 0, 25, 25))
        assert len(results) == 8

    @staticmethod
    def test_iquery_bbox(fill_store, tmp_path, store_cls):
        """Test iquery with a bounding box."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        results = store.iquery((0, 0, 25, 25))
        assert len(results) == 8
        assert all(isinstance(key, str) for key in results)

    @staticmethod
    def test_query_polygon(fill_store, tmp_path, store_cls):
        """Test query with a non-rectangular geometry."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        results = store.query(Polygon([(0, 0), (0, 25), (1, 1), (25, 0)]))
        assert len(results) == 6
        assert all(isinstance(ann, Annotation) for ann in results.values())

    @staticmethod
    def test_iquery_polygon(fill_store, tmp_path, store_cls):
        """Test iquery with a non-rectangular geometry."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        results = store.iquery(Polygon([(0, 0), (0, 25), (1, 1), (25, 0)]))
        assert len(results) == 6
        assert all(isinstance(key, str) for key in results)

    @staticmethod
    def test_patch(fill_store, tmp_path, store_cls):
        """Test patching an annotation."""
        keys, store = fill_store(store_cls, tmp_path / "polygon.db")
        key = keys[0]
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        # Geometry update
        store.patch(key, new_geometry)
        assert store[key].geometry == new_geometry
        # Properties update
        store.patch(key, properties={"abc": 123})
        assert store[key].properties["abc"] == 123

    @staticmethod
    def test_patch_append(fill_store, tmp_path, store_cls):
        """Test patching an annotation that does not exist."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        store.patch("foo", new_geometry)
        assert store["foo"].geometry == new_geometry

    @staticmethod
    def test_patch_many(fill_store, tmp_path, store_cls):
        """Test bulk patch."""
        keys, store = fill_store(store_cls, tmp_path / "polygon.db")
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        store.patch_many(
            keys, repeat(new_geometry, len(keys)), repeat({"abc": 123}, len(keys))
        )

        for _, key in enumerate(keys[:10]):
            assert store[key].geometry == new_geometry
            assert store[key].properties["abc"] == 123

    @staticmethod
    def test_patch_many_no_properies(fill_store, tmp_path, store_cls):
        """Test bulk patch with no properties."""
        keys, store = fill_store(store_cls, tmp_path / "polygon.db")
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        store.patch_many(keys, repeat(new_geometry, len(keys)))

        for _, key in enumerate(keys[:10]):
            assert store[key].geometry == new_geometry
            assert store[key].properties == {}

    @staticmethod
    def test_patch_many_no_geometry(fill_store, tmp_path, store_cls):
        """Test bulk patch with no geometry."""
        keys, store = fill_store(store_cls, tmp_path / "polygon.db")
        store.patch_many(keys, properties_iter=repeat({"abc": 123}, len(keys)))

        for _, key in enumerate(keys[:10]):
            assert store[key].geometry is not None
            assert store[key].properties["abc"] == 123

    @staticmethod
    def test_patch_many_no_geometry_no_properties(fill_store, tmp_path, store_cls):
        """Test bulk patch with no geometry and no properties."""
        keys, store = fill_store(store_cls, tmp_path / "polygon.db")
        with pytest.raises(ValueError, match="At least one"):
            store.patch_many(keys)

    @staticmethod
    def test_patch_many_append(fill_store, tmp_path, store_cls):
        """Test bulk patching annotations that do not exist."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        store.patch_many(["foo", "bar"], repeat(new_geometry, 2))
        assert store["foo"].geometry == new_geometry
        assert store["bar"].geometry == new_geometry

    @staticmethod
    def test_patch_many_len_mismatch(fill_store, tmp_path, store_cls):
        """Test bulk patch with wrong number of keys."""
        keys, store = fill_store(store_cls, tmp_path / "polygon.db")
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        with pytest.raises(ValueError, match="equal"):
            store.patch_many(keys[1:], repeat(new_geometry, 10))

    @staticmethod
    def test_keys(fill_store, tmp_path, store_cls):
        """Test getting an keys iterator."""
        keys, store = fill_store(store_cls, tmp_path / "polygon.db")
        keys = list(keys)
        assert len(list(store.keys())) == len(keys)
        assert isinstance(list(store.keys())[0], type(keys[0]))

    @staticmethod
    def test_remove(fill_store, tmp_path, store_cls):
        """Test removing an annotation."""
        keys, store = fill_store(store_cls, tmp_path / "polygon.db")
        key = keys[0]
        store.remove(key)
        assert len(store) == FILLED_LEN - 1

    @staticmethod
    def test_delitem(fill_store, tmp_path, store_cls):
        """Test using the delitem syntax."""
        keys, store = fill_store(store_cls, tmp_path / "polygon.db")
        key = keys[0]
        del store[key]
        assert len(store) == FILLED_LEN - 1

    @staticmethod
    def test_remove_many(fill_store, tmp_path, store_cls):
        """Test bulk deletion."""
        keys, store = fill_store(store_cls, tmp_path / "polygon.db")
        store.remove_many(keys)
        assert len(store) == 0

    @staticmethod
    def test_len(fill_store, tmp_path, store_cls):
        """Test finding the number of annotation via the len operator."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        assert len(store) == FILLED_LEN

    @staticmethod
    def test_contains(fill_store, tmp_path, store_cls):
        """Test using the contains (in) operator."""
        keys, store = fill_store(store_cls, tmp_path / "polygon.db")
        for key in keys:
            assert key in store

    @staticmethod
    def test_iter(fill_store, tmp_path, store_cls):
        """Test iterating over the store."""
        keys, store = fill_store(store_cls, tmp_path / "polygon.db")
        for key in store:
            assert key in keys

    @staticmethod
    def test_getitem(fill_store, tmp_path, sample_triangle, store_cls):
        """Test the getitem syntax."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        key = store.append(Annotation(sample_triangle))
        annotation = store[key]
        assert annotation.geometry == sample_triangle
        assert annotation.properties == {}

    @staticmethod
    def test_setitem(fill_store, tmp_path, sample_triangle, store_cls):
        """Test the setitem syntax."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        key = store.append(Annotation(sample_triangle))
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        new_properties = {"abc": 123}
        store[key] = Annotation(new_geometry, new_properties)
        assert store[key] == Annotation(new_geometry, new_properties)

    @staticmethod
    def test_getitem_setitem_cycle(fill_store, tmp_path, sample_triangle, store_cls):
        """Test getting an setting an annotation."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        key = store.append(Annotation(sample_triangle, {"class": 0}))
        annotation = store[key]
        store[key] = annotation
        assert store[key] == annotation

    @staticmethod
    def test_from_dataframe(cell_grid, store_cls):
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
        store = store_cls.from_dataframe(df)
        keys = list(store.keys())
        annotation = store[keys[0]]
        assert "row_id" in annotation.properties

    @staticmethod
    def test_to_dataframe(fill_store, tmp_path, store_cls):
        """Test converting to a pandas dataframe."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        df = store.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == FILLED_LEN
        assert "geometry" in df.columns
        assert df.index.name == "key"
        assert isinstance(df.geometry.iloc[0], Polygon)

    @staticmethod
    def test_features(fill_store, tmp_path, store_cls):
        """Test converting to a features dictionaries."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        features = store.features()
        assert isinstance(features, Generator)
        features = list(features)
        assert len(features) == FILLED_LEN
        assert isinstance(features[0], dict)
        assert all(
            {"type", "geometry", "properties"} == set(f.keys()) for f in features
        )

    @staticmethod
    def test_to_geodict(fill_store, tmp_path, store_cls):
        """Test converting to a geodict."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        geodict = store.to_geodict()
        assert isinstance(geodict, dict)
        assert "features" in geodict
        assert "type" in geodict
        assert geodict["type"] == "FeatureCollection"
        assert geodict["features"] == list(store.features())

    @staticmethod
    def test_from_geojson_str(fill_store, tmp_path, store_cls):
        """Test loading from geojson with a file path string."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        geojson = store.to_geojson()
        store2 = store_cls.from_geojson(geojson)
        assert len(store) == len(store2)

    @staticmethod
    def test_from_geojson_file(fill_store, tmp_path, store_cls):
        """Test loading from geojson with a file handle."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        store.to_geojson(tmp_path / "polygon.json")
        with open(tmp_path / "polygon.json", "r") as file_handle:
            store2 = store_cls.from_geojson(file_handle)
        assert len(store) == len(store2)

    @staticmethod
    def test_from_geojson_path(fill_store, tmp_path, store_cls):
        """Test loading from geojson with a file path."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        store.to_geojson(tmp_path / "polygon.json")
        store2 = store_cls.from_geojson(tmp_path / "polygon.json")
        assert len(store) == len(store2)

    @staticmethod
    def test_from_geojson_path_transform(fill_store, tmp_path, store_cls):
        """Test loading from geojson with a transform."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        com = annotations_center_of_mass(list(store.values()))
        store.to_geojson(tmp_path / "polygon.json")
        # load the store translated so that origin is (100,100) and scaled by 2
        store2 = store_cls.from_geojson(
            tmp_path / "polygon.json", scale_factor=(2, 2), origin=(100, 100)
        )
        assert len(store) == len(store2)
        com2 = annotations_center_of_mass(list(store2.values()))
        assert com2.x == pytest.approx((com.x - 100) * 2)
        assert com2.y == pytest.approx((com.y - 100) * 2)

    @staticmethod
    def test_transform(fill_store, tmp_path, store_cls):
        """Test translating a store."""

        def test_translation(geom):
            """Performs a translation of input geometry."""
            return affinity.translate(geom, 100, 100)

        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        com = annotations_center_of_mass(list(store.values()))
        store.transform(test_translation)
        com2 = annotations_center_of_mass(list(store.values()))
        assert com2.x - com.x == pytest.approx(100)
        assert com2.y - com.y == pytest.approx(100)

    @staticmethod
    def test_to_geojson_str(fill_store, tmp_path, store_cls):
        """Test exporting to ndjson with a file path string."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        geojson = store.to_geojson()
        assert isinstance(geojson, str)
        geodict = json.loads(geojson)
        assert "features" in geodict
        assert "type" in geodict
        assert geodict["type"] == "FeatureCollection"
        assert len(geodict["features"]) == len(list(store.features()))

    @staticmethod
    def test_to_geojson_file(fill_store, tmp_path, store_cls):
        """Test exporting to ndjson with a file handle."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
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
    def test_to_geojson_path(fill_store, tmp_path, store_cls):
        """Test exporting to geojson with a file path."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        geojson = store.to_geojson(fp=tmp_path / "polygon.json")
        assert geojson is None
        with open(tmp_path / "polygon.json", "r") as fh:
            geodict = json.load(fh)
        assert "features" in geodict
        assert "type" in geodict
        assert geodict["type"] == "FeatureCollection"
        assert len(geodict["features"]) == len(list(store.features()))

    @staticmethod
    def test_to_ndjson_str(fill_store, tmp_path, store_cls):
        """Test exporting to ndjson with a file path string."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        ndjson = store.to_ndjson()
        for line in ndjson.split():
            assert isinstance(line, str)
            feature = json.loads(line)
            assert "type" in feature
            assert "geometry" in feature
            assert "properties" in feature

    @staticmethod
    def test_to_ndjson_file(fill_store, tmp_path, store_cls):
        """Test exporting to ndjson with a file handle."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        with open(tmp_path / "polygon.ndjson", "w") as fh:
            ndjson = store.to_ndjson(fp=fh)
        assert ndjson is None
        with open(tmp_path / "polygon.ndjson", "r") as fh:
            for line in fh.readlines():
                assert isinstance(line, str)
                feature = json.loads(line)
                assert "type" in feature
                assert "geometry" in feature
                assert "properties" in feature

    @staticmethod
    def test_to_ndjson_path(fill_store, tmp_path, store_cls):
        """Test exporting to ndjson with a file path."""
        _, store = fill_store(store_cls, tmp_path / "polygon.db")
        ndjson = store.to_ndjson(fp=tmp_path / "polygon.ndjson")
        assert ndjson is None
        with open(tmp_path / "polygon.ndjson", "r") as fh:
            for line in fh.readlines():
                assert isinstance(line, str)
                feature = json.loads(line)
                assert "type" in feature
                assert "geometry" in feature
                assert "properties" in feature

    @staticmethod
    def test_dump(fill_store, tmp_path, store_cls):
        """Test dumping to a file path."""
        _, store = fill_store(store_cls, ":memory:")
        store.dump(tmp_path / "dump_test.db")
        assert (tmp_path / "dump_test.db").stat().st_size > 0

    @staticmethod
    def test_dump_file_handle(fill_store, tmp_path, store_cls):
        """Test dumping to a file handle."""
        _, store = fill_store(store_cls, ":memory:")
        with open(tmp_path / "dump_test.db", "w") as fh:
            store.dump(fh)
        assert (tmp_path / "dump_test.db").stat().st_size > 0

    @staticmethod
    def test_dumps(fill_store, store_cls):
        """Test dumping by returning a string."""
        _, store = fill_store(store_cls, ":memory:")
        string = store.dumps()
        assert isinstance(string, str)

    @staticmethod
    def test_iquery_predicate_str(fill_store, store_cls):
        """Test iquering with a predicate string."""
        keys, store = fill_store(store_cls, ":memory:")
        store.patch(keys[0], properties={"class": 123})
        results = store.iquery((0, 0, 1024, 1024), where="props.get('class') == 123")
        assert len(results) == 1
        assert results[0] == keys[0]

    @staticmethod
    def test_iquery_predicate_callable(fill_store, store_cls):
        """Test iquering with a predicate function."""
        keys, store = fill_store(store_cls, ":memory:")
        store.patch(keys[0], properties={"class": 123})
        results = store.iquery(
            (0, 0, 1024, 1024),
            where=lambda props: props.get("class") == 123,
        )
        assert len(results) == 1
        assert results[0] == keys[0]

    @staticmethod
    def test_iquery_predicate_pickle(fill_store, store_cls):
        """Test iquering with a pickled predicate function."""
        keys, store = fill_store(store_cls, ":memory:")
        store.patch(keys[0], properties={"class": 123})

        results = store.query((0, 0, 1024, 1024), where=pickle.dumps(sample_where_123))
        assert len(results) == 1

    @staticmethod
    def test_query_predicate_str(fill_store, store_cls):
        """Test quering with a predicate string."""
        keys, store = fill_store(store_cls, ":memory:")
        store.patch(keys[0], properties={"class": 123})
        results = store.query((0, 0, 1024, 1024), where="props.get('class') == 123")
        assert len(results) == 1

    @staticmethod
    def test_query_predicate_callable(fill_store, store_cls):
        """Test quering with a predicate function."""
        keys, store = fill_store(store_cls, ":memory:")
        store.patch(keys[0], properties={"class": 123})
        results = store.query(
            # (0, 0, 1024, 1024),  # noqa: E800
            where=lambda props: props.get("class")
            == 123,
        )
        assert len(results) == 1

    @staticmethod
    def test_query_predicate_pickle(fill_store, store_cls):
        """Test quering with a pickled predicate function."""
        keys, store = fill_store(store_cls, ":memory:")
        store.patch(keys[0], properties={"class": 123})

        results = store.query((0, 0, 1024, 1024), where=pickle.dumps(sample_where_123))
        assert len(results) == 1

    @staticmethod
    def test_append_invalid_geometry(fill_store, store_cls):
        """Test that appending invalid geometry raises an exception."""
        store = store_cls()
        with pytest.raises((TypeError, AttributeError)):
            store.append("point", {})

    @staticmethod
    def test_update_invalid_geometry(fill_store, store_cls):
        """Test that updating  a new key and None geometry raises an exception."""
        store = store_cls()
        key = "foo"
        with pytest.raises((TypeError, AttributeError)):
            store.patch(key, geometry=None, properties={"class": 123})

    @staticmethod
    def test_pop_key(fill_store, store_cls):
        """Test popping an annotation by key."""
        keys, store = fill_store(store_cls, ":memory:")
        assert keys[0] in store
        annotation = store.pop(keys[0])
        assert keys[0] not in store
        assert annotation not in store.values()

    @staticmethod
    def test_pop_key_error(fill_store, store_cls):
        """Test that popping a key that is not in the store raises an exception."""
        store = store_cls()
        with pytest.raises(KeyError):
            store.pop("foo")

    @staticmethod
    def test_popitem(fill_store, store_cls):
        """Test popping a key and annotation by key."""
        keys, store = fill_store(store_cls, ":memory:")
        key, annotation = store.popitem()
        assert key in keys
        assert key not in store
        assert annotation not in store.values()

    @staticmethod
    def test_popitem_empty_error(fill_store, store_cls):
        """Test that popping an empty store raises an exception."""
        store = store_cls()
        with pytest.raises(KeyError):
            store.popitem()

    @staticmethod
    def test_setdefault(fill_store, store_cls, sample_triangle):
        """Test setting a default value for a key."""
        store = store_cls()
        default = Annotation(sample_triangle)
        assert "foo" not in store
        assert store.setdefault("foo", default) == default
        assert "foo" in store
        assert default in store.values()

    @staticmethod
    def test_setdefault_error(fill_store, store_cls, sample_triangle):
        """Test setting a default value for a key with invalid type."""
        store = store_cls()
        with pytest.raises(TypeError):
            store.setdefault("foo", {})

    @staticmethod
    def test_get(fill_store, store_cls):
        """Test getting an annotation by key."""
        keys, store = fill_store(store_cls, ":memory:")
        assert keys[0] in store
        assert store.get(keys[0]) == store[keys[0]]

    @staticmethod
    def test_get_default(fill_store, store_cls):
        """Test getting a default value for a key."""
        store = store_cls()
        assert "foo" not in store
        assert store.get("foo") is None

    @staticmethod
    def test_eq(fill_store, store_cls):
        """Test comparing two stores for equality."""
        store1 = store_cls()
        store2 = store_cls()
        assert store1 == store2

        store1 = fill_store(store_cls, ":memory:")[1]
        assert store1 != store2

        for key, value in store1.items():
            store2[key] = value
        assert store1 == store2

    @staticmethod
    def test_ne(fill_store, store_cls):
        """Test comparing two stores for inequality."""
        store1 = fill_store(store_cls, ":memory:")[1]
        store2 = fill_store(store_cls, ":memory:")[1]
        assert store1 != store2

    @staticmethod
    def test_clear(fill_store, store_cls):
        """Test clearing a store."""
        keys, store = fill_store(store_cls, ":memory:")
        store.clear()
        assert keys[0] not in store
        assert len(store) == 0
        assert not store

    @staticmethod
    def test_update(fill_store, store_cls, sample_triangle):
        """Test updating a store with a dictionary."""
        _, store = fill_store(store_cls, ":memory:")
        annotations = {
            "foo": Annotation(sample_triangle),
        }
        assert "foo" not in store
        store.update(annotations)
        assert "foo" in store
        assert store["foo"] == annotations["foo"]

    @staticmethod
    def test_cast_dict(fill_store, store_cls):
        """Test casting a store to a dictionary."""
        keys, store = fill_store(store_cls, ":memory:")
        store_dict = dict(store)
        assert keys[0] in store_dict
        assert store[keys[0]] in store_dict.values()

    @staticmethod
    def test_query_invalid_geometry_predicate(fill_store, store_cls):
        """Test that invalid geometry predicate raises an exception."""
        store = store_cls()
        with pytest.raises(ValueError, match="Invalid geometry predicate"):
            store.query((0, 0, 1024, 1024), geometry_predicate="foo")

    @staticmethod
    def test_query_no_geometry_or_where(fill_store, store_cls):
        """Test that query raises exception when no geometry or predicate given."""
        store = store_cls()
        with pytest.raises(ValueError, match="At least one of"):
            store.query()

    @staticmethod
    def test_iquery_invalid_geometry_predicate(fill_store, store_cls):
        """Test that invalid geometry predicate raises an exception."""
        store = store_cls()
        with pytest.raises(ValueError, match="Invalid geometry predicate"):
            store.iquery((0, 0, 1024, 1024), geometry_predicate="foo")

    @staticmethod
    def test_serialise_deseialise_geometry(fill_store, store_cls):
        """Test that geometry can be serialised and deserialised."""
        _, store = fill_store(store_cls, ":memory:")
        for _, annotation in store.items():
            geometry = annotation.geometry
            serialised = store.serialise_geometry(geometry)
            deserialised = store.deserialize_geometry(serialised)
            if isinstance(serialised, str):
                assert geometry.wkt == deserialised.wkt
            else:
                assert geometry.wkb == deserialised.wkb

    @staticmethod
    def test_commit(fill_store, store_cls, tmp_path):
        """Test committing a store."""
        store_path = tmp_path / "test_store"
        test_store = store_cls(store_path)
        test_store["foo"] = Annotation(Point(0, 0))
        test_store.commit()
        assert store_path.exists()
        test_store.close()
        del test_store  # skipcq: PTC-W0043
        test_store = store_cls(store_path)
        assert "foo" in test_store

    @staticmethod
    def test_load_cases_error(fill_store, store_cls):
        """Test that loading a store with an invalid file handle raises an exception."""
        store = store_cls()
        with pytest.raises(IOError, match="Invalid file handle or path"):
            store._load_cases(["foo"], lambda: None, lambda: None)

    @staticmethod
    def test_py37_init(fill_store, store_cls, monkeypatch):
        """Test that __init__ is compatible with Python 3.7."""
        py37_version = (3, 7, 0)

        class Connection(sqlite3.Connection):
            """Mock SQLite connection."""

            def create_function(self, name: str, num_params: int, func: Any) -> None:
                """Mock create_function without `deterministic` kwarg."""
                return self.create_function(self, name, num_params)

        monkeypatch.setattr(sys, "version_info", py37_version)
        monkeypatch.setattr(sqlite3, "Connection", Connection)
        _ = store_cls()

    @staticmethod
    def test_bquery(fill_store, store_cls):
        """Test querying a store with a bounding box."""
        _, store = fill_store(store_cls, ":memory:")
        dictionary = store.bquery((0, 0, 1e10, 1e10))
        assert isinstance(dictionary, dict)
        assert len(dictionary) == len(store)

    @staticmethod
    def test_bquery_polygon(fill_store, store_cls):
        """Test querying a store with a polygon."""
        _, store = fill_store(store_cls, ":memory:")
        dictionary = store.bquery(Polygon.from_bounds(0, 0, 1e10, 1e10))
        assert isinstance(dictionary, dict)
        assert len(dictionary) == len(store)

    @staticmethod
    def test_bquery_callable(fill_store, store_cls):
        """Test querying a store with a bounding box and a callable where."""
        keys, store = fill_store(store_cls, ":memory:")
        store.patch(keys[0], properties={"class": 123})
        dictionary = store.bquery(
            (0, 0, 1e10, 1e10), where=lambda props: props.get("class") == 123
        )
        assert isinstance(dictionary, dict)
        assert len(dictionary) == 1

    @staticmethod
    def test_pquery_all(fill_store, store_cls):
        """Test querying for all properties."""
        keys, store = fill_store(store_cls, ":memory:")
        dictionary = store.pquery("*", unique=False)
        assert isinstance(dictionary, dict)
        assert len(dictionary) == len(store)
        assert isinstance(dictionary[keys[0]], dict)

    @staticmethod
    def test_pquery_all_unique_exception(fill_store, store_cls):
        """Test querying for all properties."""
        _, store = fill_store(store_cls, ":memory:")
        with pytest.raises(ValueError, match="unique"):
            _ = store.pquery("*", unique=True)

    @staticmethod
    def test_pquery_unique(fill_store, store_cls):
        """Test querying for properties."""
        _, store = fill_store(store_cls, ":memory:")
        result_set = store.pquery(select="props.get('class')")
        assert isinstance(result_set, set)
        assert result_set == {0, 1, 2, 3, 4, None}

    @staticmethod
    def test_pquery_unique_with_geometry(fill_store, store_cls):
        """Test querying for properties with a geometry intersection."""
        _, store = fill_store(store_cls, ":memory:")
        result_set = store.pquery(
            select="props.get('class')",
            geometry=Polygon.from_bounds(0, 0, 128, 128),
        )
        assert isinstance(result_set, set)
        assert result_set == {0, 1, 2, 3, 4, None}

    @staticmethod
    def test_pquery_unique_with_where(fill_store, store_cls):
        """Test querying for properties with a where predicate."""
        _, store = fill_store(store_cls, ":memory:")
        result_set = store.pquery(
            select="props.get('class')",
            where="props.get('class') == 1",
        )
        assert isinstance(result_set, set)
        assert result_set == {1}

    @staticmethod
    def test_pquery_unique_with_geometry_and_where(fill_store, store_cls):
        """Test querying for properties with a geometry and where predicate."""
        _, store = fill_store(store_cls, ":memory:")
        result_set = store.pquery(
            select="props.get('class')",
            geometry=Polygon.from_bounds(0, 0, 128, 128),
            where="props.get('class') == 1",
        )
        assert isinstance(result_set, set)
        assert result_set == {1}

    @staticmethod
    def test_pquery_callable_unique(fill_store, store_cls):
        """Test querying for properties with a callable select and where."""
        _, store = fill_store(store_cls, ":memory:")
        result_set = store.pquery(
            select=lambda props: (props.get("class"),),
            where=lambda props: props.get("class") == 1,
            unique=True,
        )
        assert isinstance(result_set, set)
        assert result_set == {1}

    @staticmethod
    def test_pquery_callable_unique_no_squeeze(fill_store, store_cls):
        """Test querying for properties with a callable select and where."""
        _, store = fill_store(store_cls, ":memory:")
        result_set = store.pquery(
            select=sample_select,
            where=sample_where_1,
            unique=True,
            squeeze=False,
        )
        assert isinstance(result_set, list)
        assert result_set == [{1}]

    @staticmethod
    def test_pquery_callable_unique_multi_select(fill_store, store_cls):
        """Test querying unique properties with a callable select and where."""
        _, store = fill_store(store_cls, ":memory:")
        result_set = store.pquery(
            select=sample_multi_select,
            where=sample_where_1,
            unique=True,
        )
        assert isinstance(result_set, list)
        assert result_set == [{1}, {1}]

    @staticmethod
    def test_pquery_callable(fill_store, store_cls):
        """Test querying for properties with a callable select and where."""
        _, store = fill_store(store_cls, ":memory:")
        result_set = store.pquery(
            select=lambda props: props.get("class"),
            where=lambda props: props.get("class") == 1,
            unique=False,
        )
        assert isinstance(result_set, dict)
        assert set(result_set.values()) == {1}

    @staticmethod
    def test_pquery_callable_no_where(fill_store, store_cls):
        """Test querying for properties with callable select, no where."""
        _, store = fill_store(store_cls, ":memory:")
        result_set = store.pquery(
            select=lambda props: props.get("class"),
            where=None,
            unique=False,
        )
        assert isinstance(result_set, dict)
        assert set(result_set.values()).issubset({0, 1, 2, 3, 4, None})

    @staticmethod
    def test_pquery_pickled(fill_store, store_cls):
        """Test querying for properties with a pickled select and where."""
        _, store = fill_store(store_cls, ":memory:")

        result_set = store.pquery(
            select=pickle.dumps(sample_select),
            where=pickle.dumps(sample_where_1),
        )
        assert isinstance(result_set, set)
        assert result_set == {1}

    @staticmethod
    def test_pquery_pickled_no_squeeze(fill_store, store_cls):
        """Test querying for properties with a pickled select and where."""
        _, store = fill_store(store_cls, ":memory:")

        result_set = store.pquery(
            select=pickle.dumps(sample_select),
            where=pickle.dumps(sample_where_1),
            squeeze=False,
        )
        assert isinstance(result_set, list)
        assert result_set == [{1}]

    @staticmethod
    def test_pquery_pickled_multi_select(fill_store, store_cls):
        """Test querying for properties with a pickled select and where."""
        _, store = fill_store(store_cls, ":memory:")

        result_set = store.pquery(
            select=pickle.dumps(sample_multi_select),
            where=pickle.dumps(sample_where_1),
        )
        assert isinstance(result_set, list)
        assert result_set == [{1}, {1}]

    @staticmethod
    def test_pquery_invalid_expression_type(fill_store, store_cls):
        """Test querying for properties with an invalid expression type."""
        _, store = fill_store(store_cls, ":memory:")
        with pytest.raises(TypeError):
            _ = store.pquery(select=123, where=456)

    @staticmethod
    def test_pquery_non_matching_type(fill_store, store_cls):
        """Test querying with a non-matching type for select and where."""
        _, store = fill_store(store_cls, ":memory:")
        with pytest.raises(TypeError):
            _ = store.pquery(select=123, where="foo")

    @staticmethod
    def test_pquery_dict(fill_store, store_cls):
        """Test querying for properties."""
        _, store = fill_store(store_cls, ":memory:")
        dictionary = store.pquery(select="props.get('class')", unique=False)
        assert isinstance(dictionary, dict)
        assert len(dictionary) == len(store)
        assert all(key in store for key in dictionary)

    @staticmethod
    def test_pquery_dict_with_geometry(fill_store, store_cls):
        """Test querying for properties with a geometry intersection."""
        _, store = fill_store(store_cls, ":memory:")
        dictionary = store.pquery(
            select="props.get('class')",
            unique=False,
            geometry=Polygon.from_bounds(0, 0, 128, 128),
        )
        assert isinstance(dictionary, dict)
        assert len(dictionary) < len(store)
        assert all(key in store for key in dictionary)

    @staticmethod
    def test_is_rectangle(store_cls):
        """Test that _is_rectangle returns True only for rectangles."""
        store = store_cls()

        # Clockwise
        assert store._is_rectangle(
            *[
                (1, 0),
                (0, 0),
                (0, 1),
                (1, 1),
            ]
        )

        # Counter-clockwise
        assert store._is_rectangle(
            *[
                (1, 1),
                (0, 1),
                (0, 0),
                (1, 0),
            ]
        )

        # From shapely
        box = Polygon.from_bounds(0, 0, 10, 10)
        assert store._is_rectangle(*box.exterior.coords)

        # Fuzz
        for _ in range(100):
            box = Polygon.from_bounds(*np.random.randint(0, 100, 4))
            assert store._is_rectangle(*box.exterior.coords)

        # Failure case
        assert not store._is_rectangle(
            *[
                (1, 1.5),
                (0, 1),
                (0, 0),
                (1, 0),
            ]
        )

    @staticmethod
    def test_is_rectangle_invalid_input(store_cls):
        """Test that _is_rectangle returns False for invalid input."""
        store = store_cls()
        assert not store._is_rectangle(1, 2, 3, 4)
        assert not store._is_rectangle((0, 0), (0, 1), (1, 1), (1, 0), (2, 0))

    @staticmethod
    def test_is_right_angle(store_cls):
        """Test that _is_right_angle returns True only for right angles."""
        store = store_cls()

        # skipcq
        r"""
        c
        |
        |
        b-----a
        """
        assert store._is_right_angle(
            *[
                (1, 0),
                (0, 0),
                (0, 1),
            ]
        )

        # skipcq
        r"""
        a
        |
        |
        b-----c
        """
        assert store._is_right_angle(
            *[
                (0, 1),
                (0, 0),
                (1, 0),
            ]
        )

        # skipcq
        r"""
           c
            \
             \
        a-----b
        """
        assert not store._is_right_angle(
            *[
                (0, 0),
                (1, 0),
                (0, 1),
            ]
        )

        assert not store._is_right_angle(
            *[
                (1, 0.2),
                (0, 0),
                (0.2, 1),
            ]
        )

    @staticmethod
    def test_box_query_polygon_intersection(store_cls):
        """Test that a box query correctly checks intersections with polygons."""
        store = store_cls()

        # Add a triangle annotation
        store["foo"] = Annotation(Polygon([(0, 10), (10, 10), (10, 0)]))
        # ASCII diagram of the annotation with points labeled from a to
        # c:
        # skipcq
        r"""
        a-----b
         \    |
          \   |
           \  |
            \ |
             \|
              c
        """

        # Query where the bounding boxes overlap but the geometries do
        # not. Should return an empty result.
        # ASCII diagram of the query:
        # skipcq
        r"""
        a-----b
         \    |
          \   |
           \  |
        +--+\ |
        |  | \|
        +--+  c
        """
        result = store.query([0, 0, 4.9, 4.9], geometry_predicate="intersects")
        assert len(result) == 0

        # Query where the bounding boxes overlap and the geometries do.
        # Should return the annotation.
        # ASCII diagram of the query:
        # skipcq
        r"""
        +------+
        a-----b|
        |\    ||
        | \   ||
        |  \  ||
        |   \ ||
        |    \||
        +-----c+
        """
        result = store.query([0, 0, 11, 11], geometry_predicate="intersects")
        assert len(result) == 1

    @staticmethod
    def test_bquery_bounds(fill_store, store_cls):
        """Test querying a store with a bounding box iterable."""
        _, store = fill_store(store_cls, ":memory:")
        dictionary = store.bquery((0, 0, 1e10, 1e10))
        assert isinstance(dictionary, dict)
        assert len(dictionary) == len(store)

    @staticmethod
    def test_validate_equal_lengths(store_cls):
        """Test that equal length lists are valid."""
        store_cls._validate_equal_lengths([1, 2, 3], [1, 2, 3])
        store_cls._validate_equal_lengths()
        with pytest.raises(ValueError, match="equal length"):
            store_cls._validate_equal_lengths([1, 2, 3], [1, 2])

    @staticmethod
    def test_connection_to_path_memory(store_cls):
        """Test converting a :memory: connection to a path."""
        path = store_cls._connection_to_path(":memory:")
        assert path == Path(":memory:")

    @staticmethod
    def test_connection_to_path_type_error(store_cls):
        """Test converting an invalid type connection to a path."""
        with pytest.raises(TypeError):
            _ = store_cls._connection_to_path(123)

    @staticmethod
    def test_connection_to_path_io(store_cls, tmp_path):
        """Test converting a named file connection to a path."""
        path = tmp_path / "foo"
        with open(path, "w") as fh:
            store_cls._connection_to_path(fh)
            assert path == Path(fh.name)
