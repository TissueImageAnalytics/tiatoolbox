from pathlib import Path
from typing import Generator, Tuple, Union, List
from numbers import Number
import json
from itertools import repeat

import pytest
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely import affinity

from tiatoolbox.annotation.storage import (
    AnnotationStoreABC,
    DictionaryStore,
    SQLiteStore,
)

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
    polygon = affinity.rotate(polygon, angle, origin="centroid")
    return polygon


# Fixtures


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


def test_SQLiteStore_compile_options():
    options = SQLiteStore.compile_options()
    assert all(isinstance(x, str) for x in options)


def test_SQLiteStore_compile_options_exception(monkeypatch):
    monkeypatch.setattr(SQLiteStore, "compile_options", lambda x: [], raising=True)
    with pytest.raises(Exception):
        SQLiteStore()


def test_SQLiteStore_multiple_connection(tmp_path):
    store = SQLiteStore(tmp_path / "annotations.db")
    store2 = SQLiteStore(tmp_path / "annotations.db")
    assert len(store) == len(store2)


# Annotation Store Interface Tests (AnnotationStoreABC)


class TestStore:
    scenarios = [
        ("Dictionary", {"Store": DictionaryStore}),
        ("SQLite", {"Store": SQLiteStore}),
    ]

    def test_open_close(self, fill_store, tmp_path, Store):
        path = tmp_path / "polygons"
        indexes, store = fill_store(Store, path)
        store.close()
        store2 = Store.open(path)
        assert len(store2) == len(indexes)

    def test_append_many(self, cell_grid, tmp_path, Store):
        store = Store(tmp_path / "polygons")
        indexes = store.append_many(
            cell_grid, ({"class": x} for x in np.random.randint(0, 7, len(cell_grid)))
        )
        assert len(indexes) == len(cell_grid)

    def test_query_bbox(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        results = store.query((0, 0, 25, 25))
        assert len(results) == 4

    def test_iquery_bbox(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        results = store.iquery((0, 0, 25, 25))
        assert len(results) == 4
        assert all(isinstance(index, int) for index in results)

    def test_iquery_polygon(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        results = store.iquery(Polygon([(0, 0), (0, 25), (1, 1), (25, 0)]))
        assert len(results) == 3
        assert all(isinstance(index, int) for index in results)

    def test_update(self, fill_store, tmp_path, Store):
        indexes, store = fill_store(Store, tmp_path / "polygon.db")
        index = indexes[0]
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        # Geometry update
        store.update(index, {"geometry": new_geometry})
        assert store[index][0] == new_geometry
        # Properties update
        store.update(index, {"abc": 123})
        assert store[index][1]["abc"] == 123

    def test_update_many(self, fill_store, tmp_path, Store):
        indexes, store = fill_store(Store, tmp_path / "polygon.db")
        # Geometry update
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        store.update_many(indexes, repeat({"geometry": new_geometry}))
        # Properties update
        store.update_many(indexes, repeat({"abc": 123}))

        for _, index in enumerate(indexes[:10]):
            assert store[index][0] == new_geometry
            assert store[index][1]["abc"] == 123

    def test_keys(self, fill_store, tmp_path, Store):
        indexes, store = fill_store(Store, tmp_path / "polygon.db")
        indexes = list(indexes)
        assert len(list(store.keys())) == len(indexes)
        assert isinstance(list(store.keys())[0], type(indexes[0]))

    def test_remove(self, fill_store, tmp_path, Store):
        indexes, store = fill_store(Store, tmp_path / "polygon.db")
        index = indexes[0]
        store.remove(index)
        assert len(store) == FILLED_LEN - 1

    def test_delitem(self, fill_store, tmp_path, Store):
        indexes, store = fill_store(Store, tmp_path / "polygon.db")
        index = indexes[0]
        del store[index]
        assert len(store) == FILLED_LEN - 1

    def test_remove_many(self, fill_store, tmp_path, Store):
        indexes, store = fill_store(Store, tmp_path / "polygon.db")
        store.remove_many(indexes)
        assert len(store) == 0

    def test_len(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        assert len(store) == FILLED_LEN

    def test_contains(self, fill_store, tmp_path, Store):
        indexes, store = fill_store(Store, tmp_path / "polygon.db")
        for index in indexes:
            assert index in store

    def test_iter(self, fill_store, tmp_path, Store):
        indexes, store = fill_store(Store, tmp_path / "polygon.db")
        for index in store:
            assert index in indexes

    def test_getitem(self, fill_store, tmp_path, sample_triangle, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        index = store.append(sample_triangle)
        geometry, properties = store[index]
        assert geometry == sample_triangle
        assert properties == {}

    def test_setitem(self, fill_store, tmp_path, sample_triangle, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        index = store.append(sample_triangle)
        new_geometry = Polygon([(0, 0), (1, 1), (2, 2)])
        new_properties = {"abc": 123}
        store[index] = (new_geometry, new_properties)
        assert store[index] == (new_geometry, new_properties)

    def test_getitem_setitem_cycle(self, fill_store, tmp_path, sample_triangle, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        index = store.append(sample_triangle, {"class": 0})
        geometry, properties = store[index]
        store[index] = (geometry, properties)
        assert store[index] == (geometry, properties)

    def test_from_dataframe(self, cell_grid, Store):
        df = pd.DataFrame.from_records(
            [
                {
                    "geometry": cell,
                    "row_id": n,
                }
                for n, cell in enumerate(cell_grid)
            ]
        )
        store = Store.from_dataframe(df)
        keys = list(store.keys())
        _, properties = store[keys[0]]
        assert "row_id" in properties

    def test_to_dataframe(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        df = store.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == FILLED_LEN
        assert "geometry" in df.columns
        assert df.index.name == "index"
        assert isinstance(df.geometry.iloc[0], Polygon)

    def test_features(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        features = store.features()
        assert isinstance(features, Generator)
        features = list(features)
        assert len(features) == FILLED_LEN
        assert isinstance(features[0], dict)
        assert all(
            {"type", "geometry", "properties"} == set(f.keys()) for f in features
        )

    def test_to_geodict(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        geodict = store.to_geodict()
        assert isinstance(geodict, dict)
        assert "features" in geodict
        assert "type" in geodict
        assert geodict["type"] == "FeatureCollection"
        assert geodict["features"] == list(store.features())

    def test_from_geojson_str(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        geojson = store.to_geojson()
        store2 = Store.from_geojson(geojson)
        assert len(store) == len(store2)

    def test_from_geojson_file(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        store.to_geojson(tmp_path / "polygon.json")
        with open(tmp_path / "polygon.json", "r") as file_handle:
            store2 = Store.from_geojson(file_handle)
        assert len(store) == len(store2)

    def test_from_geojson_path(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        store.to_geojson(tmp_path / "polygon.json")
        store2 = Store.from_geojson(tmp_path / "polygon.json")
        assert len(store) == len(store2)

    def test_to_geojson_str(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        geojson = store.to_geojson()
        assert isinstance(geojson, str)
        geodict = json.loads(geojson)
        assert "features" in geodict
        assert "type" in geodict
        assert geodict["type"] == "FeatureCollection"
        assert len(geodict["features"]) == len(list(store.features()))

    def test_to_geojson_file(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        with open(tmp_path / "polygon.json", "w") as fh:
            geojson = store.to_geojson(fp=fh)
        assert geojson is None
        with open(tmp_path / "polygon.json", "r") as fh:
            geodict = json.load(fh)
        assert "features" in geodict
        assert "type" in geodict
        assert geodict["type"] == "FeatureCollection"
        assert len(geodict["features"]) == len(list(store.features()))

    def test_to_geojson_path(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        geojson = store.to_geojson(fp=tmp_path / "polygon.json")
        assert geojson is None
        with open(tmp_path / "polygon.json", "r") as fh:
            geodict = json.load(fh)
        assert "features" in geodict
        assert "type" in geodict
        assert geodict["type"] == "FeatureCollection"
        assert len(geodict["features"]) == len(list(store.features()))

    def test_to_ldjson_str(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        ldjson = store.to_ldjson()
        for line in ldjson.split():
            assert isinstance(line, str)
            feature = json.loads(line)
            assert "type" in feature
            assert "geometry" in feature
            assert "properties" in feature

    def test_to_ldjson_file(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
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

    def test_to_ldjson_path(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, tmp_path / "polygon.db")
        ldjson = store.to_ldjson(fp=tmp_path / "polygon.ldjson")
        assert ldjson is None
        with open(tmp_path / "polygon.ldjson", "r") as fh:
            for line in fh.readlines():
                assert isinstance(line, str)
                feature = json.loads(line)
                assert "type" in feature
                assert "geometry" in feature
                assert "properties" in feature

    def test_dump(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, ":memory:")
        store.dump(tmp_path / "dump_test.db")
        assert (tmp_path / "dump_test.db").stat().st_size > 0

    def test_dump_file_handle(self, fill_store, tmp_path, Store):
        _, store = fill_store(Store, ":memory:")
        with open(tmp_path / "dump_test.db", "w") as fh:
            store.dump(fh)
        assert (tmp_path / "dump_test.db").stat().st_size > 0

    def test_dumps(self, fill_store, Store):
        _, store = fill_store(Store, ":memory:")
        string = store.dumps()
        assert isinstance(string, str)
