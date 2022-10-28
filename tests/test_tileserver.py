"""Tests for tileserver."""
import pathlib
from pathlib import Path
from typing import List, Union

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon
from shapely.geometry.point import Point

from tests.test_annotation_stores import cell_polygon
from tiatoolbox.annotation.storage import Annotation, AnnotationStore, SQLiteStore
from tiatoolbox.cli.common import cli_name
from tiatoolbox.utils.misc import imwrite
from tiatoolbox.visualization.tileserver import TileServer
from tiatoolbox.wsicore.wsireader import WSIReader


@pytest.fixture(scope="session")
def cell_grid() -> List[Polygon]:
    """Generate a grid of fake cell boundary polygon annotations."""
    np.random.seed(0)
    return [
        cell_polygon(((i + 0.5) * 100, (j + 0.5) * 100)) for i, j in np.ndindex(5, 5)
    ]


@pytest.fixture(scope="session")
def points_grid(spacing=60) -> List[Point]:
    """Generate a grid of fake point annotations."""
    np.random.seed(0)
    return [Point((600 + i * spacing, 600 + j * spacing)) for i, j in np.ndindex(7, 7)]


@pytest.fixture(scope="session")
def fill_store(cell_grid, points_grid):
    """Factory fixture to fill stores with test data."""

    def _fill_store(
        store_class: AnnotationStore,
        path: Union[str, pathlib.Path],
    ):
        """Fills store with random variety of annotations."""
        store = store_class(path)

        cells = [
            Annotation(cell, {"type": "cell", "prob": np.random.rand(1)[0]})
            for cell in cell_grid
        ]
        points = [
            Annotation(point, {"type": "pt", "prob": np.random.rand(1)[0]})
            for point in points_grid
        ]
        lines = [
            Annotation(
                LineString(((x, x + 500) for x in range(100, 400, 10))),
                {"type": "line", "prob": 0.75},
            )
        ]

        annotations = cells + points + lines
        keys = store.append_many(annotations)
        return keys, store

    return _fill_store


@pytest.fixture()
def app(sample_ndpi, tmp_path, fill_store) -> TileServer:
    """Create a testing TileServer WSGI app."""
    # Make a low-res .jpg of the right shape to be used as
    # a low-res overlay.
    wsi = WSIReader.open(Path(sample_ndpi))
    thumb = wsi.slide_thumbnail()
    thumb_path = tmp_path / "thumb.jpg"
    imwrite(thumb_path, thumb)

    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    geo_path = tmp_path / "test.geojson"
    store.to_geojson(geo_path)
    store.commit()
    store.close()

    # make tileserver with layers representing all the types
    # of things it should be able to handle
    app = TileServer(
        "Testing TileServer",
        [
            str(Path(sample_ndpi)),
            str(thumb_path),
            np.zeros(wsi.slide_dimensions(1.25, "power"), dtype=np.uint8).T,
            tmp_path / "test.geojson",
            str(tmp_path / "test.db"),
        ],
    )
    app.config.from_mapping({"TESTING": True})
    return app


def layer_get_tile(app, layer) -> None:
    """Get a single tile and check the status code and content type."""
    with app.test_client() as client:
        response = client.get(f"/layer/{layer}/zoomify/TileGroup0/0-0-0.jpg")
        assert response.status_code == 200
        assert response.content_type == "image/webp"


def test_get_tile(app):
    """do test on each layer"""
    layer_get_tile(app, "layer-0")
    layer_get_tile(app, "layer-1")
    layer_get_tile(app, "layer-2")
    layer_get_tile(app, "layer-3")
    layer_get_tile(app, "layer-4")


def layer_get_tile_404(app, layer) -> None:
    """Request a tile with an index."""
    with app.test_client() as client:
        response = client.get(f"/layer/{layer}/zoomify/TileGroup0/10-0-0.jpg")
        assert response.status_code == 404
        assert response.get_data(as_text=True) == "Tile not found"


def test_get_tile_404(app):
    """do test on each layer"""
    layer_get_tile_404(app, "layer-0")
    layer_get_tile_404(app, "layer-1")
    layer_get_tile_404(app, "layer-2")
    layer_get_tile_404(app, "layer-3")
    layer_get_tile_404(app, "layer-4")


def test_get_tile_layer_key_error(app) -> None:
    """Request a tile with an invalid layer key."""
    with app.test_client() as client:
        response = client.get("/layer/foo/zoomify/TileGroup0/0-0-0.jpg")
        assert response.status_code == 404
        assert response.get_data(as_text=True) == "Layer not found"


def test_get_index(app) -> None:
    """Get the index page and check that it is HTML."""
    with app.test_client() as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"


def test_create_with_dict(sample_svs):
    """test initialising with layers dict"""
    wsi = WSIReader.open(Path(sample_svs))

    app = TileServer(
        "Testing TileServer",
        {"Test": wsi},
    )
    app.config.from_mapping({"TESTING": True})
    with app.test_client() as client:
        response = client.get("/layer/Test/zoomify/TileGroup0/0-0-0.jpg")
        assert response.status_code == 200
        assert response.content_type == "image/webp"


def test_cli_name_multiple_flag():
    """Test cli_name multiple flag."""

    @cli_name()
    def dummy_fn():
        """It is empty because it's a dummy function"""

    assert "Multiple" not in dummy_fn.__click_params__[0].help

    @cli_name(multiple=True)
    def dummy_fn():
        """It is empty because it's a dummy function"""

    assert "Multiple" in dummy_fn.__click_params__[0].help
