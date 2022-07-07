"""Tests for tileserver."""
from pathlib import Path

import numpy as np
import pytest

from tiatoolbox.utils.misc import imwrite
from tiatoolbox.visualization.tileserver import TileServer
from tiatoolbox.wsicore.wsireader import WSIReader


@pytest.fixture()
def app(sample_ndpi, tmp_path) -> TileServer:
    """Create a testing TileServer WSGI app."""

    # Make a low-res .jpg of the right shape to be used as
    # a low-res overlay.
    wsi = WSIReader.open(Path(sample_ndpi))
    thumb = wsi.slide_thumbnail()
    thumb_path = tmp_path / "thumb.jpg"
    imwrite(thumb_path, thumb)

    app = TileServer(
        "Testing TileServer",
        [
            str(Path(sample_ndpi)),
            str(thumb_path),
            np.zeros(wsi.slide_dimensions(1.25, "power"), dtype=np.uint8).T,
        ],
    )
    app.config.from_mapping({"TESTING": True})
    return app


def layer_get_tile(app, layer) -> None:
    """Get a single tile and check the status code and content type."""
    with app.test_client() as client:
        response = client.get(f"/layer/{layer}/zoomify/TileGroup0/0-0-0.jpg")
        assert response.status_code == 200
        assert response.content_type == "image/jpeg"


def test_get_tile(app):
    """do test on each layer"""
    layer_get_tile(app, "layer-0")
    layer_get_tile(app, "layer-1")
    layer_get_tile(app, "layer-2")


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
        assert response.content_type == "image/jpeg"
