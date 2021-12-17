"""Tests for tileserver."""
import pytest

from tiatoolbox.visualization.tileserver import TileServer
from tiatoolbox.wsicore.wsireader import WSIReader


@pytest.fixture()
def app(remote_sample) -> TileServer:
    """Create a testing TileServer WSGI app."""
    wsi = WSIReader.open(remote_sample("ndpi-1"))
    app = TileServer(title="test", layers={"ndpi": wsi})
    app.config.from_mapping({"TESTING": True})
    return app


def test_get_tile(app) -> None:
    """Get a single tile and check the status code and content type."""
    with app.test_client() as client:
        response = client.get("/layer/ndpi/zoomify/TileGroup0/0-0-0.jpg")
        assert response.status_code == 200
        assert response.content_type == "image/jpeg"


def test_get_tile_404(app) -> None:
    """Request a tile with an index."""
    with app.test_client() as client:
        response = client.get("/layer/ndpi/zoomify/TileGroup0/10-0-0.jpg")
        assert response.status_code == 404
        assert response.get_data(as_text=True) == "Tile not found"


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
