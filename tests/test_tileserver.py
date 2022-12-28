"""Tests for tileserver."""
import json
import pathlib
import urllib
from pathlib import Path, PureWindowsPath
from typing import List, Union

import joblib
import numpy as np
import pytest
from matplotlib import cm
from shapely.geometry import LineString, Polygon
from shapely.geometry.point import Point

from tests.test_annotation_stores import cell_polygon
from tests.test_utils import make_simple_dat
from tiatoolbox.annotation.storage import Annotation, AnnotationStore, SQLiteStore
from tiatoolbox.cli.common import cli_name
from tiatoolbox.utils.misc import imwrite
from tiatoolbox.visualization.tileserver import TileServer
from tiatoolbox.wsicore.wsireader import WSIReader


def safe_str(name):
    """Make a name safe for use in a URL."""
    return urllib.parse.quote(str(PureWindowsPath(name)), safe="")


def setup_app(client):
    client.get("/tileserver/setup")
    # get the "user" cookie
    cookie = next(
        (cookie for cookie in client.cookie_jar if cookie.name == "user"), None
    )
    return cookie.value


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
def app(remote_sample, tmp_path, fill_store) -> TileServer:
    """Create a testing TileServer WSGI app."""
    # Make a low-res .jpg of the right shape to be used as
    # a low-res overlay.
    sample_svs = Path(remote_sample("svs-1-small"))
    wsi = WSIReader.open(sample_svs)
    thumb = wsi.slide_thumbnail()
    thumb_path = tmp_path / "thumb.jpg"
    imwrite(thumb_path, thumb)

    sample_store = Path(remote_sample("annotation_store_svs_1"))
    store = SQLiteStore(sample_store)
    geo_path = tmp_path / "test.geojson"
    store.to_geojson(geo_path)
    store.commit()
    store.close()

    # make tileserver with layers representing all the types
    # of things it should be able to handle
    app = TileServer(
        "Testing TileServer",
        {
            "slide": str(Path(sample_svs)),
            "tile": str(thumb_path),
            "im_array": np.zeros(wsi.slide_dimensions(1.25, "power"), dtype=np.uint8).T,
            "overlay": str(sample_store),
            "store_geojson": tmp_path / "test.geojson",
        },
    )
    app.config.from_mapping({"TESTING": True})

    return app


@pytest.fixture()
def empty_app(remote_sample, tmp_path, fill_store) -> TileServer:
    """Create a testing TileServer WSGI app with no layers."""
    app = TileServer(
        "Testing TileServer",
        {},
    )
    app.config.from_mapping({"TESTING": True})

    return app


def layer_get_tile(app, layer) -> None:
    """Get a single tile and check the status code and content type."""
    with app.test_client() as client:
        response = client.get(
            f"/tileserver/layer/{layer}/default/zoomify/TileGroup0/0-0-0@1x.jpg"
        )
        assert response.status_code == 200
        assert response.content_type == "image/webp"


def test_get_tile(app):
    """do test on each layer"""
    layer_get_tile(app, "slide")
    layer_get_tile(app, "tile")
    layer_get_tile(app, "im_array")
    layer_get_tile(app, "store_geojson")
    layer_get_tile(app, "overlay")


def layer_get_tile_404(app, layer) -> None:
    """Request a tile with an index."""
    with app.test_client() as client:
        response = client.get(
            f"/tileserver/layer/{layer}/default/zoomify/TileGroup0/10-0-0@1x.jpg"
        )
        assert response.status_code == 404
        assert response.get_data(as_text=True) == "Tile not found"


def test_get_tile_404(app):
    """do test on each layer"""
    layer_get_tile_404(app, "slide")
    layer_get_tile_404(app, "tile")
    layer_get_tile_404(app, "im_array")
    layer_get_tile_404(app, "store_geojson")
    layer_get_tile_404(app, "overlay")


def test_get_tile_layer_key_error(app) -> None:
    """Request a tile with an invalid layer key."""
    with app.test_client() as client:
        response = client.get(
            "/tileserver/layer/foo/default/zoomify/TileGroup0/0-0-0@1x.jpg"
        )
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
        response = client.get(
            "/tileserver/layer/Test/default/zoomify/TileGroup0/0-0-0@1x.jpg"
        )
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


def test_setup(app):
    """Test setup endpoint."""
    with app.test_client() as client:
        response = client.get("/tileserver/setup")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"


def test_color_prop(app):
    """Test endpoint to change property to color by."""
    with app.test_client() as client:
        response = client.put("/tileserver/change_color_prop/test_prop")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        # check that the color prop has been correctly set
        assert app.tia_pyramids["default"]["overlay"].renderer.score_prop == "test_prop"

        response = client.put("/tileserver/change_color_prop/None")
        assert app.tia_pyramids["default"]["overlay"].renderer.score_prop is None


def test_change_slide(app, remote_sample):
    """Test changing slide."""
    slide_path = remote_sample("svs-1-small")
    with app.test_client() as client:
        response = client.put(f"/tileserver/change_slide/slide/{safe_str(slide_path)}")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        # check that the slide has been correctly changed
        layer = app.tia_pyramids["default"]["slide"]
        assert layer.wsi.info.file_path == slide_path


def test_change_cmap(app):
    """Test changing colormap."""
    with app.test_client() as client:
        response = client.put("/tileserver/change_cmap/Reds")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        # check that the colormap has been correctly changed
        layer = app.tia_pyramids["default"]["overlay"]
        assert layer.renderer.mapper(0.5) == cm.get_cmap("Reds")(0.5)

        response = client.put("/tileserver/change_cmap/None")
        assert layer.renderer.mapper is None

        cdict = {"type1": (1, 0, 0), "type2": (0, 1, 0)}
        response = client.put(f"/tileserver/change_cmap/{json.dumps(cdict)}")
        assert layer.renderer.mapper("type2") == [0, 1, 0]


def test_load_save_annotations(app, tmp_path):
    """Test loading and saving annotations."""
    data = make_simple_dat()
    joblib.dump(data, tmp_path / "test.dat")
    with app.test_client() as client:
        num_annotations = len(app.tia_pyramids["default"]["overlay"].store)
        response = client.put(
            f"/tileserver/load_annotations/{safe_str(tmp_path / 'test.dat')}/{0.5}"
        )
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        assert set(json.loads(response.data)) == {0, 1, 2, 3, 4}
        # check that the extra 2 annotations have been correctly loaded
        assert len(app.tia_pyramids["default"]["overlay"].store) == num_annotations + 2

        response = client.get("/tileserver/commit/none")

    # check that the annotations have been correctly saved
    store = SQLiteStore(app.tia_pyramids["default"]["overlay"].store.path)
    assert len(store) == num_annotations + 2


def test_load_annotations_empty(empty_app, tmp_path, remote_sample):
    """Test loading annotations when no annotations are present."""
    data = make_simple_dat()
    joblib.dump(data, tmp_path / "test.dat")
    with empty_app.test_client() as client:
        user = setup_app(client)
        response = client.put(
            f"/tileserver/change_slide/slide/{safe_str(remote_sample('svs-1-small'))}"
        )
        assert response.status_code == 200
        response = client.put(
            f"/tileserver/load_annotations/{safe_str(tmp_path / 'test.dat')}/{0.5}"
        )
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        assert set(json.loads(response.data)) == {0, 1}
        # check that the 2 annotations have been correctly loaded
        assert len(empty_app.tia_pyramids[user]["overlay"].store) == 2


def test_change_overlay(empty_app, tmp_path, remote_sample):
    """Test changing overlay."""
    sample_store = Path(remote_sample("annotation_store_svs_1"))
    store = SQLiteStore(sample_store)
    num_annotations = len(store)
    geo_path = tmp_path / "test.geojson"
    store.to_geojson(geo_path)
    store.commit()
    store.close()
    overlay_path = remote_sample("svs-1-small")
    with empty_app.test_client() as client:
        user = setup_app(client)
        response = client.put(
            f"/tileserver/change_slide/slide/{safe_str(remote_sample('svs-1-small'))}"
        )
        assert response.status_code == 200
        response = client.put(f"/tileserver/change_overlay/{safe_str(geo_path)}")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        # check that the annotations have been correctly loaded
        assert len(empty_app.tia_pyramids[user]["overlay"].store) == num_annotations

        # reset tileserver and load overlay from .db instead
        response = client.get("tileserver/reset")
        response = client.put(
            f"/tileserver/change_slide/slide/{safe_str(remote_sample('svs-1-small'))}"
        )
        assert response.status_code == 200
        response = client.put(f"/tileserver/change_overlay/{safe_str(sample_store)}")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        assert set(json.loads(response.data)) == {0, 1, 2, 3, 4}
        # check that the annotations have been correctly loaded
        assert len(empty_app.tia_pyramids[user]["overlay"].store) == num_annotations

        # add another image layer
        response = client.put(f"/tileserver/change_overlay/{safe_str(overlay_path)}")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        # check that the overlay has been correctly added
        lname = f"layer{len(empty_app.tia_pyramids[user])-1}"
        layer = empty_app.tia_pyramids[user][lname]
        assert layer.wsi.info.file_path == overlay_path


def test_commit(empty_app, tmp_path, remote_sample):
    """Test committing annotations."""
    data = make_simple_dat()
    joblib.dump(data, tmp_path / "test.dat")
    with empty_app.test_client() as client:
        setup_app(client)
        response = client.put(
            f"/tileserver/change_slide/slide/{safe_str(remote_sample('svs-1-small'))}"
        )
        assert response.status_code == 200
        response = client.put(
            f"/tileserver/change_overlay/{safe_str(tmp_path / 'test.dat')}"
        )
        assert response.status_code == 200
        # commit the changes
        response = client.get(f"/tileserver/commit/{safe_str(tmp_path / 'test.db')}")

    # check that the annotations have been correctly saved
    store = SQLiteStore(tmp_path / "test.db")
    assert len(store) == 2


def test_update_renderer(app):
    """Test updating renderer."""
    with app.test_client() as client:
        response = client.put("/tileserver/update_renderer/edge_thickness/5")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        # check that the renderer has been correctly updated
        assert app.tia_pyramids["default"]["overlay"].renderer.edge_thickness == 5

        response = client.put("/tileserver/update_renderer/blur_radius/5")
        assert app.tia_pyramids["default"]["overlay"].renderer.blur_radius == 5
        assert app.overlaps["default"] == int(5 * 1.5)


def test_secondary_cmap(app):
    """Test secondary cmap."""
    with app.test_client() as client:
        response = client.put("/tileserver/change_secondary_cmap/0/prob/Reds")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        # check that the renderer has been correctly updated
        layer = app.tia_pyramids["default"]["overlay"]
        assert layer.renderer.secondary_cmap["type"] == "0"
        assert layer.renderer.secondary_cmap["score_prop"] == "prob"
        assert layer.renderer.secondary_cmap["mapper"](0.5) == cm.get_cmap("Reds")(0.5)


def test_get_props(app):
    """Test getting props."""
    with app.test_client() as client:
        response = client.get("/tileserver/get_prop_names")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        assert set(json.loads(response.data)) == {"prob", "type"}


def test_reset(app):
    """Test resetting tileserver."""
    with app.test_client() as client:
        response = client.get("/tileserver/reset")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        # check that the tileserver has been correctly reset
        assert len(app.tia_pyramids["default"]) == 0
        assert app.tia_layers["default"] == {}
