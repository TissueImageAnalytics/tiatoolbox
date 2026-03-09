"""Test the tiatoolbox visualization tool."""

from __future__ import annotations

import importlib
import importlib.resources as importlib_resources
import io
import json
import multiprocessing
import re
import shutil
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import bokeh.models as bkmodels
import matplotlib.pyplot as plt
import numpy as np
import pytest
import requests
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.events import ButtonClick, DoubleTap, MenuItemClick
from bokeh.models import ColorBar
from bokeh.models.tiles import WMTSTileSource
from bokeh.plotting import figure
from flask_cors import CORS
from matplotlib import colormaps
from PIL import Image
from scipy.ndimage import label

from tiatoolbox.data import _fetch_remote_sample
from tiatoolbox.visualization.bokeh_app import app_hooks, main
from tiatoolbox.visualization.tileserver import TileServer
from tiatoolbox.visualization.ui_utils import get_level_by_extent

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Generator

    from bokeh.document import Document

# constants
BOKEH_PATH = importlib_resources.files("tiatoolbox.visualization.bokeh_app")
FILLED = 0
MICRON_FORMATTER = 1
GRIDLINES = 2


# Helper function
def fetch_sample_to_dir(key: str, target_dir: Path) -> Path:
    """Fetch a remote sample and and ensure it resides directly in ``target_dir``.

     The sample is downloaded and, if it is not already located directly in
    ``target_dir``, it is moved there. If it is already in ``target_dir``,
    it is left in place and its path is returned.

    Args:
        key (str): The name of the resource to fetch.
        target_dir (Path): The directory where the file should be placed.

    Returns:
        Path: The path to the file in the target directory.
    """
    # Download to a temp location
    downloaded_path = _fetch_remote_sample(key, target_dir)

    # If the file is already in target_dir directly, return it
    if downloaded_path.parent == target_dir:
        return downloaded_path

    # Otherwise, move it to target_dir
    target_path = target_dir / downloaded_path.name
    if not target_path.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded_path), str(target_path))

    return target_path


class _DummySessionContext:
    """Simple shim matching the subset of Bokeh's SessionContext we use."""

    def __init__(self: _DummySessionContext, user: str) -> None:
        self.request = SimpleNamespace(arguments={"user": user})


# helper functions and fixtures
def get_tile(layer: str, x: float, y: float, z: float, *, show: bool) -> np.ndarray:
    """Get a tile from the server.

    Args:
        layer (str):
            The layer to get the tile from.
        x (float):
            The x coordinate of the tile.
        y (float):
            The y coordinate of the tile.
        z (float):
            The zoom level of the tile.
        show (bool):
            Whether to show the tile.

    """
    source = main.UI["p"].renderers[main.UI["vstate"].layer_dict[layer]].tile_source
    url = source.url
    # replace {x}, {y}, {z} with tile coordinates
    url = url.replace("{x}", str(x)).replace("{y}", str(y)).replace("{z}", str(z))
    im = io.BytesIO(requests.get(url, timeout=100).content)
    if show:
        plt.imshow(np.array(Image.open(im)))
        plt.show()
    return np.array(Image.open(im))


def get_renderer_prop(prop: str) -> json:
    """Get a renderer property from the server.

    Args:
        prop (str):
            The property to get.

    """
    resp = main.UI["s"].get(
        f"http://{main.host2}:{main.port}/tileserver/renderer/{prop}",
    )
    return resp.json()


def get_channel_ui_elements() -> tuple[
    bkmodels.DataTable,
    bkmodels.DataTable,
    bkmodels.ColorPicker,
    bkmodels.Button,
    bkmodels.Slider,
]:
    """Return channel selection UI widgets."""
    channel_select = main.UI["channel_select"]
    inner_column = channel_select.children[1]
    table_row = inner_column.children[0]
    channel_table, color_table = table_row.children
    picker_row = inner_column.children[1]
    color_picker, apply_button = picker_row.children
    enhance_slider = inner_column.children[2]
    return channel_table, color_table, color_picker, apply_button, enhance_slider


@pytest.fixture(scope="module", autouse=True)
def annotation_path(data_path: dict[str, Path]) -> dict[str, object]:
    """Download some testing slides and overlays.

    Set up a dictionary defining the paths to the files
    that can be grabbed as a fixture to refer to during tests.

    """
    data_path["slide1"] = fetch_sample_to_dir(
        "svs-1-small",
        data_path["base_path"] / "slides",
    )
    data_path["slide2"] = fetch_sample_to_dir(
        "ndpi-1",
        data_path["base_path"] / "slides",
    )
    data_path["slide3"] = fetch_sample_to_dir(
        "patch-extraction-vf",
        data_path["base_path"] / "slides",
    )
    data_path["qptiff"] = fetch_sample_to_dir(
        "qptiff_sample",
        data_path["base_path"] / "slides",
    )
    data_path["annotations"] = fetch_sample_to_dir(
        "annotation_store_svs_1",
        data_path["base_path"] / "overlays",
    )
    data_path["graph"] = fetch_sample_to_dir(
        "graph_svs_1",
        data_path["base_path"] / "overlays",
    )
    data_path["graph_feats"] = fetch_sample_to_dir(
        "graph_svs_1_feats",
        data_path["base_path"] / "overlays",
    )
    data_path["img_overlay"] = fetch_sample_to_dir(
        "svs_1_rendered_annotations_jpg",
        data_path["base_path"] / "overlays",
    )
    data_path["geojson_anns"] = fetch_sample_to_dir(
        "geojson_cmu_1",
        data_path["base_path"] / "overlays",
    )
    data_path["dat_anns"] = fetch_sample_to_dir(
        "annotation_dat_svs_1",
        data_path["base_path"] / "overlays",
    )
    data_path["affine_trans"] = (
        data_path["base_path"] / "overlays" / (data_path["slide1"].stem + ".npy")
    )
    # save eye as test identity transform
    np.save(data_path["affine_trans"], np.eye(3))
    data_path["config"] = fetch_sample_to_dir(
        "config_2",
        data_path["base_path"] / "overlays",
    )
    return data_path


def run_app() -> None:
    """Helper function to launch a tileserver."""
    app = TileServer(
        title="Tiatoolbox TileServer",
        layers={},
    )
    app.json.sort_keys = False
    CORS(app, send_wildcard=True)
    app.run(host="127.0.0.1", port=int(main.port), threaded=True)


@pytest.fixture(scope="module")
def doc(data_path: dict[str, object]) -> Generator[Document, object, None]:
    """Create a test document for the visualization tool."""
    # start tile server
    p = multiprocessing.Process(target=run_app, daemon=True)
    p.start()
    # wait until server is ready
    start = time.time()
    url = f"http://127.0.0.1:{main.port}/tileserver/session_id"
    while True:
        try:
            resp = requests.get(url, timeout=1)
            if resp.status_code == 200:
                break
        except requests.RequestException:
            pass
        if time.time() - start > 10:
            p.terminate()
            msg = f"Tileserver failed to start within 10s: {url}"
            raise RuntimeError(msg)
        time.sleep(0.2)

    main.doc_config.set_sys_args(argv=["dummy_str", str(data_path["base_path"])])
    handler = FunctionHandler(main.doc_config.setup_doc)
    app = Application(handler)
    yield app.create_document()
    p.terminate()


# test some utility functions


def test_to_num() -> None:
    """Test the to_num function."""
    assert main.to_num("1") == 1
    assert main.to_num("1.0") == 1.0
    assert main.to_num("1.0e-3") == 1.0e-3
    assert main.to_num(2) == 2
    assert main.to_num("None") is None


def test_get_level_by_extent() -> None:
    """Test the get_level_by_extent function."""
    max_lev = 10
    assert get_level_by_extent((1000, 1000, 1100, 1100)) == max_lev
    assert get_level_by_extent((1000, 1000, 1000000, 1000000)) == 0


# test the bokeh app


def test_roots(doc: Document) -> None:
    """Test that the document has the correct number of roots."""
    # should be 4 roots: main window, controls, slide_info, popup table
    assert len(doc.roots) == 4


def test_config_loaded(data_path: pytest.TempPathFactory) -> None:
    """Test that the config is loaded correctly."""
    # config should be loaded
    loaded_config = main.doc_config.config
    with Path(data_path["config"]).open() as f:
        file_config = json.load(f)

    # check that all keys in file_config are in doc_config
    # and that the values are the same
    for key in file_config:
        assert key in loaded_config
        assert loaded_config[key] == file_config[key]


def test_slide_select(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test slide selection."""
    slide_select = doc.get_model_by_name("slide_select0")
    # check there are four available slides
    assert len(slide_select.options) == 4
    assert slide_select.options[0][0] == data_path["slide1"].name

    # select a slide and check it is loaded
    slide_select.value = ["CMU-1.ndpi"]
    assert main.UI["vstate"].slide_path == data_path["slide2"]

    # check selecting nothing has no effect
    slide_select.value = []
    assert main.UI["vstate"].slide_path == data_path["slide2"]

    # select a slide and check it is loaded
    slide_select.value = ["TCGA-HE-7130-01Z-00-DX1.png"]
    assert main.UI["vstate"].slide_path == data_path["slide3"]


def test_dual_window(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test adding a second window."""
    control_tabs = doc.get_model_by_name("ui_layout")
    doc.get_model_by_name("slide_windows")
    control_tabs.active = 1
    slide_select = doc.get_model_by_name("slide_select1")
    assert len(slide_select.options) == 4
    assert slide_select.options[0][0] == data_path["slide1"].name

    control_tabs.active = 0
    assert main.UI.active == 0


def test_remove_dual_window(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test removing a second window."""
    control_tabs = doc.get_model_by_name("ui_layout")
    slide_wins = doc.get_model_by_name("slide_windows")
    assert len(slide_wins.children) == 2
    # remove the second window
    control_tabs.tabs.pop()
    assert len(slide_wins.children) == 1

    slide_select = doc.get_model_by_name("slide_select0")
    slide_select.value = [data_path["slide1"].name]
    assert main.UI["vstate"].slide_path == data_path["slide1"]


def test_add_slide_layer(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test adding a non-annotation slide layer."""
    slide_select = doc.get_model_by_name("slide_select0")
    slide_select.value = [data_path["slide1"].name]

    layer_drop = doc.get_model_by_name("layer_drop0")
    slide_layer_path = str(data_path["slide1"])

    click = MenuItemClick(layer_drop, slide_layer_path)
    layer_drop._trigger_event(click)

    assert len(layer_drop.menu) == 6


def test_transform_overlay(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test adding a transform overlay."""

    class DummyResponse:
        """Dummy response for mocking requests.put()."""

        text = json.dumps("dummy.npy")
        status_code = 200

    def dummy_put(*_: object, **__: object) -> DummyResponse:
        """Dummy put method to replace requests.Session.put()."""
        return DummyResponse()

    # Patch the method on the Session class
    old_put = requests.sessions.Session.put
    requests.sessions.Session.put = dummy_put

    try:
        layer_drop = doc.get_model_by_name("layer_drop0")
        affine_layer_path = str(data_path["affine_trans"])

        click = MenuItemClick(layer_drop, affine_layer_path)
        layer_drop._trigger_event(click)

        assert len(layer_drop.menu) == 6

    finally:
        requests.sessions.Session.put = old_put


def test_add_annotation_layer(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test adding annotation layers."""
    # test loading a geojson file.
    slide_select = doc.get_model_by_name("slide_select0")
    slide_select.value = [data_path["slide2"].name]
    layer_drop = doc.get_model_by_name("layer_drop0")
    # trigger an event to select the geojson file
    click = MenuItemClick(layer_drop, str(data_path["geojson_anns"]))
    layer_drop._trigger_event(click)
    assert set(main.UI["vstate"].types) == {"nucleus", "cell", "annotation"}

    # test save functionality
    save_button = doc.get_model_by_name("save_button0")
    click = ButtonClick(save_button)
    save_button._trigger_event(click)
    saved_path = (
        data_path["base_path"]
        / "overlays"
        / (data_path["slide2"].stem + "_saved_anns.db")
    )
    assert saved_path.exists()

    # test the name2type function.
    assert main.name2type("annotation") == '"annotation"'

    slide_select.value = [data_path["slide1"].name]
    saved_path.unlink()  # clean up saved file

    # test loading an annotation store
    layer_drop = doc.get_model_by_name("layer_drop0")
    assert len(layer_drop.menu) == 6
    n_renderers = len(doc.get_model_by_name("slide_windows").children[0].renderers)
    # trigger an event to select the annotation .db file
    click = MenuItemClick(layer_drop, str(data_path["annotations"]))
    layer_drop._trigger_event(click)
    # should be one more renderer now
    assert len(doc.get_model_by_name("slide_windows").children[0].renderers) == (
        n_renderers + 1
    )
    # we should have got the types of annotations back from the server too
    assert main.UI["vstate"].types == ["0", "1", "2", "3", "4"]

    # test get_mapper function
    cmap_dict = main.get_mapper_for_prop("type")
    assert set(cmap_dict.keys()) == {0, 1, 2, 3, 4}


def test_tap_query() -> None:
    """Test the double tap query functionality."""
    # trigger a tap event
    assert len(main.popup_table.source.data["property"]) == 0
    main.UI["p"]._trigger_event(
        DoubleTap(
            main.UI["p"],
            x=1138.52,
            y=-1881.5,
        ),
    )
    # the tapped annotation has 2 properties
    assert len(main.popup_table.source.data["property"]) == 2
    assert len(main.popup_table.source.data["value"]) == 2


def test_cprop_input(doc: Document) -> None:
    """Test changing the color property."""
    cprop_input = doc.get_model_by_name("cprop0")
    cmap_select = doc.get_model_by_name("cmap0")
    cprop_input.value = ["prob"]
    # as prob is continuous, cmap should be set to whatever cmap is selected
    assert main.UI["vstate"].cprop == "prob"
    assert main.UI["color_bar"].color_mapper.palette[0] == main.rgb2hex(
        colormaps[cmap_select.value](0),
    )

    # check deselecting has no effect
    cprop_input.value = []
    assert main.UI["vstate"].cprop == "prob"

    cprop_input.value = ["type"]
    # as type is discrete, cmap should be a dict mapping types to colors
    assert isinstance(main.UI["vstate"].mapper, dict)
    assert list(main.UI["vstate"].mapper.keys()) == list(
        main.UI["vstate"].orig_types.values(),
    )

    main.UI["vstate"].to_update.add("layer_1")
    # check update state
    assert main.UI["vstate"].update_state == 1
    # simulate server ticks
    main.update()
    # pending change so update state should be promoted to 2
    assert main.UI["vstate"].update_state == 2
    main.update()
    # no more changes added so tile update has been triggered and update state reset
    assert main.UI["vstate"].update_state == 0


def test_type_cmap_select(doc: Document) -> None:
    """Test changing the type cmap."""
    cmap_select = doc.get_model_by_name("type_cmap0")
    cmap_select.value = ["prob"]
    # select a type to assign the cmap to
    cmap_select.value = ["prob", "0"]
    # set edge thickness to 0 so the edges don't add an extra colour
    spinner = doc.get_model_by_name("edge_size0")
    spinner.value = 0
    im = get_tile("overlay", 1, 2, 2, show=False)
    # check there are more than just num_types unique colors in the image,
    # as we have mapped type 0 to a continuous cmap on prob
    assert len(np.unique(im.sum(axis=2))) > 10

    # remove the type cmap
    cmap_select.value = []
    resp = main.UI["s"].get(
        f"http://{main.host2}:{main.port}/tileserver/secondary_cmap"
    )
    assert resp.json()["score_prop"] == "None"

    # check callback works regardless of order
    cmap_select.value = ["0"]
    cmap_select.value = ["0", "prob"]
    resp = main.UI["s"].get(
        f"http://{main.host2}:{main.port}/tileserver/secondary_cmap"
    )
    assert resp.json()["score_prop"] == "prob"


def test_load_graph(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test loading a graph."""
    layer_drop = doc.get_model_by_name("layer_drop0")
    # trigger an event to select the graph file
    click = MenuItemClick(layer_drop, str(data_path["graph"]))
    layer_drop._trigger_event(click)
    # we should have 2144 nodes in the node_source now
    assert len(main.UI["node_source"].data["x_"]) == 2144


def test_graph_with_feats(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test loading a graph with features."""
    layer_drop = doc.get_model_by_name("layer_drop0")
    # trigger an event to select the graph .json file
    click = MenuItemClick(layer_drop, str(data_path["graph_feats"]))
    layer_drop._trigger_event(click)
    # we should have keys for the features in node data source now
    for i in range(10):
        assert f"feat_{i}" in main.UI["node_source"].data

    # test setting a node feat to color by
    cmap_select = doc.get_model_by_name("type_cmap0")
    cmap_select.value = ["graph_overlay"]
    cmap_select.value = ["graph_overlay", "feat_0"]

    node_cm = colormaps["viridis"]
    assert main.UI["node_source"].data["node_color_"][0] == main.rgb2hex(
        node_cm(main.UI["node_source"].data["feat_0"][0]),
    )

    # test prop that doesnt exist in graph has no effect
    cmap_select.value = ["graph_overlay", "prob"]
    assert main.UI["node_source"].data["node_color_"][0] == main.rgb2hex(
        node_cm(main.UI["node_source"].data["feat_0"][0]),
    )

    # test graph overlay option remains on loading new overlay
    click = MenuItemClick(layer_drop, str(data_path["annotations"]))
    layer_drop._trigger_event(click)
    assert "graph_overlay" in cmap_select.options


def test_load_img_overlay(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test loading an image overlay."""
    layer_drop = doc.get_model_by_name("layer_drop0")
    # trigger an event to select the image overlay
    click = MenuItemClick(layer_drop, str(data_path["img_overlay"]))
    layer_drop._trigger_event(click)
    l_name = data_path["img_overlay"].stem
    layer_slider = doc.get_model_by_name(f"{l_name}_slider")
    assert layer_slider is not None

    # check alpha controls
    type_column_list = doc.get_model_by_name("type_column0").children
    # last one will be image overlay controls
    assert type_column_list[-1].active
    # toggle off and check alpha is 0
    type_column_list[-1].active = False
    assert main.UI["p"].renderers[main.UI["vstate"].layer_dict[l_name]].alpha == 0
    # toggle back on and check alpha is back to default 0.75
    type_column_list[-1].active = True
    assert main.UI["p"].renderers[main.UI["vstate"].layer_dict[l_name]].alpha == 0.75
    # set alpha to 0.4
    layer_slider.value = 0.4
    # check that the alpha values have been set correctly
    assert main.UI["p"].renderers[main.UI["vstate"].layer_dict[l_name]].alpha == 0.4

    # check loading a new layer with same stem uses full file name to disambiguate
    click = MenuItemClick(layer_drop, str(data_path["annotations"]))
    layer_drop._trigger_event(click)
    click = MenuItemClick(layer_drop, str(data_path["img_overlay"]))
    layer_drop._trigger_event(click)
    full_name = data_path["img_overlay"].name
    assert full_name in main.UI["vstate"].layer_dict


def test_hovernet_on_box(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test running hovernet on a box."""
    slide_select = doc.get_model_by_name("slide_select0")
    slide_select.value = [data_path["slide2"].name]
    run_button = doc.get_model_by_name("to_model0")
    assert len(main.UI["color_column"].children) == 0
    slide_select.value = [data_path["slide1"].name]
    # set up a box selection
    main.UI["box_source"].data = {
        "x": [1200],
        "y": [-2000],
        "width": [400],
        "height": [400],
    }

    # select hovernet model and run it on box
    model_select = doc.get_model_by_name("model_drop0")
    model_select.value = "hovernet"

    click = ButtonClick(run_button)
    run_button._trigger_event(click)
    im = get_tile("overlay", 4, 8, 4, show=False)
    _, num = label(np.any(im[:, :, :3], axis=2))
    # check there are multiple cells being detected
    assert len(main.UI["color_column"].children) > 3
    assert num > 10

    # load an overlay with different types
    cprop_select = doc.get_model_by_name("cprop0")
    cprop_select.value = ["prob"]
    layer_drop = doc.get_model_by_name("layer_drop0")
    click = MenuItemClick(layer_drop, str(data_path["dat_anns"]))
    layer_drop._trigger_event(click)
    assert main.UI["vstate"].types == ["annotation"]
    # check the per-type ui controls have been updated
    assert len(main.UI["color_column"].children) == 1
    assert len(main.UI["type_column"].children) == 1


def test_sam_segment(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test running SAM on a box."""
    slide_select = doc.get_model_by_name("slide_select0")
    slide_select.value = [data_path["slide2"].name]
    run_button = doc.get_model_by_name("to_model0")
    assert len(main.UI["color_column"].children) == 0
    slide_select.value = [data_path["slide1"].name]
    # set up a box selection
    main.UI["box_source"].data = {
        "x": [1200],
        "y": [-2000],
        "width": [400],
        "height": [400],
    }

    # select SAM model and run it on box
    model_select = doc.get_model_by_name("model_drop0")
    model_select.value = "SAM"

    click = ButtonClick(run_button)
    run_button._trigger_event(click)
    assert len(main.UI["color_column"].children) > 0

    # test save functionality
    save_button = doc.get_model_by_name("save_button0")
    click = ButtonClick(save_button)
    save_button._trigger_event(click)
    saved_path = (
        data_path["base_path"]
        / "overlays"
        / (data_path["slide1"].stem + "_saved_anns.db")
    )
    assert saved_path.exists()

    # load an overlay with different types
    cprop_select = doc.get_model_by_name("cprop0")
    cprop_select.value = ["prob"]
    layer_drop = doc.get_model_by_name("layer_drop0")
    click = MenuItemClick(layer_drop, str(data_path["dat_anns"]))
    layer_drop._trigger_event(click)
    assert main.UI["vstate"].types == ["annotation"]
    # check the per-type ui controls have been updated
    assert len(main.UI["color_column"].children) == 1
    assert len(main.UI["type_column"].children) == 1


def test_alpha_sliders(doc: Document) -> None:
    """Test sliders for adjusting slide and overlay alpha."""
    slide_alpha = doc.get_model_by_name("slide_alpha0")
    overlay_alpha = doc.get_model_by_name("overlay_alpha0")

    # set alpha to 0.5
    slide_alpha.value = 0.5
    overlay_alpha.value = 0.5
    # check that the alpha values have been set correctly
    assert main.UI["p"].renderers[0].alpha == 0.5
    assert main.UI["p"].renderers[main.UI["vstate"].layer_dict["overlay"]].alpha == 0.5


def test_alpha_buttons(doc: Document) -> None:
    """Test buttons for toggling slide and overlay alpha."""
    slide_toggle = doc.get_model_by_name("slide_toggle0")
    overlay_toggle = doc.get_model_by_name("overlay_toggle0")
    # clicking the button should set alpha to 0
    slide_toggle.active = False
    assert main.UI["p"].renderers[0].alpha == 0
    overlay_toggle.active = False
    assert main.UI["p"].renderers[main.UI["vstate"].layer_dict["overlay"]].alpha == 0

    # clicking again should set alpha back to previous value
    slide_toggle.active = True
    assert main.UI["p"].renderers[0].alpha == 0.5
    overlay_toggle.active = True
    assert main.UI["p"].renderers[main.UI["vstate"].layer_dict["overlay"]].alpha == 0.5


def test_type_select(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test selecting/deselecting specific types."""
    # load annotation layer
    layer_drop = doc.get_model_by_name("layer_drop0")
    click = MenuItemClick(layer_drop, str(data_path["annotations"]))
    layer_drop._trigger_event(click)
    time.sleep(1)
    im = get_tile("overlay", 4, 8, 4, show=False)
    _, num_before = label(np.any(im[:, :, :3], axis=2))
    type_column_list = doc.get_model_by_name("type_column0").children
    # click on the first and last to deselect them
    type_column_list[0].active = False
    type_column_list[-1].active = False
    # check that the number of cells has decreased
    im = get_tile("overlay", 4, 8, 4, show=False)
    _, num_after = label(np.any(im[:, :, :3], axis=2))
    assert num_after < num_before

    # turn off all the types
    for type_toggle in type_column_list:
        type_toggle.active = False
    # check that there are no cells
    im = get_tile("overlay", 4, 8, 4, show=False)
    _, num_after = label(np.any(im[:, :, :3], axis=2))
    assert num_after == 0

    # reselect them
    for type_toggle in type_column_list:
        type_toggle.active = True
    # check we are back to original number of cells
    im = get_tile("overlay", 4, 8, 4, show=False)
    _, num_after = label(np.any(im[:, :, :3], axis=2))
    assert num_after == num_before


def test_color_boxes(doc: Document) -> None:
    """Test color boxes for setting type colors."""
    color_column_list = doc.get_model_by_name("color_column0").children
    # set type 0 to red
    color_column_list[0].color = "#ff0000"
    # set type 1 to blue
    color_column_list[1].color = "#0000ff"
    # check the mapper matches the new colors
    assert main.UI["vstate"].mapper[0] == (1, 0, 0, 1)
    assert main.UI["vstate"].mapper[1] == (0, 0, 1, 1)

    cprop_select = doc.get_model_by_name("cprop0")
    cprop_select.value = ["prob"]
    # set type 1 to green
    color_column_list[1].color = "#00ff00"
    assert main.UI["vstate"].mapper[1] == (0, 1, 0, 1)


def test_node_and_edge_alpha(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test sliders for adjusting graph node and edge alpha."""
    layer_drop = doc.get_model_by_name("layer_drop0")
    # trigger an event to select the graph .db file
    click = MenuItemClick(layer_drop, str(data_path["graph"]))
    layer_drop._trigger_event(click)

    type_column_list = doc.get_model_by_name("type_column0").children
    color_column_list = doc.get_model_by_name("color_column0").children
    # the last 2 will be edge and node controls
    # by default nodes are visible, edges are not
    assert not type_column_list[-2].active
    assert type_column_list[-1].active
    type_column_list[-1].active = False
    type_column_list[-2].active = True
    # check that the alpha values have been set correctly
    assert (
        main.UI["p"].renderers[main.UI["vstate"].layer_dict["nodes"]].glyph.fill_alpha
        == 0
    )
    assert main.UI["p"].renderers[main.UI["vstate"].layer_dict["edges"]].visible is True
    type_column_list[-1].active = True
    color_column_list[-2].value = 0.3
    color_column_list[-1].value = 0.4
    # check that the alpha values have been set correctly
    assert (
        main.UI["p"].renderers[main.UI["vstate"].layer_dict["nodes"]].glyph.fill_alpha
        == 0.4
    )
    assert main.UI["p"].renderers[main.UI["vstate"].layer_dict["edges"]].visible is True
    assert (
        main.UI["p"].renderers[main.UI["vstate"].layer_dict["edges"]].glyph.line_alpha
        == 0.3
    )
    # turn edges back off and check
    type_column_list[-2].active = False
    assert (
        main.UI["p"].renderers[main.UI["vstate"].layer_dict["edges"]].visible is False
    )
    # check changing overlay alpha doesnt affect graph alpha
    overlay_alpha = doc.get_model_by_name("overlay_alpha0")
    overlay_alpha.value = 0.5
    assert (
        main.UI["p"].renderers[main.UI["vstate"].layer_dict["nodes"]].glyph.fill_alpha
        == 0.4
    )
    # same with overlay toggle
    overlay_toggle = doc.get_model_by_name("overlay_toggle0")
    overlay_toggle.active = False
    assert (
        main.UI["p"].renderers[main.UI["vstate"].layer_dict["nodes"]].glyph.fill_alpha
        == 0.4
    )


def test_pt_size_spinner(doc: Document) -> None:
    """Test setting point size for graph nodes."""
    pt_size_spinner = doc.get_model_by_name("pt_size0")
    # set the point size to 10
    pt_size_spinner.value = 10
    # check that the point size has been set correctly
    assert (
        main.UI["p"].renderers[main.UI["vstate"].layer_dict["nodes"]].glyph.radius
        == 2 * 10
    )


def test_filter_box(doc: Document) -> None:
    """Test annotation filter box."""
    filter_input = doc.get_model_by_name("filter0")
    im = get_tile("overlay", 4, 8, 4, show=False)
    _, num_before = label(np.any(im[:, :, :3], axis=2))
    # filter for cells of type 0
    filter_input.value = "(props['type'] == 0) | (props['type'] == 1)"
    im = get_tile("overlay", 4, 8, 4, show=False)
    _, num_after = label(np.any(im[:, :, :3], axis=2))
    # should be less than without the filter
    assert num_after < num_before

    # check type toggles in combo with filter
    type_column_list = doc.get_model_by_name("type_column0").children
    type_column_list[0].active = False
    im = get_tile("overlay", 4, 8, 4, show=False)
    _, num_final = label(np.any(im[:, :, :3], axis=2))
    # should be even less
    assert num_final < num_after

    # set no filter
    filter_input.value = ""
    type_column_list[0].active = True
    im = get_tile("overlay", 4, 8, 4, show=False)
    _, num_after = label(np.any(im[:, :, :3], axis=2))
    # should be back to original number
    assert num_after == num_before
    # set an impossible filter
    filter_input.value = "props['prob'] < 0"
    im = get_tile("overlay", 4, 8, 4, show=False)
    _, num_after = label(np.any(im[:, :, :3], axis=2))
    # should be no cells
    assert num_after == 0


def test_scale_spinner(doc: Document) -> None:
    """Test setting scale for rendering small annotations."""
    scale_spinner = doc.get_model_by_name("scale0")
    # set the scale to 0.5
    scale_spinner.value = 8
    # check that the scale has been set correctly
    assert get_renderer_prop("max_scale") == 8


def test_blur_spinner(doc: Document) -> None:
    """Test setting blur for annotation layer."""
    blur_spinner = doc.get_model_by_name("blur0")
    # set the blur to 4
    blur_spinner.value = 4
    # check that the blur has been set correctly
    assert get_renderer_prop("blur_radius") == 4


def test_res_switch(doc: Document) -> None:
    """Test resolution switch."""
    res_switch = doc.get_model_by_name("res0")
    # set the resolution to 0
    res_switch.active = 0
    # check that the resolution has been set correctly
    assert main.UI["vstate"].res == 1
    res_switch.active = 1
    assert main.UI["vstate"].res == 2


def test_color_cycler() -> None:
    """Test the color cycler."""
    cycler = main.ColorCycler()
    colors = cycler.colors
    assert cycler.get_next() == colors[0]
    assert cycler.get_next() == colors[1]

    rand_color = cycler.get_random()
    assert rand_color in colors

    new_color = cycler.generate_random()
    # should be a valid hex color
    assert re.match(r"^#[0-9a-fA-F]{6}$", new_color)

    # test instantiate with custom colors
    custom_cycler = main.ColorCycler(["#ff0000", "#00ff00"])
    assert len(custom_cycler.colors) == 2
    assert custom_cycler.get_next() == "#ff0000"


def test_cmap_select(doc: Document) -> None:
    """Test changing the cmap."""
    cmap_select = doc.get_model_by_name("cmap0")

    main.UI["cprop_input"].value = ["prob"]
    # set to jet
    cmap_select.value = "jet"
    resp = main.UI["s"].get(f"http://{main.host2}:{main.port}/tileserver/cmap")
    assert resp.json() == "jet"
    # set to dict
    cmap_select.value = "dict"
    resp = main.UI["s"].get(f"http://{main.host2}:{main.port}/tileserver/cmap")
    assert isinstance(resp.json(), dict)

    main.UI["cprop_input"].value = ["type"]
    # should now be the type mapping
    resp = main.UI["s"].get(f"http://{main.host2}:{main.port}/tileserver/cmap")
    for key in main.UI["vstate"].mapper:
        assert str(key) in resp.json()
        assert np.all(
            np.array(resp.json()[str(key)]) == np.array(main.UI["vstate"].mapper[key]),
        )
    # set the cmap to "coolwarm"
    cmap_select.value = "coolwarm"
    resp = main.UI["s"].get(f"http://{main.host2}:{main.port}/tileserver/cmap")
    # as cprop is type (categorical), it should have had no effect
    for key in main.UI["vstate"].mapper:
        assert str(key) in resp.json()
        assert np.all(
            np.array(resp.json()[str(key)]) == np.array(main.UI["vstate"].mapper[key]),
        )

    main.UI["cprop_input"].value = ["prob"]
    resp = main.UI["s"].get(f"http://{main.host2}:{main.port}/tileserver/cmap")
    # should be coolwarm as that is the last cmap we set, and prob is continuous
    assert resp.json() == "coolwarm"


def test_option_buttons() -> None:
    """Test the option buttons."""
    # default will be [FILLED]
    # test outline only
    assert get_renderer_prop("thickness") == -1
    main.opt_buttons_cb(None, None, [])
    assert get_renderer_prop("thickness") == 1
    # test micron formatter
    assert isinstance(main.UI["p"].xaxis[0].formatter, bkmodels.BasicTickFormatter)
    main.opt_buttons_cb(None, None, [MICRON_FORMATTER])
    assert isinstance(main.UI["p"].xaxis[0].formatter, bkmodels.CustomJSTickFormatter)
    # test gridlines
    assert main.UI["p"].xgrid.grid_line_alpha == 0
    main.opt_buttons_cb(None, None, [GRIDLINES, MICRON_FORMATTER])
    assert main.UI["p"].xgrid.grid_line_alpha == 0.6
    # test removing above options
    main.opt_buttons_cb(None, None, [FILLED])
    assert main.UI["p"].xgrid.grid_line_alpha == 0
    assert isinstance(main.UI["p"].xaxis[0].formatter, bkmodels.BasicTickFormatter)
    assert get_renderer_prop("thickness") == -1


def test_populate_slide_list(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test populating the slide list."""
    slide_select = doc.get_model_by_name("slide_select0")
    assert len(slide_select.options) == 4
    main.populate_slide_list(
        data_path["base_path"] / "slides",
        search_txt="TCGA-HE-7130-01Z-00-DX1",
    )
    assert len(slide_select.options) == 1
    main.populate_slide_list(
        data_path["base_path"] / "slides",
    )
    assert len(slide_select.options) == 4


def test_clear_overlays(doc: Document, data_path: pytest.TempPathFactory) -> None:
    """Test clearing overlays."""
    slide_select = doc.get_model_by_name("slide_select0")
    slide_select.value = [data_path["slide1"].name]

    # load an annotation store
    layer_drop = doc.get_model_by_name("layer_drop0")
    click = MenuItemClick(layer_drop, str(data_path["annotations"]))
    layer_drop._trigger_event(click)
    assert "overlay" in main.UI["vstate"].layer_dict

    # now clear the overlays
    clear_button = doc.get_model_by_name("clear_button0")
    click = ButtonClick(clear_button)
    clear_button._trigger_event(click)
    assert "overlay" not in main.UI["vstate"].layer_dict
    assert (
        len(main.UI["vstate"].layer_dict) == 5
    )  # slide & empty box/pt/edge/node renderers

    # click again - should do nothing and not error
    click = ButtonClick(clear_button)
    clear_button._trigger_event(click)
    assert "overlay" not in main.UI["vstate"].layer_dict
    assert len(main.UI["vstate"].layer_dict) == 5


def test_channel_color_ui_callbacks(
    doc: Document,
    data_path: pytest.TempPathFactory,
) -> None:
    """Test channel color selection and apply changes callbacks on qptiff."""
    slide_select = doc.get_model_by_name("slide_select0")
    slide_select.value = [data_path["qptiff"].name]
    assert main.UI["vstate"].slide_path == data_path["qptiff"]

    channel_table, color_table, color_picker, apply_button, _ = (
        get_channel_ui_elements()
    )
    # check we see 5 channels
    assert len(channel_table.source.data["channels"]) == 5

    # if no channels selected, check apply button does nothing
    old_colors = color_table.source.data["colors"].copy()
    color_picker.color = "#ffff00"
    click = ButtonClick(apply_button)
    apply_button._trigger_event(click)
    assert color_table.source.data["colors"] == old_colors

    # select the first channel and set it to red
    channel_index = 0
    color_table.source.selected.indices = [channel_index]
    color_picker.color = "#ff0000"
    channel_table.source.selected.indices = [channel_index]
    click = ButtonClick(apply_button)
    apply_button._trigger_event(click)
    assert color_table.source.data["colors"] != old_colors

    # check that getting a tile now red
    tile = get_tile("slide", 0, 0, 0, show=False).astype(np.float32)
    sum_r = tile[:, :, 0].sum()
    sum_gb = tile[:, :, 1:].sum()
    assert sum_r > 0
    # may be tiny non-zero g and b values due to webp compression
    # but should be almost pure red
    assert (sum_gb) / (sum_r + 1e-5) < 0.1


def test_enhance_slider_callback(
    doc: Document,
    data_path: pytest.TempPathFactory,
) -> None:
    """Test enhance slider callback on qptiff."""
    slide_select = doc.get_model_by_name("slide_select0")
    slide_select.value = [data_path["qptiff"].name]
    assert main.UI["vstate"].slide_path == data_path["qptiff"]

    channel_table, color_table, color_picker, apply_button, enhance_slider = (
        get_channel_ui_elements()
    )
    assert len(channel_table.source.data["channels"]) > 0

    channel_index = 0
    color_table.source.selected.indices = [channel_index]
    color_picker.color = "#ff0000"
    channel_table.source.selected.indices = [channel_index]
    click = ButtonClick(apply_button)
    apply_button._trigger_event(click)

    before = get_tile("slide", 0, 0, 0, show=False).astype(np.float32)
    enhance_slider.value = 2.0
    after = get_tile("slide", 0, 0, 0, show=False).astype(np.float32)
    # enhance should have made it brighter
    diff = after - before
    assert np.max(diff) > 0


def test_clearing_doc(doc: Document) -> None:
    """Test that the doc can be cleared."""
    doc.clear()
    assert len(doc.roots) == 0


def test_app_hooks_session_destroyed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hook should call reset endpoint and exit."""
    recorded: dict[str, object] = {}

    def fake_get(url: str, *, timeout: int) -> None:
        """Fake requests.get to record parameters."""
        recorded["url"] = url
        recorded["timeout"] = timeout

    monkeypatch.setattr(app_hooks, "PORT", "6150")
    monkeypatch.setattr(app_hooks.requests, "get", fake_get)
    exited = False

    def fake_exit() -> None:
        """Fake sys.exit to record call."""
        nonlocal exited
        exited = True

    monkeypatch.setattr(app_hooks, "sys", SimpleNamespace(exit=fake_exit))
    app_hooks.on_session_destroyed(_DummySessionContext("user-1"))
    assert recorded["url"] == "http://127.0.0.1:6150/tileserver/reset/user-1"
    assert recorded["timeout"] == 5
    assert exited


def test_app_hooks_session_destroyed_suppresses_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ReadTimeout should be suppressed and exit still called."""

    def fake_get(*_: object, **__: object) -> None:
        """Fake requests.get to raise ReadTimeout."""
        raise app_hooks.requests.exceptions.ReadTimeout  # type: ignore[attr-defined]

    monkeypatch.setattr(app_hooks, "PORT", "6160")
    monkeypatch.setattr(app_hooks.requests, "get", fake_get)
    exited = False

    def fake_exit() -> None:
        """Fake sys.exit to record call."""
        nonlocal exited
        exited = True

    monkeypatch.setattr(app_hooks, "sys", SimpleNamespace(exit=fake_exit))
    app_hooks.on_session_destroyed(_DummySessionContext("user-2"))
    assert exited


def test_dummyattr_stores_value() -> None:
    """Ensure that DummyAttr correctly stores the provided value.

    This test verifies that the constructor assigns the input value
    to the `item` attribute.
    """
    obj = main.DummyAttr("hello")
    assert obj.item == "hello"


@pytest.mark.parametrize(
    "value",
    [
        123,
        3.14,
        {"a": 1},
        [1, 2, 3],
        (1, 2),
        None,
    ],
)
def test_dummyattr_accepts_any_type(value: object) -> None:
    """Confirm that DummyAttr accepts and stores values of any type."""
    obj = main.DummyAttr(value)
    assert obj.item is value


class DummyResponse:
    """A dummy HTTP response object containing invalid JSON."""

    text: str = "not valid json"


def test_get_channel_info_logs_json_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that get_channel_info logs a warning."""

    class DummySession:
        """A dummy session whose GET request returns invalid JSON."""

        def get(self, url: str) -> DummyResponse:  # skipcq: PYL-R0201  # noqa: ARG002
            """Dummy get function."""
            return DummyResponse()

    # Patch __getitem__ on the UIWrapper *class* so UI["s"] returns DummySession()
    monkeypatch.setattr(
        main.UI.__class__,
        "__getitem__",
        lambda _self, key: DummySession() if key == "s" else None,
    )

    with caplog.at_level("WARNING"):
        result = main.get_channel_info()

    assert result == ({}, [])

    assert any("Error decoding JSON" in message for message in caplog.messages)


# ---------------------------------------------------------------------------
# Helper stubs (purely for testing)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper stubs (purely for testing) — ruff/DeepSource compliant
# ---------------------------------------------------------------------------


class FakeInfo:
    """Lightweight stub for WSIReader.info."""

    def __init__(self, path: Path) -> None:
        """Initialize fake slide info with plausible values."""
        self.mpp = [0.25, 0.25]
        self.slide_dimensions = (2000, 1500)
        self.file_path = path

    def as_dict(self) -> dict[str, object]:
        """Return dictionary representation of slide metadata."""
        return {
            "file_path": self.file_path,
            "mpp": self.mpp,
            "dims": self.slide_dimensions,
        }


class FakeWSIReader:
    """Stub mimicking tiatoolbox WSIReader."""

    def __init__(self, path: Path) -> None:
        """Create a fake reader with associated FakeInfo."""
        self.info = FakeInfo(path)

    @staticmethod
    def open(path: Path | str) -> FakeWSIReader:
        """Replacement for WSIReader.open that returns a controlled stub."""
        return FakeWSIReader(Path(path))

    @staticmethod
    def slide_thumbnail() -> np.ndarray:
        """Return a small fake thumbnail array."""
        return 255 * np.ones((20, 20, 3), dtype="uint8")

    @staticmethod
    def read_bounds(*_args: object, **_kwargs: object) -> np.ndarray:
        """Return a fake array for ROI extraction."""
        return 255 * np.ones((10, 10, 3), dtype="uint8")


class FakeZoomifyGenerator:
    """Stub mimicking ZoomifyGenerator with a fixed zoom level count."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Initialize with a default level count."""
        self.level_count = 7


class FakeResp:
    """Lightweight fake requests.Response used for stubbing."""

    class CookieDict(dict):
        """Subclass dict to mimic requests cookie access."""

        def get(self, key: str, default: object | None = None) -> object:
            """Return cookie value or default."""
            return super().get(key, default)

    def __init__(self, text: str = "", cookies: dict[str, str] | None = None) -> None:
        """Initialize fake response with text and optional cookies."""
        self.text = text
        self._cookies = FakeResp.CookieDict(cookies or {})

    @property
    def cookies(self) -> dict[str, str]:
        """Return fake cookie jar."""
        return self._cookies


class FakeSession:
    """Minimal stub for requests.Session."""

    def __init__(self: FakeSession) -> None:
        """Initialize fake session with no proxies and empty mounts."""
        self.trust_env = False
        self.proxies: dict[str, str | None] = {}
        self.mounted: dict[str, object] = {}

    def mount(self: FakeSession, scheme: str, adapter: object) -> None:
        """Record mounted adapters (no real effect)."""
        self.mounted[scheme] = adapter

    @staticmethod
    def get(url: str, *_args: object, **_kwargs: object) -> FakeResp:
        """Return canned JSON or session cookie."""
        if url.endswith("/tileserver/session_id"):
            return FakeResp(cookies={"session_id": "test_user"})
        return FakeResp(text="{}")

    @staticmethod
    def put(
        url: str,
        _data: dict[str, object] | None = None,
        *_args: object,
        **_kwargs: object,
    ) -> FakeResp:
        """Return controlled values based on the endpoint."""
        if url.endswith("/tileserver/overlay"):
            return FakeResp(text='"slide"')
        return FakeResp(text='"ok"')

    @staticmethod
    def post(
        _url: str,
        _data: dict[str, object] | None = None,
        *_args: object,
        **_kwargs: object,
    ) -> FakeResp:
        """Return generic OK response."""
        return FakeResp(text='"ok"')


class FakeRequest:
    """Stub for request object inside session context."""

    def __init__(self, arguments: dict[str, list[bytes]] | None = None) -> None:
        """Initialize with an optional arguments dict."""
        self.arguments = arguments or {}


class FakeSessionContext:
    """Stub for Bokeh session context."""

    def __init__(self, arguments: dict[str, list[bytes]] | None = None) -> None:
        """Create a session context holding a FakeRequest."""
        self.request = FakeRequest(arguments=arguments)


class FakeDoc:
    """Stub for Bokeh Document used during import/setup."""

    def __init__(
        self,
        *,
        with_session: bool = False,
        arguments: dict[str, list[bytes]] | None = None,
    ) -> None:
        """Initialize a fake Bokeh document."""
        self._sc = FakeSessionContext(arguments) if with_session else None
        self.template_variables: dict[str, object] = {}
        self._callbacks: list[tuple[object, int]] = []
        self._roots: list[object] = []
        self.title: str = ""

    @property
    def session_context(self) -> FakeSessionContext | None:
        """Return fake session context if enabled."""
        return self._sc

    def add_periodic_callback(self, fn: object, ms: int) -> None:
        """Record callbacks for later introspection."""
        self._callbacks.append((fn, ms))

    def add_root(self, root: object) -> None:
        """Record Bokeh layout roots."""
        self._roots.append(root)


# ---------------------------------------------------------------------------
# Controlled import of main.py under patching
# ---------------------------------------------------------------------------


def reload_main(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_session: bool = False,
    req_args: dict[str, list[bytes]] | None = None,
) -> object:
    """Reload main.py in a controlled environment.

    This ensures `curdoc()`, `requests.Session`, WSIReader, and ZoomifyGenerator
    are fully stubbed before import.

    """
    # IMPORTANT: patch symbols main.py will import/use
    monkeypatch.setattr(
        "bokeh.io.curdoc",
        lambda: FakeDoc(with_session=with_session, arguments=req_args),
        raising=False,
    )
    monkeypatch.setattr("requests.Session", FakeSession, raising=True)
    monkeypatch.setattr(
        "tiatoolbox.wsicore.wsireader.WSIReader.open",
        FakeWSIReader.open,
        raising=True,
    )
    monkeypatch.setattr(
        "tiatoolbox.tools.pyramid.ZoomifyGenerator",
        FakeZoomifyGenerator,
        raising=True,
    )

    return importlib.reload(main)


# ---------------------------------------------------------------------------
# Actual tests
# ---------------------------------------------------------------------------


def test_populate_table_active_channels_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test populate_table() branch where active_channels == [].

    Ensures:
    - Table rows populate from colors.
    - No indices selected.

    """
    main = reload_main(monkeypatch, with_session=False)

    channel_ui = main.create_channel_color_ui()
    main.win_dicts.clear()
    main.win_dicts.append({"channel_select": channel_ui})
    main.UI.active = 0

    monkeypatch.setattr(
        main,
        "get_channel_info",
        lambda: ({"C1": (1, 2, 3), "C2": (4, 5, 6)}, []),
        raising=True,
    )

    main.populate_table()
    tables = main.UI["channel_select"].children[1].children[0].children

    assert tables[0].source.data["channels"] == ["C1", "C2"]
    assert len(tables[1].source.data["colors"]) == 2
    assert tables[0].source.selected.indices == []


def test_initialise_slide_initial_view_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test initialise_slide() branch where slide_name appears in initial_views.

    Ensures p.x_range/y_range are set from config.

    """
    main = reload_main(monkeypatch, with_session=False)

    slide_path = Path("/tmp/s1.svs")  # noqa: S108
    vstate = main.ViewerState(slide_path)

    p = figure(width=400, height=400)

    main.win_dicts.clear()
    main.win_dicts.append({"vstate": vstate, "p": p})
    main.UI.active = 0

    main.doc_config.config["initial_views"] = {"s1": [10, 20, 110, 120]}
    main.initialise_slide()

    assert p.x_range.start == 10
    assert p.x_range.end == 110
    assert p.y_range.start == -120
    assert p.y_range.end == -20


def test_slide_select_cb_auto_load_triggers_layer_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test slide_select_cb() when auto_load=True.

    Ensures:
    - slide_select_cb() runs without error
    - auto_load triggers layer_drop_cb() exactly once
    """
    main = reload_main(monkeypatch, with_session=False)

    class MiniHover:
        """Initialize MiniHover."""

        tooltips: object | None = None

    p = figure(width=400, height=300)

    # renderer[0] must be a TileRenderer for change_tiles()
    dummy_ts = WMTSTileSource(url="http://dummy/{Z}/{X}/{Y}.png")
    p.add_tile(dummy_ts)

    # Fill remaining permanent renderers
    for _ in range(main.N_PERMANENT_RENDERERS - 1):
        p.circle([], [])

    class FakeTable:
        """Create a minimal fake channel_select matching expected structure."""

        def __init__(self: FakeTable) -> None:
            self.source = main.ColumnDataSource({"channels": [], "dummy": []})
            self.selected = types.SimpleNamespace(indices=[])

    class FakeChannelSelect:
        """Must satisfy: UI["channel_select"].children[1].children[0].children."""

        def __init__(self: FakeChannelSelect) -> None:
            table1 = FakeTable()  # channels table
            table2 = FakeTable()  # colors table

            inner = types.SimpleNamespace(children=[table1, table2])
            outer = types.SimpleNamespace(children=[inner])

            self.children = [None, outer]

    class DummyCol:
        """Initialize DummyCol test."""

        def __init__(self: DummyCol) -> None:
            self.children: list[object] = []

    win = {
        "p": p,
        "vstate": main.ViewerState(Path("/tmp/initial.svs")),  # noqa: S108
        "pt_source": main.ColumnDataSource({"x": [], "y": []}),
        "box_source": main.ColumnDataSource(
            {"x": [], "y": [], "width": [], "height": []}
        ),
        "node_source": main.ColumnDataSource({"x_": [], "y_": [], "node_color_": []}),
        "edge_source": main.ColumnDataSource(
            {"x0_": [], "y0_": [], "x1_": [], "y1_": []}
        ),
        "hover": MiniHover(),
        "layer_drop": main.Dropdown(
            label="Add Overlay", menu=[("foo.json", "foo.json")]
        ),
        "s": FakeSession(),
        # Required UI structures
        "color_column": DummyCol(),
        "type_column": DummyCol(),
        "channel_select": FakeChannelSelect(),
        # Required by change_tiles()
        "user": "test_user",
    }

    main.win_dicts.clear()
    main.win_dicts.append(win)
    main.UI.active = 0

    # Enable auto-load
    main.doc_config.config["auto_load"] = True

    # Avoid filesystem operations
    monkeypatch.setattr(
        main,
        "populate_layer_list",
        lambda *_args, **_kwargs: None,
        raising=True,
    )

    # Spy on layer_drop_cb
    called: list[str] = []

    def spy(attr: object) -> None:
        """Spy attribute for layer_drop_cb."""
        called.append(attr.item)

    monkeypatch.setattr(main, "layer_drop_cb", spy, raising=True)

    # ACT
    new_slide_name = [Path("/tmp/newslide.svs").name]  # noqa: S108
    main.slide_select_cb(None, None, new=new_slide_name)

    # ASSERT
    assert called == ["foo.json"]


def test_layer_drop_cb_resp_equals_slide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test branch in layer_drop_cb() where resp == "slide".

    Expected behavior:
    - add_layer() is NOT called
    - change_tiles("slide") IS called (this is correct behaviour)

    """
    main = reload_main(monkeypatch, with_session=False)

    # Prepare minimal window with a FakeSession
    main.win_dicts.clear()
    main.win_dicts.append({"s": FakeSession()})
    main.UI.active = 0

    # Spy: add_layer() must not be called
    monkeypatch.setattr(
        main,
        "add_layer",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("add_layer should NOT be called")
        ),
        raising=True,
    )

    # Spy: change_tiles() SHOULD be called exactly once
    change_calls: list[str] = []

    def change_spy(arg: str) -> None:
        """Change spy."""
        change_calls.append(arg)

    monkeypatch.setattr(main, "change_tiles", change_spy, raising=True)

    # Trigger overlay load on a non-annotation file => uses resp == "slide"
    attr = main.DummyAttr("/tmp/overlay.png")  # noqa: S108
    main.layer_drop_cb(attr)

    # Assert correct behavior
    assert change_calls == ["slide"], (
        "change_tiles('slide') must be called when resp == 'slide'"
    )


def test_gather_ui_elements_ui_elements_1_and_2_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test gather_ui_elements() when config includes ui_elements_1 and ui_elements_2.

    Ensures only enabled elements are included.
    """
    main = reload_main(monkeypatch, with_session=False)

    main.doc_config.config["ui_elements_1"] = {
        "slide_select": 1,
        "layer_drop": 1,
        "slide_row": 1,
        "overlay_row": 1,
        "filter_input": 0,
        "cprop_input": 1,
        "cmap_row": 1,
        "type_cmap_select": 1,
        "model_row": 1,
        "clear_button": 1,
        "type_select_row": 1,
    }

    main.doc_config.config["ui_elements_2"] = {
        "opt_buttons": 1,
        "pt_size_spinner": 1,
        "edge_size_spinner": 0,
        "res_switch": 1,
        "channel_select": 1,
    }

    vstate = main.ViewerState(Path("/tmp/v.svs"))  # noqa: S108
    ui_layout, extra_options, _elements = main.gather_ui_elements(vstate, win_num=0)

    present = {c.name for c in ui_layout.children if hasattr(c, "name")}
    assert "filter0" not in present
    assert "slide_select0" in present
    assert "cprop0" in present

    assert len(extra_options.children) == 4


def test_make_window_sets_user_when_session_context_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test branch where make_window() sees session_context present.

    Ensures:
        - 'user' cookie gets written to session_context.request.arguments

    """
    # Import main WITHOUT a session context so module-level do_doc is False
    main = reload_main(monkeypatch, with_session=False)

    # Prepare a FakeDoc WITH a session_context
    # this is the one we want make_window to see
    req_args: dict[str, list[bytes]] = {}
    fake_doc = FakeDoc(with_session=True, arguments=req_args)

    # CRITICAL FIX: patch main.curdoc, not bokeh.io.curdoc
    monkeypatch.setattr(main, "curdoc", lambda: fake_doc, raising=False)

    # Create a vstate and call make_window
    vstate = main.ViewerState(Path("/tmp/one.svs"))  # noqa: S108
    main.make_window(vstate)

    # Assertions: make_window must write "user"
    sc = main.curdoc().session_context
    assert sc is not None
    assert "user" in sc.request.arguments
    assert sc.request.arguments["user"] == "test_user"


def test_make_window_hover_nodes_edges_colorbar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test make_window() branches.

    - opts.hover_on == 1 (skip disabling inspect tool)
    - nodes_on == False
    - edges_on == False
    - colorbar_on == 0

    """
    main = reload_main(monkeypatch, with_session=False)

    main.doc_config.config["opts"] = {
        "hover_on": 1,
        "nodes_on": False,
        "edges_on": False,
        "colorbar_on": 0,
    }

    vstate = main.ViewerState(Path("/tmp/abc.svs"))  # noqa: S108
    win = main.make_window(vstate)
    p = win["p"]

    assert p.toolbar.active_inspect is not None

    node_renderer = p.renderers[-2]
    assert node_renderer.glyph.fill_alpha == 0
    assert node_renderer.glyph.line_alpha == 0

    edge_renderer = p.renderers[-1]
    assert edge_renderer.visible is False

    assert not p.select(type=ColorBar)
