import io
import time

import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import pytest
import requests
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.events import ButtonClick, MenuItemClick
from matplotlib import colormaps
from PIL import Image
from scipy.ndimage import label

from tiatoolbox.data import _fetch_remote_sample
from tiatoolbox.visualization.bokeh_app import main

BOKEH_PATH = pkg_resources.resource_filename("tiatoolbox", "visualization/bokeh_app")


def get_tile(layer, x, y, z, show=False):
    """Get a tile from the server."""
    source = main.UI["p"].renderers[main.UI["vstate"].layer_dict[layer]].tile_source
    url = source.url
    # replace {x}, {y}, {z} with tile coordinates
    url = url.replace("{x}", str(x)).replace("{y}", str(y)).replace("{z}", str(z))
    im = io.BytesIO(requests.get(url).content)
    if show:
        plt.imshow(np.array(Image.open(im)))
        plt.show()
    return np.array(Image.open(im))


def get_renderer_prop(prop):
    """Get a renderer property from the server."""
    resp = main.UI["s"].get(f"http://{main.host2}:5000/tileserver/renderer/{prop}")
    return resp.json()


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    """Set up a temporary data directory."""
    tmp_path = tmp_path_factory.mktemp("data")
    (tmp_path / "slides").mkdir()
    (tmp_path / "overlays").mkdir()
    return {"base_path": tmp_path}


@pytest.fixture(scope="module", autouse=True)
def annotation_path(data_path):
    data_path["slide1"] = _fetch_remote_sample(
        "svs-1-small",
        data_path["base_path"] / "slides",
    )
    data_path["slide2"] = _fetch_remote_sample(
        "ndpi-1",
        data_path["base_path"] / "slides",
    )
    data_path["annotations"] = _fetch_remote_sample(
        "annotation_store_svs_1",
        data_path["base_path"] / "overlays",
    )
    data_path["graph"] = _fetch_remote_sample(
        "graph_svs_1",
        data_path["base_path"] / "overlays",
    )
    return data_path


"""Test visualization tool."""


@pytest.fixture(scope="module")
def doc(data_path):
    # make a bokeh app
    main.config.set_sys_args(argv=["dummy_str", str(data_path["base_path"])])
    handler = FunctionHandler(main.config.setup_doc)
    app = Application(handler)
    return app.create_document()


def test_roots(doc):
    # should be 2 roots, main window and controls
    assert len(doc.roots) == 2


def test_slide_select(doc, data_path):
    slide_select = doc.get_model_by_name("slide_select0")
    # check there are two available slides
    assert len(slide_select.options) == 2
    assert slide_select.options[0][0] == data_path["slide1"].name

    # select a slide and check it is loaded
    slide_select.value = ["CMU-1.ndpi"]
    assert main.UI["vstate"].slide_path == data_path["slide2"]


def test_dual_window(doc, data_path):
    control_tabs = doc.get_model_by_name("ui_layout")
    doc.get_model_by_name("slide_windows")
    control_tabs.active = 1
    slide_select = doc.get_model_by_name("slide_select1")
    assert len(slide_select.options) == 2
    assert slide_select.options[0][0] == data_path["slide1"].name


def test_remove_dual_window(doc, data_path):
    control_tabs = doc.get_model_by_name("ui_layout")
    slide_wins = doc.get_model_by_name("slide_windows")
    assert len(slide_wins.children) == 2
    # remove the second window
    control_tabs.tabs.pop()
    assert len(slide_wins.children) == 1

    slide_select = doc.get_model_by_name("slide_select0")
    slide_select.value = [data_path["slide1"].name]
    assert main.UI["vstate"].slide_path == data_path["slide1"]


def test_add_annotation_layer(doc, data_path):
    layer_drop = doc.get_model_by_name("layer_drop0")
    assert len(layer_drop.menu) == 2
    n_renderers = len(doc.get_model_by_name("slide_windows").children[0].renderers)
    # trigger an event to select the annotation .db file
    click = MenuItemClick(layer_drop, layer_drop.menu[0][0])
    layer_drop._trigger_event(click)
    # should be one more renderer now
    assert len(doc.get_model_by_name("slide_windows").children[0].renderers) == (
        n_renderers + 1
    )
    # we should have got the types of annotations back from the server too
    assert main.UI["vstate"].types == ["0", "1", "2", "3", "4"]


def test_cprop_input(doc):
    cprop_input = doc.get_model_by_name("cprop0")
    cmap_select = doc.get_model_by_name("cmap0")
    cprop_input.value = ["prob"]
    # as prob is continuous, cmap should be set to whatever cmap is selected
    assert main.UI["vstate"].cprop == "prob"
    assert main.UI["color_bar"].color_mapper.palette[0] == main.rgb2hex(
        colormaps[cmap_select.value](0),
    )

    cprop_input.value = ["type"]
    # as type is discrete, cmap should be a dict mapping types to colors
    assert isinstance(main.UI["vstate"].mapper, dict)
    assert list(main.UI["vstate"].mapper.keys()) == list(
        main.UI["vstate"].orig_types.values(),
    )


def test_type_cmap_select(doc):
    cmap_select = doc.get_model_by_name("type_cmap0")
    main.UI["vstate"]
    main.UI["user"]
    cmap_select.value = ["prob", "0"]
    # set edge thicknes to 0 so the edges don't add an extra colour
    spinner = doc.get_model_by_name("edge_size0")
    spinner.value = 0
    im = get_tile("overlay", 1, 2, 2)
    # check there are more than just num_types unique colors in the image,
    # as we have mapped type 0 to a continuous cmap on prob
    assert len(np.unique(im.sum(axis=2))) > 10


def test_load_graph(doc, data_path):
    layer_drop = doc.get_model_by_name("layer_drop0")
    # trigger an event to select the graph .db file
    click = MenuItemClick(layer_drop, layer_drop.menu[1][1])
    layer_drop._trigger_event(click)
    # we should have 2144 nodes in the node_source now
    assert len(main.UI["node_source"].data["x_"]) == 2144


def test_hovernet_on_box(doc, data_path):
    slide_select = doc.get_model_by_name("slide_select0")
    slide_select.value = [data_path["slide2"].name]
    go_button = doc.get_model_by_name("to_model0")
    assert len(main.UI["color_column"].children) == 0
    slide_select.value = [data_path["slide1"].name]
    # set up a box selection
    main.UI["box_source"].data = {
        "x": [1200],
        "y": [-2000],
        "width": [800],
        "height": [800],
    }

    # select hovernet model and run it on box
    model_select = doc.get_model_by_name("model_drop0")
    click = MenuItemClick(model_select, model_select.menu[0])
    model_select._trigger_event(click)

    click = ButtonClick(go_button)
    go_button._trigger_event(click)
    im = get_tile("overlay", 4, 8, 4)
    lab, num = label(np.any(im[:, :, :3], axis=2))
    # check there are multiple cells being detected
    assert len(main.UI["color_column"].children) > 3
    assert num > 10


def test_alpha_sliders(doc):
    slide_alpha = doc.get_model_by_name("slide_alpha0")
    overlay_alpha = doc.get_model_by_name("overlay_alpha0")

    # set alpha to 0.5
    slide_alpha.value = 0.5
    overlay_alpha.value = 0.5
    # check that the alpha values have been set correctly
    assert main.UI["p"].renderers[0].alpha == 0.5
    assert main.UI["p"].renderers[main.UI["vstate"].layer_dict["overlay"]].alpha == 0.5


def test_alpha_buttons(doc):
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


def test_type_select(doc):
    # load annotation layer
    layer_drop = doc.get_model_by_name("layer_drop0")
    click = MenuItemClick(layer_drop, layer_drop.menu[0][0])
    layer_drop._trigger_event(click)
    time.sleep(1)
    im = get_tile("overlay", 4, 8, 4)
    _, num_before = label(np.any(im[:, :, :3], axis=2))
    type_column_list = doc.get_model_by_name("type_column0").children
    # click on the first and last to deselect them
    type_column_list[0].active = False
    type_column_list[-1].active = False
    # check that the number of cells has decreased
    im = get_tile("overlay", 4, 8, 4)
    _, num_after = label(np.any(im[:, :, :3], axis=2))
    assert num_after < num_before
    # reselect them
    type_column_list[0].active = True
    type_column_list[-1].active = True
    # check we are back to original number of cells
    im = get_tile("overlay", 4, 8, 4)
    _, num_after = label(np.any(im[:, :, :3], axis=2))
    assert num_after == num_before


def test_color_boxes(doc):
    color_column_list = doc.get_model_by_name("color_column0").children
    # set type 0 to red
    color_column_list[0].color = "#ff0000"
    # set type 1 to blue
    color_column_list[1].color = "#0000ff"
    # check the mapper matches the new colors
    assert main.UI["vstate"].mapper[0] == (1, 0, 0, 1)
    assert main.UI["vstate"].mapper[1] == (0, 0, 1, 1)


def test_node_and_edge_alpha(doc, data_path):
    layer_drop = doc.get_model_by_name("layer_drop0")
    # trigger an event to select the graph .db file
    click = MenuItemClick(layer_drop, layer_drop.menu[1][1])
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


def test_hover_tool(doc):
    # how to test this?
    pass


def test_filter_box(doc):
    filter_input = doc.get_model_by_name("filter0")
    im = get_tile("overlay", 4, 8, 4)
    _, num_before = label(np.any(im[:, :, :3], axis=2))
    # filter for cells of type 0
    filter_input.value = "props['type'] == 0"
    im = get_tile("overlay", 4, 8, 4)
    _, num_after = label(np.any(im[:, :, :3], axis=2))
    # should be less than without the filter
    assert num_after < num_before
    # set no filter
    filter_input.value = ""
    im = get_tile("overlay", 4, 8, 4)
    _, num_after = label(np.any(im[:, :, :3], axis=2))
    # should be back to original number
    assert num_after == num_before
    # set an impossible filter
    filter_input.value = "props['prob'] < 0"
    im = get_tile("overlay", 4, 8, 4)
    _, num_after = label(np.any(im[:, :, :3], axis=2))
    # should be no cells
    assert num_after == 0


def test_scale_spinner(doc):
    scale_spinner = doc.get_model_by_name("scale0")
    # set the scale to 0.5
    scale_spinner.value = 0.5
    # check that the scale has been set correctly
    assert get_renderer_prop("max_scale") == 0.5


def test_blur_spinner(doc):
    blur_spinner = doc.get_model_by_name("blur0")
    # set the blur to 4
    blur_spinner.value = 4
    # check that the blur has been set correctly
    assert get_renderer_prop("blur_radius") == 4


def test_session_end(doc):
    doc.clear()
