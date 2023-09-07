"""Test the bokeh app with a config.json."""
import time
from threading import Thread

import pkg_resources
import pytest
from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler
from bokeh.client.session import pull_session
from bokeh.events import MenuItemClick
from bokeh.server.server import Server

from tiatoolbox.data import _fetch_remote_sample

BOKEH_PATH = pkg_resources.resource_filename("tiatoolbox", "visualization/bokeh_app")


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    """Set up a temporary data directory."""
    tmp_path = tmp_path_factory.mktemp("data")
    (tmp_path / "slides").mkdir()
    (tmp_path / "overlays").mkdir()
    return {"base_path": tmp_path}


@pytest.fixture(scope="module", autouse=True)
def annotation_path(data_path):
    """Set up a dictionary defining the paths to the annotation files."""
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
    data_path["config"] = _fetch_remote_sample(
        "config_2",
        data_path["base_path"] / "overlays",
    )
    return data_path


@pytest.fixture(scope="module")
def bk_session(data_path):
    def run_app() -> None:
        handler = DirectoryHandler(
            filename=BOKEH_PATH,
            argv=[str(data_path["base_path"])],
        )
        app = Application(handler)
        server = Server({"/bkapp": app})
        server.start()
        server.io_loop.start()

    proc = Thread(target=run_app, daemon=True)
    proc.start()

    return pull_session(
        url="http://localhost:5006/bkapp",
        arguments={"slide": "CMU-1.ndpi"},
    )  # , arguments={"demo": "TTB_vis_folder"})# as session:
    # customize session here


def test_slide_in_req_args(bk_session):
    """Test that the slide is in the required arguments."""
    # slide should be in the required arguments
    # assert "slide" in bk_session.document.session_context.request.arguments

    slide_select = bk_session.document.get_model_by_name("slide_select0")
    # check there are two available slides
    assert len(slide_select.options) == 2
    assert slide_select.value[0] == "CMU-1.ndpi"


def test_add_annotation_layer(bk_session):
    """Test adding an annotation layer."""
    slide_select = bk_session.document.get_model_by_name("slide_select0")
    slide_select.value = ["CMU-1-Small_region.svs"]
    bk_session.push()
    layer_drop = bk_session.document.get_model_by_name("layer_drop0")
    assert len(layer_drop.menu) == 2
    len(
        bk_session.document.get_model_by_name("slide_windows").children[0].renderers,
    )
    # trigger an event to select the annotation .db file
    click = MenuItemClick(layer_drop, layer_drop.menu[0][0])
    layer_drop._trigger_event(click)
    bk_session.push()
    cprop_input = bk_session.document.get_model_by_name("cprop0")
    cprop_input.value = ["prob"]
    bk_session.push()
    time.sleep(2)


def test_clearing_doc(bk_session):
    """Test that the doc can be cleared."""
    bk_session.close()
    bk_session.document.clear()
    assert len(bk_session.document.roots) == 0
