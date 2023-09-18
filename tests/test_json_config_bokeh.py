"""Test the bokeh app with config.json file."""
import time
from threading import Thread

import pytest
from bokeh.client.session import ClientSession, pull_session

from tiatoolbox.cli.visualize import run_bokeh, run_tileserver
from tiatoolbox.data import _fetch_remote_sample


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
    return data_path


@pytest.fixture()
def bk_session(data_path) -> ClientSession:
    """Create a bokeh session."""
    run_tileserver()
    time.sleep(1)  # allow time for server to start

    args = [
        [str(data_path["base_path"])],
        5006,
        True,
    ]
    proc = Thread(target=run_bokeh, daemon=True, args=args)
    proc.start()
    time.sleep(5)  # allow time for server to start

    session = pull_session(
        url="http://localhost:5006/bokeh_app",
        arguments={},
    )
    yield session
    session.close()


def test_slides_available(bk_session):
    """Test that the slides and overlays are available."""
    doc = bk_session.document
    slide_select = doc.get_model_by_name("slide_select0")
    # check there are two available slides
    assert len(slide_select.options) == 2

    # check that the overlays are available.
    slide_select.value = ["CMU-1-Small-region.svs"]
    layer_drop = doc.get_model_by_name("layer_drop0")
    assert len(layer_drop.menu) == 2

    bk_session.document.clear()
    assert len(bk_session.document.roots) == 0
    bk_session.close()
    time.sleep(5)  # allow time for hooks to trigger
