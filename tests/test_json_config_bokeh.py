"""Test the bokeh app with config.json file."""
import io
import json
import time
from contextlib import suppress
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import pytest
import requests
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from PIL import Image

from tiatoolbox.data import _fetch_remote_sample
from tiatoolbox.visualization.bokeh_app import main

BOKEH_PATH = pkg_resources.resource_filename("tiatoolbox", "visualization/bokeh_app")


def get_tile(layer, x, y, z, *, show: bool):
    """Get a tile from the server."""
    source = main.UI["p"].renderers[main.UI["vstate"].layer_dict[layer]].tile_source
    url = source.url
    # replace {x}, {y}, {z} with tile coordinates
    url = url.replace("{x}", str(x)).replace("{y}", str(y)).replace("{z}", str(z))
    im = io.BytesIO(requests.get(url, timeout=100).content)
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
        "config_1",
        data_path["base_path"] / "overlays",
    )
    return data_path


@pytest.fixture(scope="module")
def doc(data_path):
    """Create a test document for the visualization tool."""
    # make a bokeh app
    main.doc_config.set_sys_args(argv=["dummy_str", str(data_path["base_path"])])
    handler = FunctionHandler(main.doc_config.setup_doc)
    app = Application(handler)
    return app.create_document()


def test_roots(doc):
    """Test that the bokeh app has the correct number of roots."""
    # should be 2 roots, main window and controls
    assert len(doc.roots) == 2


def test_config_loaded(data_path):
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


def test_slide_select(doc, data_path):
    """Test slide selection."""
    slide_select = doc.get_model_by_name("slide_select0")
    # check there are two available slides
    assert len(slide_select.options) == 2
    assert slide_select.options[0][0] == data_path["slide1"].name

    # select a slide and check it is loaded
    slide_select.value = ["CMU-1-Small-Region.svs"]
    assert main.UI["vstate"].slide_path == data_path["slide1"]


def test_clearing_doc(doc):
    """Test that the doc can be cleared."""
    doc.clear()
    assert len(doc.roots) == 0
    with suppress(requests.exceptions.ConnectionError):
        # may not respond if already shutdown
        main.UI["s"].post(f"http://{main.host2}:5000/tileserver/shutdown", timeout=5)
    time.sleep(5)
