"""Test the bokeh app with config.json file."""

from __future__ import annotations

import time
from contextlib import suppress
from threading import Thread
from typing import TYPE_CHECKING

import pytest
import requests
from bokeh.client.session import ClientSession, pull_session

from tiatoolbox.cli.visualize import run_bokeh, run_tileserver
from tiatoolbox.data import _fetch_remote_sample

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module", autouse=True)
def annotation_path(data_path: dict[str, Path]) -> dict[str, Path]:
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


@pytest.fixture
def bk_session(data_path: dict[str, Path]) -> ClientSession:
    """Create a bokeh session."""
    run_tileserver()
    time.sleep(1)  # allow time for server to start

    args = [
        [
            str(data_path["base_path"] / "slides"),
            str(data_path["base_path"] / "overlays"),
        ],
        5006,
    ]
    kwargs = {"noshow": True}
    proc = Thread(target=run_bokeh, daemon=True, args=args, kwargs=kwargs)
    proc.start()
    time.sleep(5)  # allow time for server to start

    session = pull_session(
        url="http://localhost:5006/bokeh_app",
        arguments={"slide": "CMU-1-Small-Region.svs"},
    )
    yield session
    session.close()
    with suppress(requests.exceptions.ConnectionError):
        requests.post("http://localhost:5000/tileserver/shutdown", timeout=2)


def test_slides_available(bk_session: ClientSession) -> None:
    """Test that the slides and overlays are available."""
    doc = bk_session.document
    slide_select = doc.get_model_by_name("slide_select0")
    # check there are two available slides
    assert len(slide_select.options) == 2
    assert slide_select.value[0] == "CMU-1-Small-Region.svs"

    layer_drop = doc.get_model_by_name("layer_drop0")
    assert len(layer_drop.menu) == 2
    # check that the overlays are available.
    slide_select.value = ["CMU-1.ndpi"]
    assert len(layer_drop.menu) == 2

    bk_session.document.clear()
    assert len(bk_session.document.roots) == 0
    bk_session.close()
    time.sleep(5)  # allow time for hooks to trigger
