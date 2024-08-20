"""Test the bokeh app from command line."""

from __future__ import annotations

import time
from contextlib import suppress
from threading import Thread
from typing import TYPE_CHECKING

import pytest
import requests
from bokeh.client.session import ClientSession, pull_session
from click.testing import CliRunner

from tiatoolbox import cli
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
    data_path["config"] = _fetch_remote_sample(
        "config_1",
        data_path["base_path"] / "overlays",
    )
    return data_path


@pytest.fixture
def bk_session(data_path: dict[str, Path]) -> ClientSession:
    """Create a bokeh session."""
    run_tileserver()
    time.sleep(1)  # allow time for server to start

    args = [
        [str(data_path["base_path"].parent)],
        5006,
    ]
    kwargs = {"noshow": True}
    proc = Thread(target=run_bokeh, daemon=True, args=args, kwargs=kwargs)
    proc.start()
    time.sleep(5)  # allow time for server to start

    session = pull_session(
        url="http://localhost:5006/bokeh_app",
        arguments={
            "demo": str(data_path["base_path"].parts[-1]),
            "slide": "CMU-1.ndpi",
            "window": "[0, 0, 1000, 1000]",
        },
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

    # check that the overlays are available.
    slide_select.value = ["CMU-1-Small-region.svs"]
    layer_drop = doc.get_model_by_name("layer_drop0")
    assert len(layer_drop.menu) == 2

    bk_session.document.clear()
    assert len(bk_session.document.roots) == 0
    bk_session.close()
    time.sleep(5)  # allow time for hooks to trigger


def test_cli_errors(data_path: dict[str, Path]) -> None:
    """Test that the cli raises errors when expected."""
    runner = CliRunner()
    # test with no input folder
    result = runner.invoke(
        cli.main,
        [
            "visualize",
            "--noshow",
        ],
    )
    assert result.exit_code == 1
    assert (
        result.exc_info[1].args[0]
        == "Must specify either base-path or both slides and overlays."
    )

    # test with non-existent input folder
    result = runner.invoke(
        cli.main,
        [
            "visualize",
            "--noshow",
            "--slides",
            str(data_path["base_path"] / "slides"),
            "--overlays",
            "non_existent_folder",
        ],
    )
    assert result.exit_code == 1
    assert result.exc_info[1].args[0] == "non_existent_folder does not exist"
