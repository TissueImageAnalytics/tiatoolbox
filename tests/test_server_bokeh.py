"""Test the bokeh app from command line."""
import os
import time
from threading import Thread

import pkg_resources
import pytest
from click.testing import CliRunner

from bokeh.client.session import ClientSession, pull_session
from tiatoolbox import cli
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
        "config_1",
        data_path["base_path"] / "overlays",
    )
    return data_path


@pytest.fixture()
def bk_session(data_path) -> ClientSession:
    """Create a bokeh session."""

    def run_app() -> None:
        """Start a server to run the bokeh app."""
        runner = CliRunner()
        runner.invoke(
            cli.main,
            [
                "visualize",
                "--noshow",
                "--img-input",
                str(data_path["base_path"].parent),
            ],
            env=os.environ,
        )

    proc = Thread(target=run_app, daemon=True)
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


def test_slides_available(bk_session):
    """Test that the slides and overlays are available."""
    doc = bk_session.document
    slide_select = doc.get_model_by_name("slide_select0")
    # check there are two available slides
    assert len(slide_select.options) == 2
    assert slide_select.value[0] == "CMU-1.ndpi"

    # check that the overlays are available.
    slide_select = doc.get_model_by_name("slide_select0")
    slide_select.value = ["CMU-1-Small-region.svs"]
    layer_drop = doc.get_model_by_name("layer_drop0")
    assert len(layer_drop.menu) == 2

    bk_session.document.clear()
    assert len(bk_session.document.roots) == 0
    bk_session.close()
    time.sleep(10)  # allow time for hooks to trigger


def test_cli_errors(data_path):
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
    assert result.exc_info[1].args[0] == "No input directory specified."

    # test with non-existent input folder
    result = runner.invoke(
        cli.main,
        [
            "visualize",
            "--noshow",
            "--img-input",
            str(data_path["base_path"] / "slides"),
            "--img-input",
            "non_existent_folder",
        ],
    )
    assert result.exit_code == 1
    assert result.exc_info[1].args[0] == "non_existent_folder does not exist"
