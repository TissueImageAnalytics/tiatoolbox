"""Command line interface for visualization tool."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from threading import Thread

import click
import pkg_resources
from flask_cors import CORS

from tiatoolbox.cli.common import cli_img_input, tiatoolbox_cli
from tiatoolbox.visualization.tileserver import TileServer

BOKEH_PATH = pkg_resources.resource_filename("tiatoolbox", "visualization/bokeh_app")


def run_tileserver() -> None:
    """Helper function to launch a tileserver."""

    def run_app() -> None:
        """Run the tileserver app."""
        app = TileServer(
            title="Tiatoolbox TileServer",
            layers={},
        )
        CORS(app, send_wildcard=True)
        app.run(host="127.0.0.1", threaded=False)

    proc = Thread(target=run_app, daemon=True)
    proc.start()


def run_bokeh(img_input: list[str], port: int, *, noshow: bool) -> None:
    """Start the bokeh server."""
    cmd = [
        "bokeh",
        "serve",
    ]
    if not noshow:
        cmd = [*cmd, "--show"]  # pragma: no cover
    cmd = [
        *cmd,
        BOKEH_PATH,
        "--port",
        str(port),
        "--unused-session-lifetime",
        "1000",
        "--check-unused-sessions",
        "1000",
        "--args",
        *img_input,
    ]
    subprocess.run(cmd, check=True, cwd=str(Path.cwd()), env=os.environ)  # noqa: S603


@tiatoolbox_cli.command()
@cli_img_input(
    usage_help="""Path to base directory containing images to be displayed.
    If one instance of img-input is provided, Slides and overlays to be visualized
    are expected in subdirectories of the base directory named slides and overlays,
    respectively. It is also possible to provide a slide and overlay
    path separately""",
    multiple=True,
)
@click.option(
    "--port",
    type=int,
    help="Port to launch the visualization tool on.",
    default=5006,
)
@click.option("--noshow", is_flag=True, help="Do not launch browser.")
def visualize(img_input: list[str], port: int, *, noshow: bool) -> None:
    """Launches the visualization tool for the given directory(s).

    If only one path is given, Slides and overlays to be visualized are expected in
    subdirectories of the base directory named slides and overlays,
    respectively.

    """
    # sanity check the input args
    if len(img_input) == 0:
        msg = "No input directory specified."
        raise ValueError(msg)
    for input_path in img_input:
        if not Path(input_path).exists():
            msg = f"{input_path} does not exist"
            raise FileNotFoundError(msg)

    # start servers
    run_tileserver()  # pragma: no cover
    run_bokeh(img_input, port, noshow)  # pragma: no cover
