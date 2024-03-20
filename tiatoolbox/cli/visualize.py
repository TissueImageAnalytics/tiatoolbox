"""Command line interface for visualization tool."""

from __future__ import annotations

import importlib.resources as importlib_resources
import os
import subprocess
from pathlib import Path
from threading import Thread

import click
from flask_cors import CORS

from tiatoolbox.cli.common import tiatoolbox_cli


def run_tileserver() -> None:
    """Helper function to launch a tileserver."""

    def run_app() -> None:
        """Run the tileserver app."""
        from tiatoolbox.visualization.tileserver import TileServer

        app = TileServer(
            title="Tiatoolbox TileServer",
            layers={},
        )
        CORS(app, send_wildcard=True)
        app.run(host="127.0.0.1", threaded=True)

    proc = Thread(target=run_app, daemon=True)
    proc.start()


def run_bokeh(img_input: list[str], port: int, *, noshow: bool) -> None:
    """Start the bokeh server."""
    bokeh_path = importlib_resources.files("tiatoolbox.visualization.bokeh_app")
    cmd = [
        "bokeh",
        "serve",
    ]
    if not noshow:
        cmd = [*cmd, "--show"]  # pragma: no cover
    cmd = [
        *cmd,
        bokeh_path,
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
@click.option(
    "--base-path",
    help="""Path to base directory containing images to be displayed.
    Slides and overlays to be visualized are expected in subdirectories of the
    base directory named slides and overlays, respectively. It is also possible
    to provide a slide and overlay path separately
    (use --slides and --overlays).""",
)
@click.option(
    "--slides",
    help="""Path to directory containing slides to be displayed.
    This option must be used in conjunction with --overlay-path.
    The --base-path option should not be used in this case.""",
)
@click.option(
    "--overlays",
    help="""Path to directory containing overlays to be displayed.
    This option must be used in conjunction with --slides.
    The --base-path option should not be used in this case.""",
)
@click.option(
    "--port",
    type=int,
    help="Port to launch the visualization tool on.",
    default=5006,
)
@click.option("--noshow", is_flag=True, help="Do not launch browser.")
def visualize(
    base_path: str,
    slides: str,
    overlays: str,
    port: int,
    *,
    noshow: bool,
) -> None:
    """Launches the visualization tool for the given directory(s).

    If only base-path is given, Slides and overlays to be visualized are expected in
    subdirectories of the base directory named slides and overlays, respectively.

    Args:
        base_path (str): Path to base directory containing images to be displayed.
        slides (str): Path to directory containing slides to be displayed.
        overlays (str): Path to directory containing overlays to be displayed.
        port (int): Port to launch the visualization tool on.
        noshow (bool): Do not launch in browser (mainly intended for testing).

    """
    # sanity check the input args
    if base_path is None and (slides is None or overlays is None):
        msg = "Must specify either base-path or both slides and overlays."
        raise ValueError(msg)
    img_input = [base_path, slides, overlays]
    img_input = [p for p in img_input if p is not None]
    # check that the input paths exist
    for input_path in img_input:
        if not Path(input_path).exists():
            msg = f"{input_path} does not exist"
            raise FileNotFoundError(msg)

    # start servers
    run_tileserver()  # pragma: no cover
    run_bokeh(img_input, port, noshow=noshow)  # pragma: no cover
