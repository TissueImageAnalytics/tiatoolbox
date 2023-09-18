"""Command line interface for visualization tool."""
import pathlib
import subprocess
from threading import Thread

import click
from flask_cors import CORS

import tiatoolbox.visualization as vis
from tiatoolbox.cli.common import cli_img_input, tiatoolbox_cli
from tiatoolbox.visualization.tileserver import TileServer


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
def visualize(img_input, port, noshow) -> None:
    """Launches the visualization tool for the given directory(s).

    If only one path is given, Slides and overlays to be visualized are expected in
    subdirectories of the base directory named slides and overlays,
    respectively.

    """
    vis_path = pathlib.Path(vis.__file__).resolve().parent
    if len(img_input) == 0:
        msg = "No input directory specified."
        raise ValueError(msg)
    for input_path in img_input:
        if not pathlib.Path(input_path).exists():
            msg = f"{input_path} does not exist"
            raise FileNotFoundError(msg)

    def run_app() -> None:
        """Helper function to launch a tileserver."""
        app = TileServer(
            title="Tiatoolbox TileServer",
            layers={},
        )
        CORS(app, send_wildcard=True)
        app.run(host="127.0.0.1", threaded=False)

    # start tile server
    proc = Thread(target=run_app, daemon=True)
    proc.start()

    cmd = [
        "bokeh",
        "serve",
    ]
    if not noshow:
        cmd = [*cmd, "--show"]  # pragma: no cover
    cmd = [
        *cmd,
        str(vis_path.joinpath("bokeh_app")),
        "--port",
        str(port),
        "--unused-session-lifetime",
        "1000",
        "--check-unused-sessions",
        "1000",
        "--args",
        *img_input,
    ]
    subprocess.run(cmd, check=True)  # noqa: S603
