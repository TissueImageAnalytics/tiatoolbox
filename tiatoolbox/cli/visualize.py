"""Command line interface for visualization tool"""
import pathlib
import subprocess

import tiatoolbox.visualization as vis
from tiatoolbox.cli.common import cli_img_input, tiatoolbox_cli


@tiatoolbox_cli.command()
@cli_img_input(
    usage_help="""Path to base directory containing images to be displayed.
    If one instance of img-input is provided, Slides and overlays to be visualized are expected in
    subdirectories of the base directory named slides and overlays,
    respectively. It is also possible to provide a slide and overlay path separately""",
    multiple=True,
)
def visualize(img_input):
    """Launches the visualization tool for the given directory(s).
    If only one path is given, Slides and overlays to be visualized are expected in
    subdirectories of the base directory named slides and overlays,
    respectively.
    """
    vis_path = pathlib.Path(vis.__file__).resolve().parent

    if img_input is None:
        raise ValueError("No input directory specified.")
    for input_path in img_input:
        if not pathlib.Path(input_path).exists():
            raise FileNotFoundError(f"{input_path} does not exist")

    cmd = [
        "bokeh",
        "serve",
        "--show",
        str(vis_path.joinpath("render_demo")),
        "--unused-session-lifetime",
        "1000",
        "--check-unused-sessions",
        "1000",
        "--args",
        *img_input,
    ]
    subprocess.run(cmd)
