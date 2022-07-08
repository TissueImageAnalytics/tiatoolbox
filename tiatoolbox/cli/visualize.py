"""Command line interface for visualization tool"""
import pathlib
import subprocess
import tiatoolbox.visualization as vis

from tiatoolbox.cli.common import (
    cli_img_input,
    tiatoolbox_cli,
)


@tiatoolbox_cli.command()
@cli_img_input(usage_help="""Path to base directory containing images to be displayed.
    Slides and overlays to be visualized are expected in 
    subdirectories of the base directory named slides and overlays, 
    respectively.""")
def visualize(img_input):
    """Launches the visualization tool for the given base directory.
    Slides and overlays to be visualized are expected in 
    subdirectories of the base directory named slides and overlays, 
    respectively.
    """
    vis_path = pathlib.Path(vis.__file__).resolve().parent

    if img_input is None:
        raise ValueError("No input directory specified.")
    if not pathlib.Path(img_input).exists():
        raise FileNotFoundError(f"{img_input} does not exist")

    cmd = ["bokeh",
           "serve",
           "--show",
           str(vis_path.joinpath("render_demo")),
           "--unused-session-lifetime",
            "1000",
            "--check-unused-sessions",
            "1000",
            "--args",
            img_input,]
    subprocess.run(cmd)
