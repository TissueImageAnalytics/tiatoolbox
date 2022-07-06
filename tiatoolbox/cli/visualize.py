"""Command line interface for visualization tool"""
import pathlib
import subprocess

from tiatoolbox.cli.common import (
    cli_img_input,
    tiatoolbox_cli,
)


@tiatoolbox_cli.command()
@cli_img_input()
def visualize(img_input):
    """Launches the visualization tool for the given base directory.
    Slides and overlays to be visualized are expected in 
    subdirectories of the base directory named slides and overlays, 
    respectively.
    """
    cmd = ["bokeh",
           "serve",
           "--show",
           "./tiatoolbox/visualization/render_demo",
           "--unused-session-lifetime",
            "1000",
            "--check-unused-sessions",
            "1000",
            "--args",
            img_input,]
    subprocess.run(cmd)
