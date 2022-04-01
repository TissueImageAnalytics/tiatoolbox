"""Command line interface for show_overlay"""
import click

from tiatoolbox.cli.common import (
    tiatoolbox_cli,
)

@tiatoolbox_cli.command()
@click.argument('img_paths', nargs=-1, type=click.Path(exists=True))
def tile_server(img_paths):
    "show a slide together with a whole slide overlay"
    from tiatoolbox.utils.visualization import MakeTileServer

    app = MakeTileServer(img_paths)

    app.run()
