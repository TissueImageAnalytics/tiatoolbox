"""Command line interface for TileServer"""
import click

from tiatoolbox.cli.common import (
    tiatoolbox_cli,
)


@tiatoolbox_cli.command()
@click.argument("img_paths", nargs=-1, type=click.Path(exists=True))
def tile_server(img_paths):  # pragma: no cover
    """show a slide together with a whole slide overlay"""
    from tiatoolbox.visualization.tileserver import TileServer

    app = TileServer("TileServer", list(img_paths))

    app.run()
