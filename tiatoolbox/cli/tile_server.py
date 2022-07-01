"""Command line interface for TileServer"""
from tiatoolbox.cli.common import cli_img_input, cli_name, tiatoolbox_cli


@tiatoolbox_cli.command()
@cli_img_input(multiple=True)
@cli_name(multiple=True)
def tile_server(img_input, name):  # pragma: no cover
    """Show a slide together with any overlays.

    Args:
        img_input (tuple[str]):
            Paths to images to be displayed.
        name (tuple[str]):
            Names to be assigned to each layer.

    """
    from tiatoolbox.visualization.tileserver import TileServer

    if len(name) == 0:
        app = TileServer("TileServer", list(img_input))
    elif len(name) == len(img_input):
        app = TileServer("TileServer", dict(zip(name, img_input)))
    else:
        raise (
            ValueError("if names are provided, must match the number of paths provided")
        )

    app.run()
