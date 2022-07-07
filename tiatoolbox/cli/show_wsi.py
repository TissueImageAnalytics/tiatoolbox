"""Command line interface for TileServer"""
from tiatoolbox.cli.common import cli_img_input, cli_name, tiatoolbox_cli


@tiatoolbox_cli.command()
@cli_img_input(usage_help="Path to an image to be displayed.", multiple=True)
@cli_name(usage_help="Name to be assigned to a layer.", multiple=True)
def show_wsi(img_input, name):  # pragma: no cover
    """Show a slide together with any overlays."""
    from tiatoolbox.visualization.tileserver import TileServer

    if len(img_input) == 0:
        raise ValueError("At least one image path must be provided.")
    if len(name) == 0:
        app = TileServer("TileServer", list(img_input))
    elif len(name) == len(img_input):
        app = TileServer("TileServer", dict(zip(name, img_input)))
    else:
        raise (
            ValueError("if names are provided, must match the number of paths provided")
        )

    app.run()
