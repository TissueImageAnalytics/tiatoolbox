"""Command line interface for TileServer"""
from tiatoolbox.cli.common import tiatoolbox_cli, cli_img_input, cli_name


@tiatoolbox_cli.command()
@cli_img_input(multiple=True)
@cli_name(multiple=True)
def tile_server(img_input, name):  # pragma: no cover
    """show a slide, optionally together with one or more whole slide
    overlays
    """
    from tiatoolbox.visualization.tileserver import TileServer

    print(name)
    print(img_input)
    if len(name) == 0:
        app = TileServer("TileServer", list(img_input))
    elif len(name) == len(img_input):
        app = TileServer("TileServer", {n: im for n, im in zip(name, img_input)})
    else:
        raise (
            ValueError("if names are provided, must match the number of paths provided")
        )

    app.run()
