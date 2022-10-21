"""Command line interface for TileServer"""
import click

from tiatoolbox.cli.common import cli_img_input, cli_name, tiatoolbox_cli


@tiatoolbox_cli.command()
@cli_img_input(usage_help="Path to an image to be displayed.", multiple=True)
@cli_name(usage_help="Name to be assigned to a layer.", multiple=True)
@click.option(
    "--colour-by",
    "-c",
    default=None,
    help="""A property to colour by. Must also define a colour
    map if this option is used, using --colour-map (or -m).
    Will only affect layers rendered from a store.""",
)
@click.option(
    "--colour-map",
    "-m",
    default=None,
    help="""A colour map to use. Must be a matplotlib
    colour map string or 'categorical' (random colours will be generated
    for each possible value of property).""",
)
def show_wsi(img_input, name, colour_by, colour_map):  # pragma: no cover
    """Show a slide together with any overlays."""
    from tiatoolbox.utils.visualization import AnnotationRenderer
    from tiatoolbox.visualization.tileserver import TileServer

    renderer = AnnotationRenderer()
    if colour_by is not None:
        if colour_map is None:
            raise ValueError(
                "If colouring by a property, must also define a colour map."
            )
        renderer = AnnotationRenderer(score_prop=colour_by, mapper=colour_map)

    if len(img_input) == 0:
        raise ValueError("At least one image path must be provided.")
    if len(name) == 0:
        app = TileServer("TileServer", list(img_input), renderer=renderer)
    elif len(name) == len(img_input):
        app = TileServer("TileServer", dict(zip(name, img_input)), renderer=renderer)
    else:
        raise (
            ValueError("if names are provided, must match the number of paths provided")
        )

    app.run(threaded=False)
