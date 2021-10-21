"""Command line interface for read_bounds."""
import pathlib

import click
from PIL import Image

from tiatoolbox import utils, wsicore


@click.group()
def main():  # pragma: no cover
    """Define read_bounds click group."""
    return 0


@main.command()
@click.option("--img-input", help="Path to WSI file")
@click.option(
    "--output-path",
    help="Path to output file to save the image region in save mode,"
    " default=img_input_dir/../im_region.jpg",
)
@click.option(
    "--region",
    type=int,
    nargs=4,
    help="image region in the whole slide image to read, default=0 0 2000 2000",
)
@click.option(
    "--resolution",
    type=float,
    default=0,
    help="resolution to read the image at, default=0",
)
@click.option(
    "--units",
    default="level",
    type=click.Choice(["mpp", "power", "level", "baseline"], case_sensitive=False),
    help="resolution units, default=level",
)
@click.option(
    "--mode",
    default="show",
    help="'show' to display image region or 'save' to save at the output path"
    ", default=show",
)
def read_bounds(img_input, region, resolution, units, output_path, mode):
    """Read a region in an whole slide image as specified."""
    if not region:
        region = [0, 0, 2000, 2000]

    if output_path is None and mode == "save":
        input_dir = pathlib.Path(img_input).parent
        output_path = str(input_dir.parent / "im_region.jpg")

    wsi = wsicore.wsireader.get_wsireader(input_img=img_input)

    im_region = wsi.read_bounds(
        region,
        resolution=resolution,
        units=units,
    )
    if mode == "show":
        im_region = Image.fromarray(im_region)
        im_region.show()

    if mode == "save":
        utils.misc.imwrite(output_path, im_region)
