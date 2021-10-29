"""Command line interface for slide_thumbnail."""
import pathlib

import click
from PIL import Image

from tiatoolbox import utils, wsicore


@click.group()
def main():  # pragma: no cover
    """Define slide_thumbnail click group."""
    return 0


@main.command()
@click.option("--img-input", help="Path to WSI file")
@click.option(
    "--output-path",
    help="Path to output file to save the image region in save mode,"
    " default=img_input_dir/../slide_thumb.jpg",
)
@click.option(
    "--mode",
    default="show",
    help="'show' to display image region or 'save' to save at the output path"
    ", default=show",
)
def slide_thumbnail(img_input, output_path, mode):
    """Read whole slide image thumbnail."""
    if output_path is None and mode == "save":
        input_dir = pathlib.Path(img_input).parent
        output_path = str(input_dir.parent / "slide_thumb.jpg")

    wsi = wsicore.wsireader.get_wsireader(input_img=img_input)

    slide_thumb = wsi.slide_thumbnail()

    if mode == "show":
        im_region = Image.fromarray(slide_thumb)
        im_region.show()

    if mode == "save":
        utils.misc.imwrite(output_path, slide_thumb)
