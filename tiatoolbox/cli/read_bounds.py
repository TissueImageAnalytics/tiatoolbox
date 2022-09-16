"""Command line interface for read_bounds."""
import pathlib

from tiatoolbox.cli.common import (
    cli_img_input,
    cli_mode,
    cli_output_path,
    cli_region,
    cli_resolution,
    cli_units,
    no_input_message,
    tiatoolbox_cli,
)


@tiatoolbox_cli.command()
@cli_img_input(usage_help="Path to WSI file.")
@cli_output_path(
    usage_help="Path to output file in save mode. "
    "default=img_input_dir/../im_region.jpg"
)
@cli_region(
    usage_help="Image region in the whole slide image to read from. "
    "default=0 0 2000 2000"
)
@cli_resolution()
@cli_units()
@cli_mode(default="show")
def read_bounds(img_input, region, resolution, units, output_path, mode):
    """Read a region in a whole slide image as specified."""
    from PIL import Image

    from tiatoolbox.utils.misc import imwrite
    from tiatoolbox.wsicore.wsireader import WSIReader

    no_input_message(input_file=img_input)

    if not region:
        region = [0, 0, 2000, 2000]

    if output_path is None and mode == "save":
        input_dir = pathlib.Path(img_input).parent
        output_path = str(input_dir.parent / "im_region.jpg")

    wsi = WSIReader.open(input_img=img_input)

    im_region = wsi.read_bounds(
        region,
        resolution=resolution,
        units=units,
    )
    if mode == "show":  # pragma: no cover
        # Skipped on CI, and unless SHOW_TESTS is set
        im_region = Image.fromarray(im_region)
        im_region.show()
        return

    # the only other option left for mode is "save".
    imwrite(output_path, im_region)
