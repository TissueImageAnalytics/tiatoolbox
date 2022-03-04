# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Command line interface for read_bounds."""
import pathlib

from PIL import Image

from tiatoolbox import utils
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
from tiatoolbox.wsicore.wsireader import WSIReader


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
        im_region = Image.fromarray(im_region)
        im_region.show()

    if mode == "save":
        utils.misc.imwrite(output_path, im_region)
