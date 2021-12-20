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

"""Command line interface for slide_thumbnail."""
import pathlib

import click
from PIL import Image

from tiatoolbox import utils
from tiatoolbox.wsicore.wsireader import WSIReader


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
    "--file-types",
    help="file types to capture from directory, default='*.ndpi', '*.svs', '*.mrxs'",
    default="*.ndpi, *.svs, *.mrxs, *.jp2",
)
@click.option(
    "--mode",
    default="save",
    help="'show' to display image region or 'save' to save at the output path"
    ", default=save",
)
def slide_thumbnail(img_input, output_path, file_types, mode):
    """Read whole slide image thumbnail."""
    files_all, output_path = utils.misc.prepare_file_dir_cli(
        img_input, output_path, file_types, mode, "slide-thumbnail"
    )

    for curr_file in files_all:
        wsi = WSIReader.open(input_img=curr_file)

        slide_thumb = wsi.slide_thumbnail()
        if mode == "show":
            im_region = Image.fromarray(slide_thumb)
            im_region.show()

        if mode == "save":
            utils.misc.imwrite(
                output_path / (pathlib.Path(curr_file).stem + ".jpg"), slide_thumb
            )
