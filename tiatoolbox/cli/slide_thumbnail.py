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

from PIL import Image

from tiatoolbox import utils
from tiatoolbox.cli.common import (
    TIAToolboxCLI,
    cli_file_type,
    cli_img_input,
    cli_mode,
    cli_output_path,
    no_input_message,
    prepare_file_dir_cli,
)
from tiatoolbox.wsicore.wsireader import WSIReader

slide_thumbnail_cli = TIAToolboxCLI()
slide_thumbnail_cli.help = (
    "Reads whole slide image thumbnail and shows or saves based on mode argument."
)


@slide_thumbnail_cli.command()
@cli_img_input
@cli_output_path
@cli_file_type
@cli_mode
def slide_thumbnail(img_input, output_path, file_types, mode):
    """Read whole slide image thumbnail."""
    if img_input is None:
        no_input_message("No image input provided.\n")

    files_all, output_path = prepare_file_dir_cli(
        img_input, output_path, file_types, mode, "output"
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
