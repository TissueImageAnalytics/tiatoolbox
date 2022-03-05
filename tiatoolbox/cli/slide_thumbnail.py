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
    cli_file_type,
    cli_img_input,
    cli_mode,
    cli_output_path,
    prepare_file_dir_cli,
    tiatoolbox_cli,
)
from tiatoolbox.wsicore.wsireader import WSIReader


@tiatoolbox_cli.command()
@cli_img_input()
@cli_output_path(
    usage_help="Path to output directory to save the output. "
    "default=img_input/../slide-thumbnail"
)
@cli_file_type(default="*.ndpi, *.svs, *.mrxs, *.jp2")
@cli_mode(default="save")
def slide_thumbnail(img_input, output_path, file_types, mode):
    """Reads whole slide image thumbnail and shows or saves based on mode argument.

    The default inputs are:

    img-input='', output-path=img-input-path/../meta-data,  mode="save",
    file-types="*.ndpi, *.svs, *.mrxs, *.jp2".

    """
    files_all, output_path = prepare_file_dir_cli(
        img_input, output_path, file_types, mode, "slide-thumbnail"
    )

    for curr_file in files_all:
        wsi = WSIReader.open(input_img=curr_file)

        slide_thumb = wsi.slide_thumbnail()
        if mode == "show":  # pragma: no cover
            # skipped on travis
            im_region = Image.fromarray(slide_thumb)
            im_region.show()

        # the only other option left for mode is "save".
        utils.misc.imwrite(
            output_path / (pathlib.Path(curr_file).stem + ".jpg"), slide_thumb
        )
