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

"""Command line interface for slide_info."""
import pathlib

from tiatoolbox import utils, wsicore
from tiatoolbox.cli.common import (
    TIAToolboxCLI,
    cli_file_type,
    cli_img_input,
    cli_mode,
    cli_output_path,
    cli_verbose,
    no_input_message,
    prepare_file_dir_cli,
)

slide_info_cli = TIAToolboxCLI()
slide_info_cli.help = "Displays or saves WSI metadata."


@slide_info_cli.command()
@cli_img_input
@cli_output_path
@cli_file_type
@cli_mode
@cli_verbose
def slide_info(img_input, output_path, file_types, mode, verbose):
    """Display or save WSI metadata."""
    if img_input is None:
        print("No image input provided.\n")
        return no_input_message()

    files_all, output_path = prepare_file_dir_cli(
        img_input, output_path, file_types, mode, "output"
    )

    for curr_file in files_all:
        slide_param = wsicore.slide_info.slide_info(
            input_path=curr_file, verbose=verbose
        )

        if mode == "save":
            out_path = pathlib.Path(
                output_path, slide_param.file_path.with_suffix(".yaml").name
            )
            utils.misc.save_yaml(
                slide_param.as_dict(),
                out_path,
            )
            print("Meta files saved at " + str(output_path))

    return 0
