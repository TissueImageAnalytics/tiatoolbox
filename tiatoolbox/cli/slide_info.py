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
import sys

import click

from tiatoolbox import utils, wsicore
from tiatoolbox.cli.common import (
    cli_file_type,
    cli_img_input,
    cli_mode,
    cli_output_path,
    cli_verbose,
    prepare_file_dir_cli,
)


@click.group()
def main():  # pragma: no cover
    """Define slide_info click group."""
    return 0


@main.command()
@cli_img_input
@cli_output_path
@cli_file_type
@cli_mode
@cli_verbose
def slide_info(img_input, output_path, file_types, mode, verbose):
    """Display or save WSI metadata."""
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


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
