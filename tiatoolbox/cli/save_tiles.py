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

"""Command line interface for save_tiles."""
import os

import click

from tiatoolbox import utils, wsicore


@click.group()
def main():  # pragma: no cover
    """Define save_tiles click group."""
    return 0


@main.command()
@click.option("--img-input", help="input path to WSI file or directory path")
@click.option(
    "--output-path",
    default="tiles",
    help="Path to output directory to save the output, default=tiles",
)
@click.option(
    "--file-types",
    help="file types to capture from directory, default='*.ndpi', '*.svs', '*.mrxs'",
    default="*.ndpi, *.svs, *.mrxs, *.jp2",
)
@click.option(
    "--tile-objective-value",
    type=int,
    default=20,
    help="objective value at which tile is generated- default=20",
)
@click.option(
    "--tile-read-size",
    type=int,
    nargs=2,
    default=[5000, 5000],
    help="tile width, height default=5000 5000",
)
@click.option(
    "--verbose",
    type=bool,
    default=True,
    help="Print output, default=True",
)
def save_tiles(
    img_input,
    output_path,
    file_types,
    tile_objective_value,
    tile_read_size,
    verbose=True,
):
    """Display or save WSI metadata."""
    file_types = utils.misc.string_to_tuple(in_str=file_types)

    if not os.path.exists(img_input):
        raise FileNotFoundError

    files_all = [
        img_input,
    ]

    if os.path.isdir(img_input):
        files_all = utils.misc.grab_files_from_dir(
            input_path=img_input, file_types=file_types
        )

    print(files_all)

    for curr_file in files_all:
        wsicore.save_tiles.save_tiles(
            input_path=curr_file,
            output_dir=output_path,
            tile_objective_value=tile_objective_value,
            tile_read_size=tile_read_size,
            verbose=verbose,
        )
