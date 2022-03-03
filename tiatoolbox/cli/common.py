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

"""Defines common code required for cli."""
import click

cli_img_input = click.option(
    "--img-input", help="input path to WSI file or directory path"
)
cli_output_path = click.option(
    "--output-path",
    help="Path to output directory to save the output, default=img_input/../output",
)
cli_file_type = click.option(
    "--file-types",
    help="file types to capture from directory, default='*.ndpi', '*.svs', '*.mrxs'",
    default="*.ndpi, *.svs, *.mrxs, *.jp2",
)
cli_mode = click.option(
    "--mode",
    default="show",
    help="'show' to display the image or required information or 'save' to save "
    "the output, default=save",
)
cli_verbose = click.option(
    "--verbose",
    type=bool,
    default=True,
    help="Print output, default=True",
)
