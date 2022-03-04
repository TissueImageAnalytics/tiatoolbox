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
import os
import pathlib

import click

from tiatoolbox.utils.misc import grab_files_from_dir, string_to_tuple

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


class TIAToolboxCLI(click.Group):
    def __init__(self, *args, **kwargs):
        super(TIAToolboxCLI, self).__init__(*args, **kwargs)
        self.help = "Computational pathology toolbox by TIA Centre."
        self.add_help_option = {"help_option_names": ["-h", "--help"]}


def no_input_message(message):
    """This function is called if no input is provided.

    Args:
        message (str): Failure message to display.

    """
    ctx = click.get_current_context()
    ctx.fail(message=message)


def prepare_file_dir_cli(img_input, output_path, file_types, mode, sub_dirname):
    """Prepares CLI for running code on multiple files or a directory.

    Checks for existing directories to run tests.
    Converts file path to list of file paths or
    creates list of file paths if input is a directory.

    Args:
        img_input (str or pathlib.Path): file path to images.
        output_path (str or pathlib.Path): output directory path.
        file_types (str): file types to process using cli.
        mode (str): wsi or tile mode.
        sub_dirname (str): name of subdirectory to save output.

    Returns:
        files_all (list): list of file paths to process.
        output_path (pathlib.Path): updated output path.

    """
    file_types = string_to_tuple(in_str=file_types)

    if isinstance(output_path, str):
        output_path = pathlib.Path(output_path)

    if not os.path.exists(img_input):
        raise FileNotFoundError

    files_all = [
        img_input,
    ]

    if os.path.isdir(img_input):
        files_all = grab_files_from_dir(input_path=img_input, file_types=file_types)

    if output_path is None and mode == "save":
        input_dir = pathlib.Path(img_input).parent
        output_path = input_dir / sub_dirname

    if mode == "save":
        output_path.mkdir(parents=True, exist_ok=True)

    return files_all, output_path


def prepare_model_cli(img_input, output_path, masks, file_types, mode):
    """Prepares cli for running models.

    Checks for existing directories to run tests.
    Converts file path to list of file paths or
    creates list of file paths if input is a directory.

    Args:
        img_input (str or pathlib.Path): file path to images.
        output_path (str or pathlib.Path): output directory path.
        masks (str or pathlib.Path): file path to masks.
        file_types (str): file types to process using cli.
        mode (str): wsi or tile mode.

    Returns:
        files_all (list): list of file paths to process.
        masks_all (list): list of masks corresponding to input files.
        output_path (pathlib.Path): output path

    """
    output_path = pathlib.Path(output_path)
    file_types = string_to_tuple(in_str=file_types)

    if output_path.exists():
        raise FileExistsError("Path already exists.")

    if not os.path.exists(img_input):
        raise FileNotFoundError

    if mode not in ["wsi", "tile"]:
        raise ValueError('`mode` must be in ("wsi", "tile").')

    files_all = [
        img_input,
    ]

    if masks is None:
        masks_all = None
    else:
        masks_all = [
            masks,
        ]

    if os.path.isdir(img_input):
        files_all = grab_files_from_dir(input_path=img_input, file_types=file_types)

    if os.path.isdir(str(masks)):
        masks_all = grab_files_from_dir(input_path=masks, file_types=("*.jpg", "*.png"))

    return files_all, masks_all, output_path
