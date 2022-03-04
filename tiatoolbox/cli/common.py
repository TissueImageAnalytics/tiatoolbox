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


def add_default_to_usage_help(usage_help: str, default: str or int or float) -> str:
    """Adds default value to usage help string.

    Args:
        usage_help (str):
            usage help for click option.
        default (str or int or float):
            default value as string for click option.

    Returns:
        str:
            New usage_help value.

    """
    if default is not None:
        return f"{usage_help} default={default}"
    else:
        return usage_help


def cli_img_input(
    usage_help: str = "Input path to WSI file or directory path.",
) -> callable:
    """Enables --img-input option for cli."""
    return click.option("--img-input", help=usage_help, type=str)


def cli_output_path(
    usage_help: str = "Path to output directory to save the output.",
    default: str = None,
) -> callable:
    """Enables --output-path option for cli."""
    return click.option(
        "--output-path",
        help=add_default_to_usage_help(usage_help, default),
        type=str,
    )


def cli_file_type(
    usage_help: str = "File types to capture from directory.",
    default: str = "*.ndpi, *.svs, *.mrxs, *.jp2",
) -> callable:
    """Enables --file-types option for cli."""
    return click.option(
        "--file-types",
        help=add_default_to_usage_help(usage_help, default),
        default=default,
        type=str,
    )


def cli_mode(
    usage_help: str = "Selected mode to show or save the required information.",
    default: str = "save",
) -> callable:
    """Enables --mode option for cli."""
    return click.option(
        "--mode",
        help=add_default_to_usage_help(usage_help, default),
        default=default,
        type=str,
    )


def cli_region(
    usage_help: str = "Image region in the whole slide image to read from. "
    "default=0 0 2000 2000",
) -> callable:
    """Enables --region option for cli."""
    return click.option(
        "--region",
        type=int,
        nargs=4,
        help=usage_help,
    )


def cli_units(
    usage_help: str = "Image resolution units to read the image.",
    default: str = "level",
    input_type: click.Choice = None,
) -> callable:
    """Enables --units option for cli."""
    if input_type is None:
        input_type = click.Choice(
            ["mpp", "power", "level", "baseline"], case_sensitive=False
        )
    return click.option(
        "--units",
        default=default,
        type=input_type,
        help=add_default_to_usage_help(usage_help, default),
    )


def cli_resolution(
    usage_help: str = "Image resolution to read the image.", default: float = 0
) -> callable:
    """Enables --resolution option for cli."""
    return click.option(
        "--resolution",
        type=float,
        default=default,
        help=add_default_to_usage_help(usage_help, default),
    )


def cli_tile_objective(
    usage_help: str = "Objective value for the saved tiles.", default: int = 20
) -> callable:
    """Enables --tile-objective-value option for cli."""
    return click.option(
        "--tile-objective-value",
        type=int,
        default=default,
        help=add_default_to_usage_help(usage_help, default),
    )


def cli_tile_read_size(
    usage_help: str = "Width and Height of saved tiles. default=5000 5000",
) -> callable:
    """Enables --tile-objective-value option for cli."""
    return click.option(
        "--tile-read-size",
        type=int,
        nargs=2,
        default=[5000, 5000],
        help=usage_help,
    )


def cli_method(
    usage_help: str = "Select method of choice.", default: str = "Otsu"
) -> callable:
    """Enables --tile-objective-value option for cli."""
    return click.option(
        "--method",
        type=click.Choice(["Otsu", "Morphological"], case_sensitive=True),
        default=default,
        help=add_default_to_usage_help(usage_help, default),
    )


def cli_verbose(
    usage_help: str = "Prints the console output.", default: bool = True
) -> callable:
    """Enables --verbose option for cli."""
    return click.option(
        "--verbose",
        type=bool,
        help=add_default_to_usage_help(usage_help, str(default)),
        default=default,
    )


class TIAToolboxCLI(click.Group):
    def __init__(self, *args, **kwargs):
        super(TIAToolboxCLI, self).__init__(*args, **kwargs)
        self.help = "Computational pathology toolbox by TIA Centre."
        self.add_help_option = {"help_option_names": ["-h", "--help"]}


def no_input_message(
    input_file: str or pathlib.Path = None, message: str = "No image input provided.\n"
) -> None:
    """This function is called if no input is provided.

    Args:
        input_file (str or pathlib.Path): Path to input file.
        message (str): Error message to display.

    """
    if input_file is None:
        ctx = click.get_current_context()
        ctx.fail(message=message)


def prepare_file_dir_cli(
    img_input: str or pathlib.Path,
    output_path: str or pathlib.Path,
    file_types: str,
    mode: str,
    sub_dirname: str,
) -> [list, pathlib.Path]:
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
        list: list of file paths to process.
        pathlib.Path: updated output path.

    """
    no_input_message(input_file=img_input)
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

    return [files_all, output_path]


def prepare_model_cli(
    img_input: str or pathlib.Path,
    output_path: str or pathlib.Path,
    masks: str or pathlib.Path,
    file_types: str,
    mode: str,
) -> [list, list, pathlib.Path]:
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
        list: list of file paths to process.
        list: list of masks corresponding to input files.
        pathlib.Path: output path

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

    return [files_all, masks_all, output_path]


tiatoolbox_cli = TIAToolboxCLI()
