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

"""Command line interface for stain_norm."""
import os

import click

from tiatoolbox import utils
from tiatoolbox.cli.common import (
    cli_file_type,
    cli_img_input,
    cli_method,
    cli_output_path,
    prepare_file_dir_cli,
    tiatoolbox_cli,
)
from tiatoolbox.tools import stainnorm as sn


@tiatoolbox_cli.command()
@cli_img_input(
    usage_help="Input path to the source image or a directory of source images."
)
@cli_output_path(default="stainnorm_output")
@cli_file_type(default="*.png, *.jpg, *.tif, *.tiff")
@cli_method(
    usage_help="Stain normalization method to use.",
    default="reinhard",
    input_type=click.Choice(
        ["reinhard", "custom", "ruifrok", "macenko", "vahadane"], case_sensitive=False
    ),
)
# inputs specific to this function
@click.option("--target-input", help="Input path to the target image")
@click.option(
    "--stain-matrix",
    help="stain matrix to use in custom normalizer. This can either be a numpy array"
    ", a path to a npy file or a path to a csv file. If using a path to a csv file, "
    "there must not be any column headers.",
    default=None,
)
def stain_norm(img_input, target_input, method, stain_matrix, output_path, file_types):
    """Stain normalize an input image/directory of input images."""
    files_all, output_path = prepare_file_dir_cli(
        img_input, output_path, file_types, "save", "stainnorm_output"
    )

    # init stain normalization method
    norm = sn.get_normalizer(method, stain_matrix)

    # get stain information of target image
    norm.fit(utils.misc.imread(target_input))

    for curr_file in files_all:
        basename = os.path.basename(curr_file)
        # transform source image
        transform = norm.transform(utils.misc.imread(curr_file))
        utils.misc.imwrite(os.path.join(output_path, basename), transform)
