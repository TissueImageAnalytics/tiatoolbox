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

"""Command line interface for tissue_mask."""
import pathlib

import click
import numpy as np
from PIL import Image

from tiatoolbox import utils
from tiatoolbox.cli.common import (
    cli_file_type,
    cli_img_input,
    cli_method,
    cli_mode,
    cli_output_path,
    cli_resolution,
    cli_units,
    prepare_file_dir_cli,
    tiatoolbox_cli,
)
from tiatoolbox.tools import tissuemask
from tiatoolbox.wsicore.wsireader import WSIReader


def get_masker(method, kernel_size, units, resolution):
    """Get Tissue Masker."""

    if method not in ["Otsu", "Morphological"]:
        raise utils.exceptions.MethodNotSupported

    if method == "Otsu":
        return tissuemask.OtsuTissueMasker()

    if kernel_size:
        return tissuemask.MorphologicalMasker(kernel_size=kernel_size)

    if units == "mpp":
        return tissuemask.MorphologicalMasker(mpp=resolution)

    if units == "power":
        return tissuemask.MorphologicalMasker(power=resolution)

    raise utils.exceptions.MethodNotSupported(
        "Specified units not supported for tissue masking."
    )


@tiatoolbox_cli.command()
@cli_img_input()
@cli_output_path(default="tissue_mask")
@cli_method(default="Otsu")
@cli_resolution(default=1.25)
@cli_units(default="power")
@cli_mode(default="show")
@cli_file_type(default="*.svs, *.ndpi, *.jp2, *.png, *.jpg, *.tif, *.tiff")
# specific to this function
@click.option(
    "--kernel-size",
    type=int,
    nargs=2,
    help="kernel size for morphological dilation, default=1, 1",
)
def tissue_mask(
    img_input, output_path, method, resolution, units, kernel_size, mode, file_types
):
    """Generate tissue mask for a WSI."""
    files_all, output_path = prepare_file_dir_cli(
        img_input, output_path, file_types, mode, "meta-data"
    )

    masker = get_masker(method, kernel_size, units, resolution)

    for curr_file in files_all:
        wsi = WSIReader.open(input_img=curr_file)
        wsi_thumb = wsi.slide_thumbnail(resolution=1.25, units="power")
        mask = masker.fit_transform(wsi_thumb[np.newaxis, :])

        if mode == "show":  # pragma: no cover
            im_region = Image.fromarray(mask[0])
            im_region.show()

        if mode == "save":
            utils.misc.imwrite(
                output_path.joinpath(pathlib.Path(curr_file).stem + ".png"),
                mask[0].astype(np.uint8) * 255,
            )
