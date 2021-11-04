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
import os
import pathlib

import click
import numpy as np
from PIL import Image

from tiatoolbox import utils, wsicore
from tiatoolbox.tools import tissuemask


@click.group()
def main():  # pragma: no cover
    """Define tissue_mask click group."""
    return 0


@main.command()
@click.option("--img-input", help="Path to WSI file")
@click.option(
    "--output-path",
    help="Path to output file to save the image region in save mode,"
    " default=tissue_mask",
    default="tissue_mask",
)
@click.option(
    "--method",
    help="Tissue masking method to use. Choose from 'Otsu', 'Morphological',"
    " default=Otsu",
    default="Otsu",
)
@click.option(
    "--resolution",
    type=float,
    default=1.25,
    help="resolution to read the image at, default=1.25",
)
@click.option(
    "--units",
    default="power",
    help="resolution units, default=power",
)
@click.option(
    "--kernel-size",
    type=int,
    nargs=2,
    help="kernel size for morphological dilation, default=1, 1",
)
@click.option(
    "--mode",
    default="show",
    help="'show' to display tissue mask or 'save' to save at the output path"
    ", default=show",
)
@click.option(
    "--file-types",
    help="file types to capture from directory, "
    "default='*.svs, *.ndpi, *.jp2, *.png', '*.jpg', '*.tif', '*.tiff'",
    default="*.svs, *.ndpi, *.jp2, *.png, *.jpg, *.tif, *.tiff",
)
def tissue_mask(
    img_input, output_path, method, resolution, units, kernel_size, mode, file_types
):
    """Generate tissue mask for a WSI."""
    file_types = utils.misc.string_to_tuple(in_str=file_types)
    output_path = pathlib.Path(output_path)

    if not os.path.exists(img_input):
        raise FileNotFoundError

    files_all = [
        img_input,
    ]

    if os.path.isdir(img_input):
        files_all = utils.misc.grab_files_from_dir(
            input_path=img_input, file_types=file_types
        )

    if mode == "save" and not output_path.is_dir():
        os.makedirs(output_path)

    if method not in ["Otsu", "Morphological"]:
        raise utils.exceptions.MethodNotSupported

    masker = None

    if method == "Otsu":
        masker = tissuemask.OtsuTissueMasker()

    if method == "Morphological":
        if not kernel_size:
            if units not in ["mpp", "power"]:
                raise utils.exceptions.MethodNotSupported(
                    "Specified units not supported for tissue masking."
                )
            if units == "mpp":
                masker = tissuemask.MorphologicalMasker(mpp=resolution)
            if units == "power":
                masker = tissuemask.MorphologicalMasker(power=resolution)
        else:
            masker = tissuemask.MorphologicalMasker(kernel_size=kernel_size)

    for curr_file in files_all:
        wsi = wsicore.wsireader.get_wsireader(input_img=curr_file)
        wsi_thumb = wsi.slide_thumbnail(resolution=1.25, units="power")
        mask = masker.fit_transform(wsi_thumb[np.newaxis, :])

        if mode == "show":
            im_region = Image.fromarray(mask[0])
            im_region.show()

        if mode == "save":
            utils.misc.imwrite(
                output_path.joinpath(pathlib.Path(curr_file).stem + ".png"),
                mask[0].astype(np.uint8) * 255,
            )
