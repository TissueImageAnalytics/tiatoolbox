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
# The Original Code is Copyright (C) 2020, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Console script for tiatoolbox."""
from tiatoolbox import __version__
from tiatoolbox import wsicore
from tiatoolbox.tools import stainnorm as sn
from tiatoolbox import utils
from tiatoolbox.utils.exceptions import MethodNotSupported
from tiatoolbox.models.classification.pretrained_info import __pretrained_model
from tiatoolbox.models.classification.patch_predictor import CNN_Patch_Predictor
from tiatoolbox.models.dataset.classification import Patch_Dataset

import sys
import click
import os
import pathlib
from PIL import Image


def version_msg():
    """Return a string with tiatoolbox package version and python version."""
    python_version = sys.version[:3]
    location = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    message = "tiatoolbox %(version)s from {} (Python {})"
    return message.format(location, python_version)


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(
    __version__, "--version", "-V", help="Version", message=version_msg()
)
def main():
    """Computational pathology toolbox by TIA LAB."""
    return 0


@main.command()
@click.option("--wsi_input", help="input path to WSI file or directory path")
@click.option(
    "--output_dir",
    help="Path to output directory to save the output, default=wsi_input/../meta",
)
@click.option(
    "--file_types",
    help="file types to capture from directory, default='*.ndpi', '*.svs', '*.mrxs'",
    default="*.ndpi, *.svs, *.mrxs, *.jp2",
)
@click.option(
    "--mode",
    default="show",
    help="'show' to display meta information only or 'save' to save "
    "the meta information, default=show",
)
@click.option(
    "--verbose",
    type=bool,
    default=True,
    help="Print output, default=True",
)
def slide_info(wsi_input, output_dir, file_types, mode, verbose):
    """Display or save WSI metadata."""
    file_types = tuple(file_types.split(", "))

    if isinstance(output_dir, str):
        output_dir = pathlib.Path(output_dir)

    if os.path.isdir(wsi_input):
        files_all = utils.misc.grab_files_from_dir(
            input_path=wsi_input, file_types=file_types
        )
        if output_dir is None and mode == "save":
            input_dir = pathlib.Path(wsi_input).parent
            output_dir = input_dir / "meta"

    elif os.path.isfile(wsi_input):
        files_all = [
            wsi_input,
        ]
        if output_dir is None and mode == "save":
            input_dir = pathlib.Path(wsi_input).parent
            output_dir = input_dir.parent / "meta"
    else:
        raise FileNotFoundError

    print(files_all)

    if mode == "save":
        output_dir.mkdir(parents=True, exist_ok=True)

    for curr_file in files_all:
        slide_param = wsicore.slide_info.slide_info(
            input_path=curr_file, verbose=verbose
        )
        if mode == "show":
            print(slide_param.as_dict())

        if mode == "save":
            out_path = pathlib.Path(
                output_dir, slide_param.file_path.with_suffix(".yaml").name
            )
            utils.misc.save_yaml(
                slide_param.as_dict(),
                out_path,
            )
            print("Meta files saved at " + str(output_dir))


@main.command()
@click.option("--wsi_input", help="Path to WSI file")
@click.option(
    "--output_path",
    help="Path to output file to save the image region in save mode,"
    " default=wsi_input_dir/../im_region",
)
@click.option(
    "--region",
    type=int,
    nargs=4,
    help="image region in the whole slide image to read, default=0 0 2000 2000",
)
@click.option(
    "--resolution",
    type=float,
    default=0,
    help="resolution to read the image at, default=0",
)
@click.option(
    "--units",
    default="level",
    type=click.Choice(["mpp", "power", "level", "baseline"], case_sensitive=False),
    help="resolution units, default=level",
)
@click.option(
    "--mode",
    default="show",
    help="'show' to display image region or 'save' to save at the output path"
    ", default=show",
)
def read_bounds(wsi_input, region, resolution, units, output_path, mode):
    """Read a region in an whole slide image as specified."""
    if not region:
        region = [0, 0, 2000, 2000]

    if output_path is None and mode == "save":
        input_dir = pathlib.Path(wsi_input).parent
        output_path = str(input_dir.parent / "im_region.jpg")

    wsi = wsicore.wsireader.get_wsireader(input_img=wsi_input)

    im_region = wsi.read_bounds(
        region,
        resolution=resolution,
        units=units,
    )
    if mode == "show":
        im_region = Image.fromarray(im_region)
        im_region.show()

    if mode == "save":
        utils.misc.imwrite(output_path, im_region)


@main.command()
@click.option("--wsi_input", help="Path to WSI file")
@click.option(
    "--output_path",
    help="Path to output file to save the image region in save mode,"
    " default=wsi_input_dir/../im_region",
)
@click.option(
    "--mode",
    default="show",
    help="'show' to display image region or 'save' to save at the output path"
    ", default=show",
)
def slide_thumbnail(wsi_input, output_path, mode):
    """Read whole slide image thumbnail."""
    if output_path is None and mode == "save":
        input_dir = pathlib.Path(wsi_input).parent
        output_path = str(input_dir.parent / "slide_thumb.jpg")

    wsi = wsicore.wsireader.get_wsireader(input_img=wsi_input)

    slide_thumb = wsi.slide_thumbnail()

    if mode == "show":
        im_region = Image.fromarray(slide_thumb)
        im_region.show()

    if mode == "save":
        utils.misc.imwrite(output_path, slide_thumb)


@main.command()
@click.option("--wsi_input", help="input path to WSI file or directory path")
@click.option(
    "--output_dir",
    default="tiles",
    help="Path to output directory to save the output, default=tiles",
)
@click.option(
    "--file_types",
    help="file types to capture from directory, default='*.ndpi', '*.svs', '*.mrxs'",
    default="*.ndpi, *.svs, *.mrxs, *.jp2",
)
@click.option(
    "--tile_objective_value",
    type=int,
    default=20,
    help="objective value at which tile is generated- default=20",
)
@click.option(
    "--tile_read_size",
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
    wsi_input,
    output_dir,
    file_types,
    tile_objective_value,
    tile_read_size,
    verbose=True,
):
    """Display or save WSI metadata."""
    file_types = tuple(file_types.split(", "))
    if os.path.isdir(wsi_input):
        files_all = utils.misc.grab_files_from_dir(
            input_path=wsi_input, file_types=file_types
        )
    elif os.path.isfile(wsi_input):
        files_all = [
            wsi_input,
        ]
    else:
        raise FileNotFoundError

    print(files_all)

    for curr_file in files_all:
        wsicore.save_tiles.save_tiles(
            input_path=curr_file,
            output_dir=output_dir,
            tile_objective_value=tile_objective_value,
            tile_read_size=tile_read_size,
            verbose=verbose,
        )


@main.command()
@click.option(
    "--source_input",
    help="input path to the source image or a directory of source images",
)
@click.option("--target_input", help="input path to the target image")
@click.option(
    "--method",
    help="Stain normlisation method to use. Choose from 'reinhard', 'custom',"
    "'ruifrok', 'macenko, 'vahadane'",
    default="reinhard",
)
@click.option(
    "--stain_matrix",
    help="stain matrix to use in custom normaliser. This can either be a numpy array"
    ", a path to a npy file or a path to a csv file. If using a path to a csv file, "
    "there must not be any column headers.",
    default=None,
)
@click.option(
    "--output_dir",
    help="output directory for stain normalisation",
    default="stainorm_output",
)
@click.option(
    "--file_types",
    help="file types to capture from directory"
    "default='*.png', '*.jpg', '*.tif', '*.tiff'",
    default="*.png, *.jpg, *.tif, *.tiff",
)
def stainnorm(source_input, target_input, method, stain_matrix, output_dir, file_types):
    """Stain normalise an input image/directory of input images."""
    file_types = tuple(file_types.split(", "))
    if os.path.isdir(source_input):
        files_all = utils.misc.grab_files_from_dir(
            input_path=source_input, file_types=file_types
        )
    elif os.path.isfile(source_input):
        files_all = [
            source_input,
        ]
    else:
        raise FileNotFoundError

    if method not in ["reinhard", "custom", "ruifrok", "macenko", "vahadane"]:
        raise MethodNotSupported

    # init stain normalisation method
    norm = sn.get_normaliser(method, stain_matrix)

    # get stain information of target image
    norm.fit(utils.misc.imread(target_input))

    for curr_file in files_all:
        basename = os.path.basename(curr_file)
        # transform source image
        transform = norm.transform(utils.misc.imread(curr_file))
        utils.misc.imwrite(os.path.join(output_dir, basename), transform)


@main.command()
@click.option(
    "--predefined_model",
    help="predefined model used to process the data. the format is "
    "<model_name>_<dataset_trained_on>. For example, `resnet18_kather` "
    "is a resnet18 model trained on the kather dataset.",
    default="resnet18_kather",
)
@click.option(
    "--predefined_weight",
    help="path to the model weight file. If not supplied, the default "
    "pretrained weight will be used.",
    default=None,
)
@click.option(
    "--input_dir", help="path to the input directory containing images to process"
)
@click.option(
    "--output_dir", help="output directory where model predictions will be saved."
)
@click.option(
    "--batch_size",
    help="number of images to feed into the model each time.",
    default=16,
)
@click.option(
    "--return_probs",
    help="whether to return raw model probabilities.",
    default=False,
)
@click.option(
    "--file_types",
    help="file types to capture from directory"
    "default='*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'",
    default="*.png, *.jpg, *.jpeg, *.tif, *.tiff",
)
def patch_predictor(
    predefined_model,
    pretrained_weight,
    data_type,
    input_dir,
    output_dir,
    batch_size,
    file_types,
    return_probs,
):
    """Process an image/directory of input images with a patch classification CNN."""
    file_types = tuple(file_types.split(", "))
    if os.path.isdir(input_dir):
        img_files = utils.misc.grab_files_from_dir(
            input_path=input_dir, file_types=file_types
        )
    elif os.path.isfile(input_dir):
        img_files = [
            input_dir,
        ]
    else:
        raise FileNotFoundError

    if predefined_model not in __pretrained_model:
        raise ValueError("Predefined model `%s` does not exist." % predefined_model)

    dataset = Patch_Dataset(img_files)

    predictor = CNN_Patch_Predictor(
        predefined_model=predefined_model,
        pretrained_weight=pretrained_weight,
        batch_size=batch_size,
    )

    output = predictor.predict(dataset, return_probs=return_probs)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
