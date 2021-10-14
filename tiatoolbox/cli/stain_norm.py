"""Command line interface for stain_norm."""
import os

import click

from tiatoolbox import utils
from tiatoolbox.tools import stainnorm as sn


@click.group()
def main():
    """Define stain_norm click group."""
    return 0


@main.command()
@click.option(
    "--source_input",
    help="input path to the source image or a directory of source images",
)
@click.option("--target_input", help="input path to the target image")
@click.option(
    "--method",
    help="Stain normalisation method to use. Choose from 'reinhard', 'custom',"
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
    "--output_path",
    help="Output directory for stain normalisation",
    default="stainorm_output",
)
@click.option(
    "--file_types",
    help="file types to capture from directory"
    "default='*.png', '*.jpg', '*.tif', '*.tiff'",
    default="*.png, *.jpg, *.tif, *.tiff",
)
def stain_norm(
    source_input, target_input, method, stain_matrix, output_path, file_types
):
    """Stain normalise an input image/directory of input images."""
    file_types = utils.misc.string_to_tuple(in_str=file_types)

    if not os.path.exists(source_input):
        raise FileNotFoundError

    files_all = [
        source_input,
    ]

    if os.path.isdir(source_input):
        files_all = utils.misc.grab_files_from_dir(
            input_path=source_input, file_types=file_types
        )

    if method not in ["reinhard", "custom", "ruifrok", "macenko", "vahadane"]:
        raise utils.exceptions.MethodNotSupported

    # init stain normalisation method
    norm = sn.get_normaliser(method, stain_matrix)

    # get stain information of target image
    norm.fit(utils.misc.imread(target_input))

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for curr_file in files_all:
        basename = os.path.basename(curr_file)
        # transform source image
        transform = norm.transform(utils.misc.imread(curr_file))
        utils.misc.imwrite(os.path.join(output_path, basename), transform)
