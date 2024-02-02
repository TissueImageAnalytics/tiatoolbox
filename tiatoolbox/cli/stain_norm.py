"""Command line interface for stain_norm."""

from pathlib import Path

import click

from tiatoolbox.cli.common import (
    cli_file_type,
    cli_img_input,
    cli_method,
    cli_output_path,
    prepare_file_dir_cli,
    tiatoolbox_cli,
)

input_type = click.Choice(
    ["reinhard", "custom", "ruifrok", "macenko", "vahadane"],
    case_sensitive=False,
)


@tiatoolbox_cli.command()
@cli_img_input(
    usage_help="Input path to the source image or a directory of source images.",
)
@cli_output_path(default="stainnorm_output")
@cli_file_type(default="*.png, *.jpg, *.tif, *.tiff")
@cli_method(
    usage_help="Stain normalization method to use.",
    default="reinhard",
    input_type=input_type,
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
def stain_norm(
    img_input: str,
    target_input: str,
    method: str,
    stain_matrix: str,
    output_path: str,
    file_types: str,
) -> None:
    """Stain normalize an input image/directory of input images."""
    from tiatoolbox.tools import stainnorm as sn
    from tiatoolbox.utils import imread, imwrite

    files_all, output_path = prepare_file_dir_cli(
        img_input,
        output_path,
        file_types,
        "save",
        "stainnorm_output",
    )

    # init stain normalization method
    norm = sn.get_normalizer(method, stain_matrix)

    # get stain information of target image
    norm.fit(imread(target_input))

    for curr_file in files_all:
        basename = Path(curr_file).name
        # transform source image
        transform = norm.transform(imread(curr_file))
        imwrite(output_path / basename, transform)
