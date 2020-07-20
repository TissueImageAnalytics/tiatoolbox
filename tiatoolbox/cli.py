"""Console script for tiatoolbox."""
from tiatoolbox import __version__
from tiatoolbox import dataloader
from tiatoolbox import utils

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
    """Computational pathology toolbox developed by TIA LAB"""
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
    default="*.ndpi, *.svs, *.mrxs",
)
@click.option(
    "--mode",
    default="show",
    help="'show' to display meta information only or 'save' to save "
    "the meta information, default=show",
)
@click.option(
    "--workers",
    type=int,
    help="num of cpu cores to use for multiprocessing, "
    "default=multiprocessing.cpu_count()",
)
def slide_info(wsi_input, output_dir, file_types, mode, workers=None):
    """Displays or saves WSI metadata"""
    file_types = tuple(file_types.split(", "))

    if os.path.isdir(wsi_input):
        files_all = utils.misc.grab_files_from_dir(
            input_path=wsi_input, file_types=file_types
        )
        if output_dir is None and mode == "save":
            input_dir, _, _ = utils.misc.split_path_name_ext(wsi_input)
            output_dir = pathlib.Path(input_dir).joinpath("meta")

    elif os.path.isfile(wsi_input):
        files_all = [
            wsi_input,
        ]
        if output_dir is None and mode == "save":
            input_dir, _, _ = utils.misc.split_path_name_ext(wsi_input)
            output_dir = pathlib.Path(input_dir).joinpath("..").joinpath("meta")
    else:
        raise ValueError("wsi_input path is not valid")

    print(files_all)

    slide_params = dataloader.slide_info.slide_info(
        input_path=files_all, workers=workers,
    )

    if mode == "show":
        for slide_param in slide_params:
            print(slide_param)

    if mode == "save":
        output_dir.mkdir(parents=True, exist_ok=True)
        for slide_param in slide_params:
            utils.misc.save_yaml(
                slide_param,
                pathlib.Path(output_dir).joinpath(slide_param["file_name"] + ".yaml"),
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
    help="image region in the whole slide image to read" "default=0 0 2000 2000",
)
@click.option(
    "--level",
    type=int,
    default=0,
    help="pyramid level to read the image, " "default=0",
)
@click.option(
    "--mode",
    default="show",
    help="'show' to display image region or 'save' to save at the output path"
    ", default=show",
)
def read_region(wsi_input, region, level, output_path, mode):
    """Reads a region in an whole slide image as specified"""
    if all(region):
        region = [0, 0, 2000, 2000]

    input_dir, file_name, ext = utils.misc.split_path_name_ext(full_path=wsi_input)
    if output_path is None and mode == "save":
        output_path = str(pathlib.Path(input_dir).joinpath("../im_region.jpg"))
    wsi_obj = dataloader.wsireader.WSIReader(
        input_dir=input_dir, file_name=file_name + ext
    )
    im_region = wsi_obj.read_region(region[0], region[1], region[2], region[3], level)
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
    """Reads whole slide image thumbnail"""

    input_dir, file_name, ext = utils.misc.split_path_name_ext(full_path=wsi_input)
    if output_path is None and mode == "save":
        output_path = str(pathlib.Path(input_dir).joinpath("../im_region.jpg"))
    wsi_obj = dataloader.wsireader.WSIReader(
        input_dir=input_dir, file_name=file_name + ext
    )

    slide_thumb = wsi_obj.slide_thumbnail()

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
    default="*.ndpi, *.svs, *.mrxs",
)
@click.option(
    "--tile_objective_value",
    type=int,
    default=20,
    help="objective value at which tile is generated, " "default=20",
)
@click.option(
    "--tile_read_size_w", type=int, default=5000, help="tile width, " "default=5000",
)
@click.option(
    "--tile_read_size_h", type=int, default=5000, help="tile height, " "default=5000",
)
@click.option(
    "--workers",
    type=int,
    help="num of cpu cores to use for multiprocessing, "
    "default=multiprocessing.cpu_count()",
)
def save_tiles(
    wsi_input,
    output_dir,
    file_types,
    tile_objective_value,
    tile_read_size_w,
    tile_read_size_h,
    workers=None,
):
    """Displays or saves WSI metadata"""
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
        raise ValueError("wsi_input path is not valid")

    print(files_all)

    dataloader.save_tiles.save_tiles(
        input_path=files_all,
        output_dir=output_dir,
        tile_objective_value=tile_objective_value,
        tile_read_size_w=tile_read_size_w,
        tile_read_size_h=tile_read_size_h,
        workers=workers,
    )


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
