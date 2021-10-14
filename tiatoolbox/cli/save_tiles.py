"""Command line interface for save_tiles."""
import os

import click

from tiatoolbox import utils, wsicore


@click.group()
def main():
    """Define save_tiles click group."""
    return 0


@main.command()
@click.option("--img_input", help="input path to WSI file or directory path")
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
    img_input,
    output_dir,
    file_types,
    tile_objective_value,
    tile_read_size,
    verbose=True,
):
    """Display or save WSI metadata."""
    file_types = utils.misc.string_to_tuple(file_types=file_types)

    if not os.path.exists(img_input):
        raise FileNotFoundError

    files_all = [
        img_input,
    ]

    if os.path.isdir(img_input):
        files_all = utils.misc.grab_files_from_dir(
            input_path=img_input, file_types=file_types
        )

    print(files_all)

    for curr_file in files_all:
        wsicore.save_tiles.save_tiles(
            input_path=curr_file,
            output_dir=output_dir,
            tile_objective_value=tile_objective_value,
            tile_read_size=tile_read_size,
            verbose=verbose,
        )
