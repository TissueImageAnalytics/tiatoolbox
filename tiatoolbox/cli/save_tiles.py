"""Command line interface for save_tiles."""

import logging

from tiatoolbox import logger
from tiatoolbox.cli.common import (
    cli_file_type,
    cli_img_input,
    cli_output_path,
    cli_tile_format,
    cli_tile_objective,
    cli_tile_read_size,
    cli_verbose,
    prepare_file_dir_cli,
    tiatoolbox_cli,
)


@tiatoolbox_cli.command()
@cli_img_input()
@cli_output_path(
    usage_help="Path to output directory to save the output.",
    default="tiles",
)
@cli_file_type()
@cli_tile_objective()
@cli_tile_read_size()
@cli_tile_format()
@cli_verbose(default=False)
def save_tiles(
    img_input: str,
    output_path: str,
    file_types: str,
    tile_objective_value: int,
    tile_read_size: str,
    tile_format: str,
    *,
    verbose: bool,
) -> None:
    """Display or save WSI metadata."""
    from tiatoolbox.wsicore.wsireader import WSIReader

    files_all, output_path = prepare_file_dir_cli(
        img_input,
        output_path,
        file_types,
        "save",
        "tiles",
    )
    if verbose:
        logger.setLevel(logging.DEBUG)

    for curr_file in files_all:
        wsi = WSIReader.open(curr_file)
        wsi.save_tiles(
            output_dir=output_path,
            tile_objective_value=tile_objective_value,
            tile_read_size=tile_read_size,
            tile_format=tile_format,
        )
