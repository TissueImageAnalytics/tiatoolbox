"""Command line interface for save_tiles."""
from tiatoolbox.cli.common import (
    cli_file_type,
    cli_img_input,
    cli_output_path,
    cli_tile_objective,
    cli_tile_read_size,
    cli_verbose,
    prepare_file_dir_cli,
    tiatoolbox_cli,
)


@tiatoolbox_cli.command()
@cli_img_input()
@cli_output_path(
    usage_help="Path to output directory to save the output.", default="tiles"
)
@cli_file_type()
@cli_tile_objective()
@cli_tile_read_size()
@cli_verbose()
def save_tiles(
    img_input,
    output_path,
    file_types,
    tile_objective_value,
    tile_read_size,
    verbose=True,
):
    """Display or save WSI metadata."""
    from tiatoolbox import wsicore

    files_all, output_path = prepare_file_dir_cli(
        img_input, output_path, file_types, "save", "tiles"
    )

    for curr_file in files_all:
        wsicore.save_tiles.save_tiles(
            input_path=curr_file,
            output_dir=output_path,
            tile_objective_value=tile_objective_value,
            tile_read_size=tile_read_size,
            verbose=verbose,
        )
