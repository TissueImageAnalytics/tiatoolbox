"""Command line interface for slide_info."""

import logging
from pathlib import Path

from tiatoolbox import logger
from tiatoolbox.cli.common import (
    cli_file_type,
    cli_img_input,
    cli_mode,
    cli_output_path,
    cli_verbose,
    prepare_file_dir_cli,
    tiatoolbox_cli,
)


@tiatoolbox_cli.command()
@cli_img_input()
@cli_output_path(
    usage_help="Path to output directory to save the output. "
    "default=img_input/../meta-data",
)
@cli_file_type(default="*.ndpi, *.svs, *.mrxs, *.jp2")
@cli_mode(default="show")
@cli_verbose(default=False)
def slide_info(
    img_input: str,
    output_path: str,
    file_types: str,
    mode: str,
    *,
    verbose: bool,
) -> None:
    """Displays or saves WSI metadata depending on the mode argument."""
    from tiatoolbox import utils, wsicore

    all_files, output_path = prepare_file_dir_cli(
        img_input,
        output_path,
        file_types,
        mode,
        "meta-data",
    )

    for file in all_files:
        curr_file = Path(file)
        wsi = wsicore.wsireader.WSIReader.open(input_img=curr_file)

        if verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug(curr_file.name)

        if mode == "show":
            logger.info(wsi.info.as_dict())

        if mode == "save":
            out_path = Path(
                output_path,
                wsi.info.file_path.with_suffix(".yaml").name,
            )
            utils.save_yaml(
                wsi.info.as_dict(),
                out_path,
            )
            logger.info("Meta files saved at %s.", str(output_path))
