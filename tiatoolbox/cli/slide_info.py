"""Command line interface for slide_info."""
import pathlib

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
    "default=img_input/../meta-data"
)
@cli_file_type(default="*.ndpi, *.svs, *.mrxs, *.jp2")
@cli_mode(default="show")
@cli_verbose(default=True)
def slide_info(img_input, output_path, file_types, mode, verbose):
    """Displays or saves WSI metadata depending on the mode argument."""
    from tiatoolbox import utils, wsicore

    files_all, output_path = prepare_file_dir_cli(
        img_input, output_path, file_types, mode, "meta-data"
    )

    for curr_file in files_all:
        slide_param = wsicore.slide_info.slide_info(
            input_path=curr_file, verbose=verbose or mode == "show"
        )

        if mode == "save":
            out_path = pathlib.Path(
                output_path, slide_param.file_path.with_suffix(".yaml").name
            )
            utils.misc.save_yaml(
                slide_param.as_dict(),
                out_path,
            )
            print("Meta files saved at " + str(output_path))
