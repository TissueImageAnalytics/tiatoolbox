"""Command line interface for slide_info."""
import os
import pathlib
import sys

import click

from tiatoolbox import utils, wsicore


@click.group()
def main():
    """Define slide_info click group."""
    return 0  # pragma: no cover


@main.command()
@click.option("--img_input", help="input path to WSI file or directory path")
@click.option(
    "--output_dir",
    help="Path to output directory to save the output, default=img_input/../meta",
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
def slide_info(img_input, output_dir, file_types, mode, verbose):
    """Display or save WSI metadata."""
    file_types = utils.misc.string_to_tuple(in_str=file_types)

    if isinstance(output_dir, str):
        output_dir = pathlib.Path(output_dir)

    if not os.path.exists(img_input):
        raise FileNotFoundError

    files_all = [
        img_input,
    ]

    if os.path.isdir(img_input):
        files_all = utils.misc.grab_files_from_dir(
            input_path=img_input, file_types=file_types
        )

    if output_dir is None and mode == "save":
        input_dir = pathlib.Path(img_input).parent
        output_dir = input_dir / "meta"

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


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
