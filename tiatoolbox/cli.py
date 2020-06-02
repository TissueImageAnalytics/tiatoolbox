"""Console script for tiatoolbox."""
from tiatoolbox import dataloader
from tiatoolbox.utils import misc_utils as misc
import sys
import click
import os


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main():
    """
    Computational pathology toolbox developed by TIALAB
    """
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
    help="'show' to display meta information only or 'save' to save the meta information, default=show",
)
# @click.option(
#     "--workers",
#     type=int,
#     help="num of cpu cores to use for multiprocessing, default=multiprocessing.cpu_count()",
# )
def slide_info(wsi_input, output_dir, file_types, mode, workers=None):
    """
    Displays or saves WSI metadata
    """
    file_types = tuple(file_types.split(", "))
    if os.path.isdir(wsi_input):
        files_all = misc.grab_files_from_dir(
            input_path=wsi_input, file_types=file_types
        )
    elif os.path.isfile(wsi_input):
        files_all = [
            wsi_input,
        ]
    else:
        raise ValueError("wsi_input path is not valid")

    print(files_all)

    dataloader.slide_info.slide_info(
        input_path=files_all, output_dir=output_dir, mode=mode,
        # workers=workers,
    )


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
