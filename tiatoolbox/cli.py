"""Console script for tiatoolbox."""
from tiatoolbox import dataloader
import sys
import click


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main():
    """
    Computational pathology toolbox designed by TIALAB.
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
    help="file types to capture from directory, default=('*.ndpi', '*.svs', '*.mrxs')",
)
@click.option(
    "--mode",
    help="'show' to display meta information only or 'save' to save the meta information, default=show",
)
@click.option(
    "--num_cpu",
    type=int,
    help="num of cpus to use for multiprocessing, default=multiprocessing.cpu_count()",
)
def slide_info(wsi_input, output_dir, file_types, mode, num_cpu):
    """
    Displays or saves WSI metadata
    """
    dataloader.slide_info.slide_info(
        wsi_input=wsi_input,
        output_dir=output_dir,
        file_types=file_types,
        mode=mode,
        num_cpu=num_cpu,
    )


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
