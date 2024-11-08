"""Console script for tiatoolbox."""

import sys
from platform import platform, python_version

import click

from tiatoolbox import __version__
from tiatoolbox.cli.common import tiatoolbox_cli
from tiatoolbox.cli.nucleus_instance_segment import nucleus_instance_segment
from tiatoolbox.cli.patch_classifier import patch_classifier
from tiatoolbox.cli.read_bounds import read_bounds
from tiatoolbox.cli.save_tiles import save_tiles
from tiatoolbox.cli.semantic_segment import semantic_segment
from tiatoolbox.cli.show_wsi import show_wsi
from tiatoolbox.cli.slide_info import slide_info
from tiatoolbox.cli.slide_thumbnail import slide_thumbnail
from tiatoolbox.cli.stain_norm import stain_norm
from tiatoolbox.cli.tissue_mask import tissue_mask
from tiatoolbox.cli.visualize import visualize


def version_msg() -> str:
    """Return a string with tiatoolbox package version and python version."""
    return f"tiatoolbox {__version__} (Python {python_version()}) on {platform()}."


@tiatoolbox_cli.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(
    __version__,
    "--version",
    "-v",
    help="Show the tiatoolbox version",
    message=version_msg(),
)
def main() -> click.BaseCommand:
    """Computational pathology toolbox by TIA Centre."""
    return 0


main.add_command(nucleus_instance_segment)
main.add_command(patch_classifier)
main.add_command(read_bounds)
main.add_command(save_tiles)
main.add_command(semantic_segment)
main.add_command(slide_info)
main.add_command(slide_thumbnail)
main.add_command(tissue_mask)
main.add_command(stain_norm)
main.add_command(visualize)
main.add_command(show_wsi)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
