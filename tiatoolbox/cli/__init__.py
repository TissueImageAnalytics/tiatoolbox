# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Console script for tiatoolbox."""
import os
import sys

import click

from tiatoolbox import __version__
from tiatoolbox.cli.nucleus_instance_segment import nucleus_instance_segment
from tiatoolbox.cli.patch_predictor import patch_predictor
from tiatoolbox.cli.read_bounds import read_bounds
from tiatoolbox.cli.save_tiles import save_tiles
from tiatoolbox.cli.semantic_segment import semantic_segment
from tiatoolbox.cli.slide_info import slide_info
from tiatoolbox.cli.slide_thumbnail import slide_thumbnail
from tiatoolbox.cli.stain_norm import stain_norm
from tiatoolbox.cli.tissue_mask import tissue_mask


def version_msg():
    """Return a string with tiatoolbox package version and python version."""
    python_version = sys.version[:3]
    location = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    message = "tiatoolbox %(version)s from {} (Python {})"
    return message.format(location, python_version)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(
    __version__, "--version", "-V", help="Version", message=version_msg()
)
def main():
    """Computational pathology toolbox by TIA Centre."""
    return 0


main.add_command(nucleus_instance_segment)
main.add_command(patch_predictor)
main.add_command(read_bounds)
main.add_command(save_tiles)
main.add_command(semantic_segment)
main.add_command(slide_info)
main.add_command(slide_thumbnail)
main.add_command(tissue_mask)
main.add_command(stain_norm)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
