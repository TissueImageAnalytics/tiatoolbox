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
# The Original Code is Copyright (C) 2020, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Save image tiles from the whole slide image."""
from tiatoolbox.dataloader import wsireader

import pathlib


def save_tiles(
    input_path,
    output_dir="tiles",
    tile_objective_value=20,
    tile_read_size=(5000, 5000),
    verbose=True,
):
    """Save image tiles for whole slide image.
    Default file format for tiles is jpg.

    Args:
        input_path (str): Path to whole slide image
        output_dir (str): Path to output directory to save the output
        tile_objective_value (int): objective value at which tile is generated,
                default=20
        tile_read_size (tuple of int): Tile (width, height), default=(5000, 5000).
        verbose (bool): Print output, default=True

    Returns:

    Examples:
        >>> from tiatoolbox.dataloader.save_tiles import save_tiles
        >>> from tiatoolbox.utils import misc
        >>> file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
        >>> files_all = misc.grab_files_from_dir(input_path,
        ...     file_types=file_types)
        >>> for curr_file in files_all:
        ...     save_tiles(input_path=curr_file,
        ...         output_dir="tiles",
        ...         tile_objective_value=10,
        ...         tile_read_size_w=5000,
        ...         tile_read_size_h=5000
        ...         )

    """
    input_path = pathlib.Path(input_path)
    if verbose:
        print(input_path.name, flush=True)

    wsi = wsireader.get_wsireader(input_img=input_path)
    wsi.save_tiles(
        output_dir=output_dir,
        tile_objective_value=tile_objective_value,
        tile_read_size=tile_read_size,
        verbose=verbose,
    )
