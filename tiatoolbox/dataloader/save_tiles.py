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
# The Original Code is Copyright (C) 2006, Blender Foundation
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Save image tiles from the whole slide image"""
from tiatoolbox.dataloader import wsireader
from tiatoolbox.decorators.multiproc import TIAMultiProcess
from tiatoolbox.utils import misc
from tiatoolbox.utils.exceptions import FileNotSupported


@TIAMultiProcess(iter_on="input_path")
def save_tiles(
    input_path,
    output_dir="tiles",
    tile_objective_value=20,
    tile_read_size_w=5000,
    tile_read_size_h=5000,
    verbose=True,
):
    """Save image tiles for whole slide image.
    Default file format for tiles is jpg.
    Multiprocessing decorator runs this function in parallel using
    the number of specified cpu cores.

    Args:
        input_path (str): Path to whole slide image
        output_dir (str): Path to output directory to save the output
        tile_objective_value (int): objective value at which tile is generated,
                default=20
        tile_read_size_w (int): tile width, default=5000
        tile_read_size_h (int): tile height, default=5000
        verbose (bool): Print output, default=True
        workers (int): num of cpu cores to use for multiprocessing

    Returns:

    Examples:
        >>> from tiatoolbox.dataloader.save_tiles import save_tiles
        >>> from tiatoolbox.utils import misc
        >>> file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
        >>> files_all = misc.grab_files_from_dir(input_path,
        ...     file_types=file_types)
        >>> save_tiles(input_path=files_all,
        ...     output_dir="tiles",
        ...     tile_objective_value=10,
        ...     tile_read_size_w=5000,
        ...     tile_read_size_h=5000
        ...     )

    """

    input_dir, file_name, ext = misc.split_path_name_ext(input_path)
    if verbose:
        print(file_name + ext, flush=True)

    if ext in (".svs", ".ndpi", ".mrxs"):
        wsi_reader = wsireader.OpenSlideWSIReader(
            input_dir=input_dir,
            file_name=file_name + ext,
            output_dir=output_dir,
            tile_objective_value=tile_objective_value,
            tile_read_size_w=tile_read_size_w,
            tile_read_size_h=tile_read_size_h,
        )
        wsi_reader.save_tiles(verbose=verbose)
    elif ext in (".jp2",):
        wsi_reader = wsireader.OmnyxJP2WSIReader(
            input_dir=input_dir,
            file_name=file_name + ext,
            output_dir=output_dir,
            tile_objective_value=tile_objective_value,
            tile_read_size_w=tile_read_size_w,
            tile_read_size_h=tile_read_size_h,
        )
        wsi_reader.save_tiles(verbose=verbose)
    else:
        raise FileNotSupported
