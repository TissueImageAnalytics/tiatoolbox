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

"""Get Slide Meta Data information."""
from tiatoolbox.dataloader import wsireader

import pathlib


def slide_info(input_path, verbose=True):
    """Return WSI meta data.

    Args:
        input_path (str, pathlib.Path): Path to whole slide image
        output_dir (str, pathlib.Path): Path to output directory to save the output
        verbose (bool): Print output, default=True

    Returns:
        WSIMeta: containing meta information

    Examples:
        >>> from tiatoolbox.dataloader.slide_info import slide_info
        >>> from tiatoolbox import utils
        >>> file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
        >>> files_all = utils.misc.grab_files_from_dir(input_path,
        ...     file_types=file_types)
        >>> for curr_file in files_all:
        ...     slide_param = slide_info(input_path=curr_file)
        ...     utils.misc.save_yaml(slide_param.as_dict(),
        ...           slide_param.file_name + ".yaml")
        ...     print(slide_param.as_dict())

    """
    input_path = pathlib.Path(input_path)
    if verbose:
        print(input_path.name, flush=True)

    wsi = wsireader.get_wsireader(input_img=input_path)
    info = wsi.info
    if verbose:
        print(info.as_dict())

    return info
