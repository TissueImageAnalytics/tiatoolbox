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

"""Get Slide Meta Data information"""
from tiatoolbox.dataloader import wsireader
from tiatoolbox.utils.exceptions import FileNotSupported

import os


def slide_info(input_path, output_dir=None, verbose=True):
    """Returns WSI meta data. Multiprocessing decorator runs this function in parallel.

    Args:
        input_path (str): Path to whole slide image
        output_dir (str): Path to output directory to save the output
        verbose (bool): Print output, default=True

    Returns:
        list: list of dictionary Whole Slide meta information

    Examples:
        >>> from tiatoolbox.dataloader.slide_info import slide_info
        >>> from tiatoolbox import utils
        >>> file_types = ("*.ndpi", "*.svs", "*.mrxs")
        >>> files_all = utils.misc.grab_files_from_dir(input_path,
        ...     file_types=file_types)
        >>> for curr_file in files_all:
        ...     slide_param = slide_info(input_path=curr_file)
        ...     utils.misc.save_yaml(slide_param.as_dict(),
        ...           slide_param.file_name + ".yaml")
        ...     print(slide_param.as_dict())

    """

    input_dir, file_name = os.path.split(input_path)

    if verbose:
        print(file_name, flush=True)
    _, file_type = os.path.splitext(file_name)

    if file_type in (".svs", ".ndpi", ".mrxs"):
        wsi_reader = wsireader.OpenSlideWSIReader(
            input_dir=input_dir, file_name=file_name, output_dir=output_dir
        )
        info = wsi_reader.slide_info
        if verbose:
            print(info.as_dict())
    else:
        raise FileNotSupported(file_type + " file format is not supported.")

    return info
