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

"""Get Slide Meta Data information"""
from tiatoolbox.dataloader import wsireader
from tiatoolbox.decorators.multiproc import TIAMultiProcess

import os


@TIAMultiProcess(iter_on="input_path")
def slide_info(input_path, output_dir=None):
    """Single file run to output or save WSI meta data.

    Multiprocessing uses this function to run slide_info in parallel

    Args:
        input_path (str): Path to whole slide image
        output_dir (str): Path to output directory to save the output
        workers (int): num of cpu cores to use for multiprocessing
    Returns:
        list: list of dictionary Whole Slide meta information

    Examples:
        >>> from tiatoolbox.dataloader.slide_info import slide_info
        >>> from tiatoolbox import utils
        >>> file_types = ("*.ndpi", "*.svs", "*.mrxs")
        >>> files_all = utils.misc.grab_files_from_dir(input_path,
        ...     file_types=file_types)
        >>> slide_params = slide_info(input_path=files_all, workers=2)
        >>> for slide_param in slide_params:
        ...        utils.misc.save_yaml(slide_param,
        ...             slide_param["file_name"] + ".yaml")
        ...        print(slide_param)

    """

    input_dir, file_name = os.path.split(input_path)

    print(file_name, flush=True)
    _, file_type = os.path.splitext(file_name)

    if file_type in (".svs", ".ndpi", ".mrxs"):
        wsi_reader = wsireader.WSIReader(
            input_dir=input_dir, file_name=file_name, output_dir=output_dir
        )
        info = wsi_reader.slide_info()
    else:
        print("File type not supported")
        info = None

    return info
