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

"""Miscellaneous small functions repeatedly used in tiatoolbox."""
import cv2
import pathlib
import yaml
import pandas as pd
import numpy as np


def split_path_name_ext(full_path):
    """Split path of a file to directory path, file name and extension.

    Args:
        full_path (str): Path to a file

    Returns:
        tuple: Three sections of the input file path
        (input directory path, file name, file extension)

    Examples:
        >>> from tiatoolbox import utils
        >>> dir_path, file_name, extension =
        ...     utils.misc.split_path_name_ext(full_path)

    """
    input_path = pathlib.Path(full_path)
    return input_path.parent.absolute(), input_path.name, input_path.suffix


def grab_files_from_dir(input_path, file_types=("*.jpg", "*.png", "*.tif")):
    """Grab file paths specified by file extensions.

    Args:
        input_path (str, pathlib.Path): Path to the directory where files
            need to be searched
        file_types (str, tuple): File types (extensions) to be searched

    Returns:
        list: File paths as a python list

    Examples:
        >>> from tiatoolbox import utils
        >>> file_types = ("*.ndpi", "*.svs", "*.mrxs")
        >>> files_all = utils.misc.grab_files_from_dir(input_path,
        ...     file_types=file_types,)

    """
    input_path = pathlib.Path(input_path)

    if type(file_types) is str:
        if len(file_types.split(",")) > 1:
            file_types = tuple(file_types.split(","))
        else:
            file_types = (file_types,)

    files_grabbed = []
    for files in file_types:
        files_grabbed.extend(input_path.glob(files))

    return list(files_grabbed)


def save_yaml(input_dict, output_path="output.yaml"):
    """Save dictionary as yaml.
    Args:
        input_dict (dict): A variable of type 'dict'
        output_path (str, pathlib.Path): Path to save the output file

    Returns:

    Examples:
        >>> from tiatoolbox import utils
        >>> input_dict = {'hello': 'Hello World!'}
        >>> utils.misc.save_yaml(input_dict, './hello.yaml')


    """
    with open(str(pathlib.Path(output_path)), "w") as yaml_file:
        yaml.dump(input_dict, yaml_file)


def imwrite(image_path, img):
    """Write numpy array to an image.

    Args:
        image_path (str, pathlib.Path): file path (including extension)
            to save image
        img (ndarray): image array of dtype uint8, MxNx3

    Returns:

    Examples:
        >>> from tiatoolbox import utils
        >>> import numpy as np
        >>> utils.misc.imwrite('BlankImage.jpg',
        ...     np.ones([100, 100, 3]).astype('uint8')*255)

    """
    if isinstance(image_path, pathlib.Path):
        image_path = str(image_path)
    cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def imread(image_path):
    """Read an image as numpy array.

    Args:
        image_path (str, pathlib.Path): file path (including extension) to read image

    Returns:
        img (ndarray): image array of dtype uint8, MxNx3

    Examples:
        >>> from tiatoolbox import utils
        >>> image = utils.misc.imread('ImagePath.jpg')

    """
    if isinstance(image_path, pathlib.Path):
        image_path = str(image_path)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return image.astype("uint8")


def load_stain_matrix(stain_matrix_input):
    """Load a stain matrix as a numpy array.

    Args:
        stain_matrix_input (ndarray or str, pathlib.Path): either a 2x3 / 3x3
            numpy array or a path to a saved .npy / .csv file. If using a .csv file,
            there should be no column headers provided

    Returns:
        stain_matrix (ndarray): the loaded stain matrix

    Examples:
        >>> from tiatoolbox import utils
        >>> stain_matrix = utils.misc.load_stain_matrix(stain_matrix_input)

    """
    if isinstance(stain_matrix_input, str):
        _, __, ext = split_path_name_ext(stain_matrix_input)
        if ext == "csv":
            stain_matrix = np.array(pd.read_csv(stain_matrix_input, header=None))
        elif ext == "npy":
            stain_matrix = np.load(stain_matrix_input)
        else:
            raise Exception(
                "If supplying a path to a stain matrix, use either a \
                npy or a csv file"
            )
    elif isinstance(stain_matrix_input, np.ndarray):
        stain_matrix = stain_matrix_input
    else:
        raise Exception("stain_matrix must be either a path or a numpy array")

    return stain_matrix


def get_luminosity_tissue_mask(img, threshold):
    """Get tissue mask based on the luminosity of the input image.

    Args:
        img (ndarray): input image used to obtain tissue mask.
        threshold (float): luminosity threshold used to determine tissue area.

    Returns:
        tissue_mask (ndarray): binary tissue mask.

    Examples:
        >>> from tiatoolbox import utils
        >>> tissue_mask = utils.misc.get_luminosity_tissue_mask(img, threshold=0.8)

    """
    img = img.astype("uint8")  # ensure input image is uint8
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_lab = img_lab[:, :, 0] / 255.0  # Convert to range [0,1].
    tissue_mask = l_lab < threshold

    # check it's not empty
    if tissue_mask.sum() == 0:
        raise Exception("Empty tissue mask computed")

    return tissue_mask
