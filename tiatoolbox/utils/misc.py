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
from skimage import exposure


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
    img = contrast_enhancer(img, low_p=2, high_p=98)  # Contrast  enhancement
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_lab = img_lab[:, :, 0] / 255.0  # Convert to range [0,1].
    tissue_mask = l_lab < threshold

    # check it's not empty
    if tissue_mask.sum() == 0:
        raise Exception("Empty tissue mask computed")

    return tissue_mask


def mpp2objective_power(mpp):
    """Approximate objective power from mpp.

    Ranges for approximation::

            mpp < 0.10 -> 100x
    0.10 <= mpp < 0.15 -> 60x
    0.15 <= mpp < 0.30 -> 40x
    0.30 <= mpp < 0.60 -> 20x
    0.60 <= mpp < 1.20 -> 10x
    1.20 <= mpp < 2.40 -> 5x
    2.40 <= mpp < 4.80 -> 2.5x
    4.80 <= mpp < 9.60 -> 1.25x
    9.60 <= mpp -> ValueError

    Args:
        mpp (float or tuple of float): Microns per-pixel.

    Returns:
        float: Objective power approximation.

    Raises:
        ValueError
    """
    mpp = np.mean(mpp)
    if mpp < 0.10:
        return 100
    if mpp < 0.15:
        return 60
    if mpp < 9.60:
        # Double the objective power as mpp halves
        return 10 * 2 ** (np.ceil(np.log2(0.15 / mpp)) + 2)
    raise ValueError()


def image_luminosity_standardiser(img, percentile=0.95):
    """Standardize (adjust) the luminosity component of the input RGB image.

    Args:
        img (ndarray): input image used to obtain tissue mask.
            Image should be RGB uint8.
        percentile (int): Percentile for luminosity saturation.
            At least (100 - percentile)% of pixels should be
            fully luminous (white).

    Returns:
        img (ndarray): Image uint8 RGB with standardized brightness.

    Examples:
        >>> from tiatoolbox import utils
        >>> img = utils.misc.image_luminosity_standardiser(img, percentile=0.95)

    """
    assert img.dtype == np.uint8, "Image should be RGB uint8."
    img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L_float = img_LAB[:, :, 0].astype(float)
    p = np.percentile(L_float, percentile)
    img_LAB[:, :, 0] = np.clip(255 * L_float / p, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2RGB)
    return img


def luminosity_standardiser(lum, percentile=0.95):
    """Standardize (adjust) the luminosity component.

    Args:
        lum (ndarray): input Luminosity component of an image used to
            obtain tissue mask. lum must be in range of 0-255.
        percentile (int): Percentile for luminosity saturation. At least
        (100 - percentile)% of pixels should be fully luminous (white).

    Returns:
        lum (ndarray): Luminosity component with standardized brightness.

    Examples:
        >>> from tiatoolbox import utils
        >>> lum_out = utils.misc.luminosity_standardiser(lum_in, percentile=0.95)

    """
    l_float = lum.astype(float)
    perc = np.percentile(l_float, percentile)
    lum = np.clip(255 * l_float / perc, 0, 255).astype(np.uint8)
    return lum


def contrast_enhancer(img, low_p=2, high_p=98):
    """Enhancing contrast of the input RGB image using intensity adjustment through
        image low and high percentiles.

    Args:
        img (ndarray): input image used to obtain tissue mask.
            Image should be RGB uint8.
        low_p (scalar): low percentile of image values to be saturated to 0.
        high_p (scalar): high percentile of image values to be saturated to 255.
            high_p should always be greater than low_p.

    Returns:
        img (ndarray): Image uint8 RGB with contrast enhanced.

    Examples:
        >>> from tiatoolbox import utils
        >>> img = utils.misc.contrast_enhancer(img, low_p=2, high_p=98)

    """
    assert img.dtype == np.uint8, "Image should be RGB uint8."
    img_out = img.copy()
    p_low, p_high = np.percentile(img_out, (low_p, high_p))
    if p_low >= p_high:
        p_low, p_high = np.min(img_out), np.max(img_out)
    if p_high > p_low:
        img_out = exposure.rescale_intensity(img_out, in_range=(p_low, p_high),
                                             out_range=(0., 255.))
    return np.uint8(img_out)


# just for temporary testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img = imread('D:/PythonProjects/tiatoolbox/tests/data/source_image.png')

    # showing original RGB image and its luminosity standardized counterpart
    plt.figure('Image'), plt.imshow(img)
    img_luminosity_std = image_luminosity_standardiser(img, percentile=0.95)
    plt.figure('img_luminosity_std'), plt.imshow(img_luminosity_std)
    img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    plt.figure('luminosity'), plt.imshow(img_LAB[:, :, 0]/255.)
    img_luminosity_std_LAB = cv2.cvtColor(img_luminosity_std, cv2.COLOR_RGB2LAB)
    plt.figure('luminosity_std'), plt.imshow(img_luminosity_std_LAB[:, :, 0]/255.)

    # using image contract enhancement
    img_contrast = contrast_enhancer(img, low_p=2, high_p=98)
    plt.figure('Image Contrast enhanced'), plt.imshow(img_contrast)
    img_contrast_LAB = cv2.cvtColor(img_contrast, cv2.COLOR_RGB2LAB)
    plt.figure('luminosity_contrasted'), plt.imshow(img_contrast_LAB[:, :, 0]/255.)
    plt.show()
