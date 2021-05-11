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
from typing import Union
from tiatoolbox.utils.exceptions import FileNotSupported

import cv2
import pathlib
import yaml
import pandas as pd
import numpy as np
from skimage import exposure


def split_path_name_ext(full_path):
    """Split path of a file to directory path, file name and extension.

    Args:
        full_path (str or pathlib.Path): Path to a file

    Returns:
        tuple: Three parts of the input file path:
            - :py:obj:`pathlib.Path` - parent directory path
            - :py:obj:`str` - file name
            - :py:obj:`str` - file extension

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
        input_path (str or pathlib.Path): Path to the directory where files
            need to be searched
        file_types (str or tuple(str)): File types (extensions) to be searched

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
            file_types = tuple(file_types.replace(" ", "").split(","))
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
        output_path (str or pathlib.Path): Path to save the output file

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
        image_path (str or pathlib.Path): file path (including extension)
            to save image
        img (:class:`numpy.ndarray`): image array of dtype uint8, MxNx3

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
        image_path (str or pathlib.Path): file path (including extension) to read image

    Returns:
        img (:class:`numpy.ndarray`): image array of dtype uint8, MxNx3

    Examples:
        >>> from tiatoolbox import utils
        >>> img = utils.misc.imread('ImagePath.jpg')

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
        stain_matrix (:class:`numpy.ndarray`): the loaded stain matrix.

    Examples:
        >>> from tiatoolbox import utils
        >>> sm = utils.misc.load_stain_matrix(stain_matrix_input)

    """
    if isinstance(stain_matrix_input, (str, pathlib.Path)):
        _, __, ext = split_path_name_ext(stain_matrix_input)
        if ext == ".csv":
            stain_matrix = pd.read_csv(stain_matrix_input).to_numpy()
        elif ext == ".npy":
            stain_matrix = np.load(str(stain_matrix_input))
        else:
            raise FileNotSupported(
                "If supplying a path to a stain matrix, use either a \
                npy or a csv file"
            )
    elif isinstance(stain_matrix_input, np.ndarray):
        stain_matrix = stain_matrix_input
    else:
        raise TypeError(
            "Stain_matrix must be either a path to npy/csv file or a numpy array"
        )

    return stain_matrix


def get_luminosity_tissue_mask(img, threshold):
    """Get tissue mask based on the luminosity of the input image.

    Args:
        img (:class:`numpy.ndarray`): input image used to obtain tissue mask.
        threshold (float): luminosity threshold used to determine tissue area.

    Returns:
        tissue_mask (:class:`numpy.ndarray`): binary tissue mask.

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
        raise ValueError("Empty tissue mask computed")

    return tissue_mask


def mpp2common_objective_power(
    mpp, common_powers=(1, 1.25, 2, 2.5, 4, 5, 10, 20, 40, 60, 90, 100)
):
    """Approximate (commonly used value) of objective power from mpp.

    Uses :func:`mpp2objective_power` to estimate and then rounds to the
    nearest value in `common_powers`.

    Args:
        mpp (float or tuple(float)): Microns per-pixel.
        common_powers (list of float): A sequence of objective
            power values to round to. Defaults to
            (1, 1.25, 2, 2.5, 4, 5, 10, 20, 40, 60, 90, 100).

    Returns:
        float: Objective power approximation.

    Examples:
        >>> mpp2common_objective_power(0.253)
        array(40)

        >>> mpp2common_objective_power(
        ...     [0.253, 0.478],
        ...     common_powers=(10, 20, 40),
        ... )
        array([40, 20])

    """
    op = mpp2objective_power(mpp)
    distances = [np.abs(op - power) for power in common_powers]
    closest_match = common_powers[np.argmin(distances)]
    return closest_match


mpp2common_objective_power = np.vectorize(
    mpp2common_objective_power, excluded={"common_powers"}
)


@np.vectorize
def objective_power2mpp(objective_power):
    r"""Approximate mpp from objective power.

    The formula used for estimation is :math:`power = \frac{10}{mpp}`.
    This is a self-inverse function and therefore
    :func:`mpp2objective_power` is simply an alias to this function.

    Note that this function is wrapped in :class:`numpy.vectorize`.

    Args:
        objective_power (float or tuple(float)): Objective power.

    Returns:
        numpy.ndarray: Microns per-pixel (MPP) approximations.

    Examples:
        >>> objective_power2mpp(40)
        array(0.25)

        >>> objective_power2mpp([40, 20, 10])
        array([0.25, 0.5, 1.])

    """
    return 10 / float(objective_power)


@np.vectorize
def mpp2objective_power(mpp):
    """Approximate objective_power from mpp.

    Alias to :func:`objective_power2mpp` as it is a self-inverse
    function.

    Args:
        mpp (float or tuple(float)): Microns per-pixel.

    Returns:
        :class:`numpy.ndarray`: Objective power approximations.

    Examples:
        >>> mpp2objective_power(0.25)
        array(40.)

        >>> mpp2objective_power([0.25, 0.5, 1.0])
        array([40., 20., 10.])

        >>> mpp2objective_power(0.253)
        array(39.5256917)

    """
    return objective_power2mpp(mpp)


def contrast_enhancer(img, low_p=2, high_p=98):
    """Enhancing contrast of the input image using intensity adjustment.
       This method uses both image low and high percentiles.

    Args:
        img (:class:`numpy.ndarray`): input image used to obtain tissue mask.
            Image should be uint8.
        low_p (scalar): low percentile of image values to be saturated to 0.
        high_p (scalar): high percentile of image values to be saturated to 255.
            high_p should always be greater than low_p.

    Returns:
        img (:class:`numpy.ndarray`): Image (uint8) with contrast enhanced.

    Raises:
        AssertionError: Internal errors due to invalid img type.

    Examples:
        >>> from tiatoolbox import utils
        >>> img = utils.misc.contrast_enhancer(img, low_p=2, high_p=98)

    """
    # check if image is not uint8
    if not img.dtype == np.uint8:
        raise AssertionError("Image should be uint8.")
    img_out = img.copy()
    p_low, p_high = np.percentile(img_out, (low_p, high_p))
    if p_low >= p_high:
        p_low, p_high = np.min(img_out), np.max(img_out)
    if p_high > p_low:
        img_out = exposure.rescale_intensity(
            img_out, in_range=(p_low, p_high), out_range=(0.0, 255.0)
        )
    return np.uint8(img_out)


def read_locations(input_table):
    """Read annotations as pandas DataFrame.

    Args:
        input_table (str or pathlib.Path or :class:`numpy.ndarray` or
         :class:`pandas.DataFrame`): path to csv, npy or json. Input can also be a
         :class:`numpy.ndarray` or :class:`pandas.DataFrame`.
         First column in the table represents x position, second
         column represents y position. The third column represents the class. If the
         table has headers, the header should be x, y & class. Json should have `x`, `y`
         and `class` fields.

    Returns:
        pd.DataFrame: DataFrame with x, y location and class type.

    Examples:
        >>> from tiatoolbox.utils.misc import read_locations
        >>> labels = read_locations('./annotations.csv')

    """
    if isinstance(input_table, (str, pathlib.Path)):
        _, _, suffix = split_path_name_ext(input_table)

        if suffix == ".npy":
            out_table = np.load(input_table)
            if out_table.shape[1] == 2:
                out_table = pd.DataFrame(out_table, columns=["x", "y"])
                out_table["class"] = None
            elif out_table.shape[1] == 3:
                out_table = pd.DataFrame(out_table, columns=["x", "y", "class"])
            else:
                raise ValueError(
                    "numpy table should be of format `x, y` or " "`x, y, class`"
                )

        elif suffix == ".csv":
            out_table = pd.read_csv(input_table, sep=None, engine="python")
            if "x" not in out_table.columns:
                out_table = pd.read_csv(
                    input_table,
                    header=None,
                    names=["x", "y", "class"],
                    sep=None,
                    engine="python",
                )
            if out_table.shape[1] == 2:
                out_table["class"] = None

        elif suffix == ".json":
            out_table = pd.read_json(input_table)
            if out_table.shape[1] == 2:
                out_table["class"] = None

        else:
            raise FileNotSupported("Filetype not supported.")

    elif isinstance(input_table, np.ndarray):
        if input_table.shape[1] == 3:
            out_table = pd.DataFrame(input_table, columns=["x", "y", "class"])
        elif input_table.shape[1] == 2:
            out_table = pd.DataFrame(input_table, columns=["x", "y"])
            out_table["class"] = None
        else:
            raise ValueError("Input array must have 2 or 3 columns.")

    elif isinstance(input_table, pd.DataFrame):
        out_table = input_table
        if out_table.shape[1] == 2:
            out_table["class"] = None
        elif out_table.shape[1] < 2:
            raise ValueError("Input table must have 2 or 3 columns.")

    else:
        raise TypeError("Please input correct image path or an ndarray image.")

    return out_table


@np.vectorize
def conv_out_size(in_size, kernel_size=1, padding=0, stride=1):
    r"""Calculate convolution output size.

    This is a numpy vectorised function.

    .. math::
        \begin{split}
        n_{out} &= \bigg\lfloor {{\frac{n_{in} +2p - k}{s}}} \bigg\rfloor + 1 \\
        n_{in} &: \text{Number of input features} \\
        n_{out} &: \text{Number of output features} \\
        p &: \text{Padding size} \\
        k &: \text{Kernel size} \\
        s &: \text{Stride size} \\
        \end{split}

    Args:
        in_size (int): Input size / number of input features.
        kernel_size (int): Kernel size.
        padding (int): Kernel size.
        stride (int): Stride size.

    Returns:
        int: Output size / number of features.

    Examples:
        >>> from tiatoolbox import utils
        >>> utils.misc.conv_out_size(100, 3)
        >>> array(98)
        >>> utils.misc.conv_out_size(99, kernel_size=3, stride=2)
        >>> array(98)
        >>> utils.misc.conv_out_size((100, 100), kernel_size=3, stride=2)
        >>> array([49, 49])

  """
    return (np.floor((in_size - kernel_size + (2 * padding)) / stride) + 1).astype(int)


def parse_cv2_interpolaton(interpolation: Union[str, int]) -> int:
    """Convert a string to a OpenCV (cv2) interpolation enum.

    Interpolation modes:
        - nearest
        - linear
        - area
        - cubic
        - lanczos

    Valid integer values for cv2 interpolation enums are passed through.
    See the `cv::InterpolationFlags`_ documentation for more
    on cv2 (OpenCV) interpolation modes.

    .. _cv::InterpolationFlags:
        https://docs.opencv.org/4.0.0/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121

    Args:
        interpolation (Union[str, int]): Interpolation mode string.
            Possible values are: neares, linear, cubic, lanczos, area.

    Raises:
        ValueError: Invalid interpolation mode.

    Returns:
        int: OpenCV (cv2) interpolation enum.
    """
    if isinstance(interpolation, str):
        interpolation = interpolation.lower()
    if interpolation in ["nearest", cv2.INTER_NEAREST]:
        return cv2.INTER_NEAREST
    if interpolation in ["area", cv2.INTER_AREA]:
        return cv2.INTER_AREA
    if interpolation in ["linear", cv2.INTER_LINEAR]:
        return cv2.INTER_LINEAR
    if interpolation in ["cubic", cv2.INTER_CUBIC]:
        return cv2.INTER_CUBIC
    if interpolation in ["lanczos", cv2.INTER_LANCZOS4]:
        return cv2.INTER_LANCZOS4
    raise ValueError("Invalid interpolation mode.")


def assert_dtype_int(input_var, message="Input must be integer."):
    """Generate error if dtype is not int.

    Args:
        input_var (ndarray): input variable to be tested.
        message (str): Error message to be displayed.

    Returns:
        Generates an AssertionError message if input is not an int.

    """
    if not np.issubdtype(np.array(input_var).dtype, np.integer):
        raise AssertionError(message)
