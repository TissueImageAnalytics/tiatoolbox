"""Miscellaneous small functions repeatedly used in tiatoolbox."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import IO, TYPE_CHECKING

import cv2
import joblib
import numcodecs
import numpy as np
import pandas as pd
import requests
import yaml
import zarr
from filelock import FileLock
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.geometry import shape as feature2geometry
from skimage import exposure

from tiatoolbox import logger
from tiatoolbox.annotation.storage import Annotation, AnnotationStore, SQLiteStore
from tiatoolbox.utils.exceptions import FileNotSupportedError

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike

    from shapely import geometry


def split_path_name_ext(
    full_path: PathLike | str,
) -> tuple[Path, str, list[str]]:
    """Split path of a file to directory path, file name and extensions.

    Args:
        full_path (PathLike | str):
            Path to a file.

    Returns:
        tuple:
            Three parts of the input file path:
            - :py:obj:`Path` - Parent directory path
            - :py:obj:`str` - File name
            - :py:obj:`list(str)` - File extensions

    Examples:
        >>> from tiatoolbox.utils.misc import split_path_name_ext
        >>> dir_path, file_name, extensions = split_path_name_ext(full_path)

    """
    input_path = Path(full_path)
    return input_path.parent.absolute(), input_path.name, input_path.suffixes


def grab_files_from_dir(
    input_path: PathLike,
    file_types: str | tuple[str, ...] = ("*.jpg", "*.png", "*.tif"),
) -> list[Path]:
    """Grab file paths specified by file extensions.

    Args:
        input_path (PathLike):
            Path to the directory where files
            need to be searched.
        file_types (str or tuple(str)):
            File types (extensions) to be searched.

    Returns:
        list:
            File paths as a python list. It has been sorted to ensure
            the same ordering across platforms.

    Examples:
        >>> from tiatoolbox import utils
        >>> file_types = ("*.ndpi", "*.svs", "*.mrxs")
        >>> files_all = utils.misc.grab_files_from_dir(input_path,
        ...     file_types=file_types)

    """
    input_path = Path(input_path)

    if isinstance(file_types, str):
        if len(file_types.split(",")) > 1:
            file_types = tuple(file_types.replace(" ", "").split(","))
        else:
            file_types = (file_types,)

    files_grabbed: list[Path] = []
    for files in file_types:
        files_grabbed.extend(input_path.glob(files))
    # Ensure same ordering
    files_grabbed.sort()
    return list(files_grabbed)


def save_yaml(
    input_dict: dict,
    output_path: PathLike = Path("output.yaml"),
    *,
    parents: bool = False,
    exist_ok: bool = False,
) -> None:
    """Save dictionary as yaml.

    Args:
        input_dict (dict):
            A variable of type 'dict'.
        output_path (PathLike):
            Path to save the output file.
        parents (bool):
            Make parent directories if they do not exist. Default is
            False.
        exist_ok (bool):
            Overwrite the output file if it exists. Default is False.

    Examples:
        >>> from tiatoolbox import utils
        >>> input_dict = {'hello': 'Hello World!'}
        >>> utils.misc.save_yaml(input_dict, './hello.yaml')

    """
    path = Path(output_path)
    if path.exists() and not exist_ok:
        msg = "File already exists."
        raise FileExistsError(msg)
    if parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w+") as yaml_file:
        yaml.dump(input_dict, yaml_file)


def imwrite(image_path: PathLike, img: np.ndarray) -> None:
    """Write numpy array to an image.

    Args:
        image_path (PathLike):
            File path (including extension) to save image to.
        img (:class:`numpy.ndarray`):
            Image array of dtype uint8, MxNx3.

    Examples:
        >>> from tiatoolbox import utils
        >>> import numpy as np
        >>> utils.misc.imwrite('BlankImage.jpg',
        ...     np.ones([100, 100, 3]).astype('uint8')*255)

    """
    image_path_str = str(image_path)

    if not cv2.imwrite(image_path_str, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
        msg = "Could not write image."
        raise OSError(msg)


def imread(image_path: PathLike, as_uint8: bool | None = None) -> np.ndarray:
    """Read an image as numpy array.

    Args:
        image_path (PathLike):
            File path (including extension) to read image.
        as_uint8 (bool):
            Read an image in uint8 format.

    Returns:
        :class:`numpy.ndarray`:
            Image array of dtype uint8, MxNx3.

    Examples:
        >>> from tiatoolbox import utils
        >>> img = utils.misc.imread('ImagePath.jpg')

    """
    if as_uint8 is None:
        as_uint8 = True  # default reading of images is in uint8 format.

    if not isinstance(image_path, (str, Path)):
        msg = "Please provide path to an image."
        raise TypeError(msg)

    if isinstance(image_path, str):
        image_path = Path(image_path)

    if image_path.suffix == ".npy":
        image = np.load(str(image_path))
    else:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if as_uint8:
        return image.astype(np.uint8)

    return image


def load_stain_matrix(stain_matrix_input: np.ndarray | PathLike) -> np.ndarray:
    """Load a stain matrix as a numpy array.

    Args:
        stain_matrix_input (ndarray | PathLike):
            Either a 2x3 or 3x3 numpy array or a path to a saved .npy /
            .csv file. If using a .csv file, there should be no column
            headers provided

    Returns:
        stain_matrix (:class:`numpy.ndarray`):
            The loaded stain matrix.

    Examples:
        >>> from tiatoolbox import utils
        >>> sm = utils.misc.load_stain_matrix(stain_matrix_input)

    """
    if isinstance(stain_matrix_input, (str, Path)):
        _, __, suffixes = split_path_name_ext(stain_matrix_input)
        if suffixes[-1] not in [".csv", ".npy"]:
            msg = (
                "If supplying a path to a stain matrix, "
                "use either a npy or a csv file"
            )
            raise FileNotSupportedError(
                msg,
            )

        if suffixes[-1] == ".csv":
            return pd.read_csv(stain_matrix_input).to_numpy()

        # only other option left for suffix[-1] is .npy
        return np.load(str(stain_matrix_input))

    if isinstance(stain_matrix_input, np.ndarray):
        return stain_matrix_input

    msg = "Stain_matrix must be either a path to npy/csv file or a numpy array"
    raise TypeError(
        msg,
    )


def get_luminosity_tissue_mask(img: np.ndarray, threshold: float) -> np.ndarray:
    """Get tissue mask based on the luminosity of the input image.

    Args:
        img (:class:`numpy.ndarray`):
            Input image used to obtain tissue mask.
        threshold (float):
            Luminosity threshold used to determine tissue area.

    Returns:
        tissue_mask (:class:`numpy.ndarray`):
            Binary tissue mask.

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
        msg = "Empty tissue mask computed."
        raise ValueError(msg)

    return tissue_mask


def mpp2common_objective_power(
    mpp: float | tuple[float, float],
    common_powers: tuple[float, ...] = (
        1,
        1.25,
        2,
        2.5,
        4,
        5,
        10,
        20,
        40,
        60,
        90,
        100,
    ),
) -> float:
    """Approximate (commonly used value) of objective power from mpp.

    Uses :func:`mpp2objective_power` to estimate and then rounds to the
    nearest value in `common_powers`.

    Args:
        mpp (float or tuple(float)): Microns per-pixel.
        common_powers (tuple or tuple(float, ...)): A sequence of objective
            power values to round to. Defaults to
            (1, 1.25, 2, 2.5, 4, 5, 10, 20, 40, 60, 90, 100).

    Returns:
        float:
            Objective power approximation.

    Examples:
        >>> mpp2common_objective_power(0.253)
        array(40)

        >>> mpp2common_objective_power(
        ...     (0.253, 0.478),
        ...     common_powers=(10.0, 20.0, 40.0),
        ... )
        array([40, 20])

    """
    op = mpp2objective_power(mpp)
    distances = [np.abs(op - power) for power in common_powers]
    return common_powers[np.argmin(distances)]


mpp2common_objective_power = np.vectorize(
    mpp2common_objective_power,
    excluded={"common_powers"},
)


@np.vectorize
def objective_power2mpp(
    objective_power: float | tuple[float, ...],
) -> float | np.ndarray:
    r"""Approximate mpp from objective power.

    The formula used for estimation is :math:`power = \frac{10}{mpp}`.
    This is a self-inverse function and therefore
    :func:`mpp2objective_power` is simply an alias to this function.

    Note that this function is wrapped in :class:`numpy.vectorize`.

    Args:
        objective_power (float or tuple(float, ...)): Objective power.

    Returns:
        float or tuple(float | np.ndarray):
            Microns per-pixel (MPP) approximations.

    Examples:
        >>> objective_power2mpp(40)
        array(0.25)

        >>> objective_power2mpp([40, 20, 10])
        array([0.25, 0.5, 1.])

    """
    return 10.0 / np.array(objective_power)


@np.vectorize
def mpp2objective_power(mpp: float | tuple[float]) -> float | tuple[float]:
    """Approximate objective_power from mpp.

    Alias to :func:`objective_power2mpp` as it is a self-inverse
    function.

    Args:
        mpp (float or tuple(float)): Microns per-pixel.

    Returns:
        float or tuple(float):
            Objective power approximations.

    Examples:
        >>> mpp2objective_power(0.25)
        array(40.)

        >>> mpp2objective_power([0.25, 0.5, 1.0])
        array([40., 20., 10.])

        >>> mpp2objective_power(0.253)
        array(39.5256917)

    """
    return objective_power2mpp(mpp)


def contrast_enhancer(img: np.ndarray, low_p: int = 2, high_p: int = 98) -> np.ndarray:
    """Enhance contrast of the input image using intensity adjustment.

    This method uses both image low and high percentiles.

    Args:
        img (:class:`numpy.ndarray`): input image used to obtain tissue mask.
            Image should be uint8.
        low_p (scalar): low percentile of image values to be saturated to 0.
        high_p (scalar): high percentile of image values to be saturated to 255.
            high_p should always be greater than low_p.

    Returns:
        img (:class:`numpy.ndarray`):
            Image (uint8) with contrast enhanced.

    Raises:
        AssertionError: Internal errors due to invalid img type.

    Examples:
        >>> from tiatoolbox import utils
        >>> img = utils.misc.contrast_enhancer(img, low_p=2, high_p=98)

    """
    # check if image is not uint8
    if img.dtype != np.uint8:
        msg = "Image should be uint8."
        raise AssertionError(msg)
    img_out = img.copy()
    percentiles = np.array(np.percentile(img_out, (low_p, high_p)))
    p_low, p_high = percentiles[0], percentiles[1]
    if p_low >= p_high:
        p_low, p_high = np.min(img_out), np.max(img_out)
    if p_high > p_low:
        img_out = exposure.rescale_intensity(
            img_out,
            in_range=(p_low, p_high),
            out_range=(0.0, 255.0),
        )
    return img_out.astype(np.uint8)


def __numpy_array_to_table(input_table: np.ndarray) -> pd.DataFrame:
    """Check numpy array to be 2 or 3 columns.

    If it has two columns then class should be assigned None.

    Args:
        input_table (np.ndarray): input table.

    Returns:
       table (:class:`pd.DataFrame`): Pandas DataFrame with desired features.

    Raises:
        ValueError: If the number of columns is not equal to 2 or 3.

    """
    if input_table.shape[1] == 2:  # noqa: PLR2004
        out_table = pd.DataFrame(input_table, columns=["x", "y"])
        out_table["class"] = None
        return out_table

    if input_table.shape[1] == 3:  # noqa: PLR2004
        return pd.DataFrame(input_table, columns=["x", "y", "class"])

    msg = "Numpy table should be of format `x, y` or `x, y, class`."
    raise ValueError(msg)


def __assign_unknown_class(input_table: pd.DataFrame) -> pd.DataFrame:
    """Creates a column and assigns None if class is unknown.

    Args:
        input_table: (pd.DataFrame):
            input table.

    Returns:
        table (:class:`pd.DataFrame`): Pandas DataFrame with desired features.

    Raises:
        ValueError:
            If the number of columns is not equal to 2 or 3.

    """
    if input_table.shape[1] not in [2, 3]:
        msg = "Input table must have 2 or 3 columns."
        raise ValueError(msg)

    if input_table.shape[1] == 2:  # noqa: PLR2004
        input_table["class"] = None

    return input_table


def read_locations(
    input_table: str | Path | PathLike | np.ndarray | pd.DataFrame,
) -> pd.DataFrame:
    """Read annotations as pandas DataFrame.

    Args:
        input_table (str| Path| PathLike | np.ndarray | pd.DataFrame`):
            Path to csv, npy or json. Input can also be a
            :class:`numpy.ndarray` or :class:`pandas.DataFrame`.
            First column in the table represents x position, second
            column represents y position. The third column represents the class.
            If the table has headers, the header should be x, y & class.
            Json should have `x`, `y` and `class` fields.

    Returns:
        pd.DataFrame:
            DataFrame with x, y location and class type.

    Raises:
        FileNotSupportedError:
            If the path to input table is not of supported type.

    Examples:
        >>> from tiatoolbox.utils.misc import read_locations
        >>> labels = read_locations('./annotations.csv')

    """
    if isinstance(input_table, (str, Path)):
        _, _, suffixes = split_path_name_ext(input_table)

        if suffixes[-1] == ".npy":
            out_table = np.load(input_table)
            return __numpy_array_to_table(out_table)

        if suffixes[-1] == ".csv":
            out_table = pd.read_csv(input_table, sep=None, engine="python")
            if "x" not in out_table.columns:
                out_table = pd.read_csv(
                    input_table,
                    header=None,
                    names=["x", "y", "class"],
                    sep=None,
                    engine="python",
                )

            return __assign_unknown_class(out_table)

        if suffixes[-1] == ".json":
            out_table = pd.read_json(input_table)
            return __assign_unknown_class(out_table)

        msg = "File type not supported."
        raise FileNotSupportedError(msg)

    if isinstance(input_table, np.ndarray):
        return __numpy_array_to_table(input_table)

    if isinstance(input_table, pd.DataFrame):
        return __assign_unknown_class(input_table)

    msg = "Please input correct image path or an ndarray image."
    raise TypeError(msg)


@np.vectorize
def conv_out_size(
    in_size: int,
    kernel_size: int = 1,
    padding: int = 0,
    stride: int = 1,
) -> int:
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
        int:
            Output size / number of features.

    Examples:
        >>> from tiatoolbox import utils
        >>> import numpy as np
        >>> utils.misc.conv_out_size(100, 3)
        >>> np.array(98)
        >>> utils.misc.conv_out_size(99, kernel_size=3, stride=2)
        >>> np.array(98)
        >>> utils.misc.conv_out_size((100, 100), kernel_size=3, stride=2)
        >>> np.array([49, 49])

    """
    return (np.floor((in_size - kernel_size + (2 * padding)) / stride) + 1).astype(int)


def parse_cv2_interpolaton(interpolation: str | int) -> int:
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
        interpolation (Union[str, int]):
            Interpolation mode string. Possible values are: nearest,
            linear, cubic, lanczos, area.

    Raises:
        ValueError:
            Invalid interpolation mode.

    Returns:
        int:
            OpenCV (cv2) interpolation enum.

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
    msg = "Invalid interpolation mode."
    raise ValueError(msg)


def assert_dtype_int(
    input_var: np.ndarray,
    message: str = "Input must be integer.",
) -> None:
    """Generate error if dtype is not int.

    Args:
        input_var (ndarray):
            Input variable to be tested.
        message (str):
            Error message to be displayed.

    Raises:
        AssertionError:
            If input_var is not of type int.

    """
    if not np.issubdtype(np.array(input_var).dtype, np.integer):
        raise AssertionError(message)


def download_data(
    url: str,
    save_path: PathLike | None = None,
    save_dir: PathLike | None = None,
    *,
    overwrite: bool = False,
    unzip: bool = False,
) -> Path:
    """Download data from a given URL to location.

    The function can overwrite data if demanded else no action is taken.

    Args:
        url (str):
            URL from where to download the data.
        save_path (PathLike):
            Location to download the data (including filename).
            Can't be used with save_dir.
        save_dir (PathLike):
            Directory to save the data. Can't be used with save_path.
        overwrite (bool):
            True to force overwriting of existing data, default=False
        unzip (bool):
            True to unzip the data, default=False

    """
    if save_path is not None and save_dir is not None:
        msg = "save_path and save_dir can't both be specified"
        raise ValueError(msg)

    if save_path is not None:
        save_dir = Path(save_path).parent
        save_path = Path(save_path)

    elif save_dir is not None:
        save_dir = Path(save_dir)
        save_path = save_dir / Path(url).name

    else:
        msg = "save_path or save_dir must be specified"
        raise ValueError(msg)

    logger.debug("Download from %s to %s", url, save_path)

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    if not overwrite and save_path.exists() and not unzip:
        return save_path

    lock_path = save_path.with_suffix(".lock")

    with FileLock(lock_path):
        if not overwrite and save_path.exists():
            pass  # file was downloaded by another process
        else:
            # Start the connection with a 5-second timeout
            # to avoid hanging indefinitely.
            response = requests.get(url, stream=True, timeout=5)
            # Raise an exception for status codes != 200
            response.raise_for_status()
            # Write the file in blocks of 1024 bytes to avoid running out of memory

            with tempfile.NamedTemporaryFile(mode="wb", delete=False) as templ_file:
                for block in response.iter_content(1024):
                    templ_file.write(block)

            # Move the temporary file to the desired location
            shutil.move(templ_file.name, save_path)

        if unzip:
            unzip_path = save_dir / save_path.stem
            unzip_data(save_path, unzip_path, del_zip=False)
            return unzip_path

    return save_path


def unzip_data(
    zip_path: PathLike,
    save_path: PathLike,
    *,
    del_zip: bool = True,
) -> None:
    """Extract data from zip file.

    Args:
        zip_path (PathLike): Path where the zip file is located.
        save_path (PathLike): Path where to save extracted files.
        del_zip (bool): Whether to delete initial zip file after extraction.

    """
    # Extract data from zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(save_path)
    if del_zip:
        zip_path = Path(zip_path)
        # Remove zip file
        Path.unlink(zip_path)


def __walk_list_dict(in_list_dict: dict | list[dict]) -> dict | list[dict]:
    """Recursive walk and jsonify in place.

    Args:
        in_list_dict (list or dict):  input list or a dictionary.

    Returns:
        list or dict

    """
    if isinstance(in_list_dict, dict):
        __walk_dict(in_list_dict)
    elif isinstance(in_list_dict, list):
        __walk_list(in_list_dict)
    elif isinstance(in_list_dict, np.ndarray):
        in_list_dict = in_list_dict.tolist()
        __walk_list(in_list_dict)
    elif isinstance(in_list_dict, np.generic):
        in_list_dict = in_list_dict.item()
    elif in_list_dict is not None and not isinstance(
        in_list_dict,
        (int, float, str, bool),
    ):
        msg = f"Value type `{type(in_list_dict)}` `{in_list_dict}` is not jsonified."
        raise TypeError(
            msg,
        )
    return in_list_dict


def __walk_list(lst: list) -> None:
    """Recursive walk and jsonify a list in place.

    Args:
        lst (list):  input list.

    """
    for i, v in enumerate(lst):
        lst[i] = __walk_list_dict(v)


def __walk_dict(dct: dict) -> None:
    """Recursive walk and jsonify a dictionary in place.

    Args:
        dct (dict):  input dictionary.

    """
    for k, v in dct.items():
        if not isinstance(k, (int, float, str, bool)):
            msg = f"Key type `{type(k)}` `{k}` is not jsonified."
            raise TypeError(msg)
        dct[k] = __walk_list_dict(v)


def save_as_json(
    data: dict | list,
    save_path: str | PathLike,
    *,
    parents: bool = False,
    exist_ok: bool = False,
) -> None:
    """Save data to a json file.

    The function will deepcopy the `data` and then jsonify the content
    in place. Support data types for jsonify consist of `str`, `int`, `float`,
    `bool` and their np.ndarray respectively.

    Args:
        data (dict or list):
            Input data to save.
        save_path (PathLike):
            Output to save the json of `input`.
        parents (bool):
            Make parent directories if they do not exist. Default is
            False.
        exist_ok (bool):
            Overwrite the output file if it exists. Default is False.


    """
    shadow_data = copy.deepcopy(data)  # make a copy of source input
    if not isinstance(shadow_data, (dict, list)):
        msg = f"Type of `data` ({type(data)}) must be in (dict, list)."
        raise TypeError(msg)

    if isinstance(shadow_data, dict):
        __walk_dict(shadow_data)
    else:
        __walk_list(shadow_data)

    save_path = Path(save_path)
    if save_path.exists() and not exist_ok:
        msg = "File already exists."
        raise FileExistsError(msg)
    if parents:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(save_path, "w") as handle:  # skipcq: PTC-W6004
        json.dump(shadow_data, handle)


def select_device(*, on_gpu: bool) -> str:
    """Selects the appropriate device as requested.

    Args:
        on_gpu (bool):
            Selects gpu if True.

    Returns:
        str:
            "gpu" if on_gpu is True otherwise returns "cpu"

    """
    if on_gpu:
        return "cuda"

    return "cpu"


def get_bounding_box(img: np.ndarray) -> np.ndarray:
    """Get bounding box coordinate information.

    Given an image with zero and non-zero values. This function will
    return the minimal box that contains all non-zero values.

    Args:
        img (ndarray):
            Image to get the bounding box.

    Returns:
        bound (ndarray):
            Coordinates of the box in the form of `[start_x, start_y,
            end_x, end_y]`.

    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    r_max += 1
    cmax += 1
    return np.array([c_min, r_min, cmax, r_max])


def string_to_tuple(in_str: str) -> tuple[str, ...]:
    """Splits input string to tuple at ','.

    Args:
        in_str (str):
            input string.

    Returns:
        tuple[str, ...]:
            Return a tuple of strings by splitting in_str at ','.

    """
    return tuple(substring.strip() for substring in in_str.split(","))


def ppu2mpp(ppu: int, units: str | int) -> float:
    """Convert pixels per unit (ppu) to microns per pixel (mpp).

    Args:
        ppu (int):
            Pixels per unit.
        units (Union[str, int]):
            Units of pixels per unit. Valid options are "cm",
            "centimeter", "inch", 2 (inches), 3(cm).

    Returns:
        mpp (float):
            Microns per pixel.

    """
    microns_per_unit = {
        "meter": 1e6,  # 1,000,000
        "m": 1e6,  # 1,000,000
        "centimeter": 1e4,  # 10,000
        "cm": 1e4,  # 10,000
        "mm": 1e3,  # 1,000
        "inch": 25400,
        "in": 25400,
        2: 25400,  # inches in TIFF tags
        3: 1e4,  # cm in TIFF tags
    }
    if units not in microns_per_unit:
        msg = f"Invalid units: {units}"
        raise ValueError(msg)
    return 1 / ppu * microns_per_unit[units]


def select_cv2_interpolation(scale_factor: float | np.ndarray) -> str:
    """Return appropriate interpolation method for opencv based image resize.

    Args:
        scale_factor (float or np.ndarray[float, float]):
            Image resize scale factor.

    Returns:
        str:
            interpolation type

    """
    if np.any(scale_factor > 1.0):
        return "cubic"
    return "area"


def store_from_dat(
    fp: IO | PathLike,
    scale_factor: tuple[float, float] = (1, 1),
    typedict: dict | None = None,
    origin: tuple[float, float] = (0, 0),
    cls: type[AnnotationStore] = SQLiteStore,
) -> AnnotationStore:
    """Load annotations from a hovernet-style .dat file.

    Args:
        fp (IO | PathLike):
            The file path or handle to load from.
        scale_factor (Tuple[float, float]):
            The scale factor in each dimension to use when loading the annotations.
            All coordinates will be multiplied by this factor to allow import of
            annotations saved at non-baseline resolution. Should be model_mpp/slide_mpp,
            where model_mpp is the resolution at which the annotations were saved.
            If scale information is stored in the .dat file (as in cerberus output),
            that will be used and this arg will be ignored.
        typedict (Dict[str, str]):
            A dictionary mapping annotation types to annotation keys. Annotations
            with a type that is a key in the dictionary, will have their type
            replaced by the corresponding value. Useful for providing descriptive
            names to non-descriptive types,
            eg {1: 'Epithelial Cell', 2: 'Lymphocyte', 3: ...}.
            For multi-head output, should be a dict of dicts, eg:
            {'head1': {1: 'Epithelial Cell', 2: 'Lymphocyte', 3: ...},
            'head2': {1: 'Gland', 2: 'Lumen', 3: ...}, ...}.
        origin (Tuple[float, float]):
            The x and y coordinates to use as the origin for the annotations.
        cls (AnnotationStore):
            The class to use for the annotation store. Defaults to SQLiteStore.

    Returns:
        AnnotationStore:
            A new annotation store with the annotations loaded from the file.

    """
    store = cls()
    add_from_dat(store, fp, scale_factor, typedict=typedict, origin=origin)
    if isinstance(store, SQLiteStore):
        store.create_index("area", '"area"')
    return store


def make_valid_poly(
    poly: geometry,
    origin: tuple[float, float] | None = None,
) -> geometry:
    """Helper function to make a valid polygon.

    Args:
        poly (Polygon):
            The polygon to make valid.
        origin (Tuple[float, float]):
            The x and y coordinates to use as the origin for the annotation.

    Returns:
        geometry:
            A valid geometry.

    """
    if origin != (0, 0) and origin is not None:
        # transform coords to be relative to given pt.
        poly = translate(poly, -origin[0], -origin[1])
    if poly.is_valid:
        return poly
    logger.warning("Invalid geometry found, fix using buffer().", stacklevel=3)
    return poly.buffer(0.01)


def anns_from_hoverdict(
    data: dict,
    props: list,
    typedict: dict | None,
    origin: tuple[float, float],
    scale_factor: tuple[float, float],
) -> list[Annotation]:
    """Helper function to create list of Annotation objects.

    Creates annotations from a hovernet-style dict of segmentations, mapping types
    using type dict if provided.

    Args:
        data (dict):
            A dictionary of segmentations
        props (list):
            A list of properties
        typedict (dict):
            A dictionary mapping annotation types to more descriptive names.
        origin (tuple[float, float]):
            The x and y coordinates to use as the origin for the annotations.
        scale_factor (tuple[float, float]):
            The scale factor to use when loading the annotations. All coordinates
            will be multiplied by this factor.

    Returns:
        list(Annotation):
            A list of Annotation objects.

    """
    return [
        Annotation(
            make_valid_poly(
                feature2geometry(
                    {
                        "type": ann.get("geom_type", "Polygon"),
                        "coordinates": scale_factor * np.array([ann["contour"]]),
                    },
                ),
                origin,
            ),
            {
                prop: (
                    typedict[ann[prop]]
                    if prop == "type" and typedict is not None
                    else ann[prop]
                )
                for prop in props[3:]
                if prop in ann
            },
        )
        for ann in data.values()
    ]


def make_default_dict(data: dict, subcat: str) -> dict:
    """Helper function to create a default typedict if none is provided.

    The unique types in the data are given a prefix to differentiate
    types from different heads of a multi-head model.
    For example, types 1,2, etc. in the 'Gland' head will become
    'Gla: 1', 'Gla: 2', etc.

    Args:
        data (dict):
            The data loaded from the .dat file.
        subcat (str):
            The subcategory of the data, eg 'Gland' or 'Nuclei'.

    Returns:
        A dictionary mapping types to more descriptive names.

    """
    types = {
        data[subcat][ann_id]["type"]
        for ann_id in data[subcat]
        if "type" in data[subcat][ann_id]
    }
    num_chars = np.minimum(3, len(subcat))
    return {t: f"{subcat[:num_chars]}: {t}" for t in types}


def add_from_dat(
    store: AnnotationStore,
    fp: IO | PathLike,
    scale_factor: tuple[float, float] = (1, 1),
    typedict: dict | None = None,
    origin: tuple[float, float] = (0, 0),
) -> None:
    """Add annotations from a .dat file to an existing store.

    Make the best effort to create valid shapely geometries from provided contours.

    Args:
        store (AnnotationStore):
            An :class:`AnnotationStore` object.
        fp (IO | PathLike):
            The file path or handle to load from.
        scale_factor (tuple[float, float]):
            The scale factor to use when loading the annotations. All coordinates
            will be multiplied by this factor to allow import of annotations saved
            at non-baseline resolution. Should be model_mpp/slide_mpp, where
            model_mpp is the resolution at which the annotations were saved.
            If scale information is stored in the .dat file (as in cerberus output),
            that will be used and this arg will be ignored.
        typedict (Dict[str, str]):
            A dictionary mapping annotation types to annotation keys. Annotations
            with a type that is a key in the dictionary, will have their type
            replaced by the corresponding value. Useful for providing descriptive
            names to non-descriptive types,
            e.g., {1: 'Epithelial Cell', 2: 'Lymphocyte', 3: ...}.
            For multi-head output, should be a dict of dicts, e.g.:
            {'head1': {1: 'Epithelial Cell', 2: 'Lymphocyte', 3: ...},
            'head2': {1: 'Gland', 2: 'Lumen', 3: ...}, ...}.
        origin (tuple(float, float)):
            The x and y coordinates to use as the origin for the annotations.

    """
    data = joblib.load(fp)
    props = list(data[next(iter(data.keys()))].keys())
    if "base_resolution" in data and "proc_resolution" in data:
        # we can infer scalefactor from resolutions
        scale_factor = (
            data["proc_resolution"]["resolution"]
            / data["base_resolution"]["resolution"]
        )
        logger.info("Scale factor inferred from resolutions: %s", scale_factor)
    if "contour" not in props:
        # assume cerberus format with objects subdivided into categories
        anns = []
        for subcat in data:
            if (
                subcat in {"resolution", "proc_dimensions", "base_dimensions"}
                or "resolution" in subcat
            ):
                continue
            props = next(iter(data[subcat].values()))
            if not isinstance(props, dict):
                continue
            props = list(props.keys())
            # use type dictionary if available else auto-generate
            if typedict is None:
                typedict_sub = make_default_dict(data, subcat)
            else:
                typedict_sub = typedict[subcat]
            anns.extend(
                anns_from_hoverdict(
                    data[subcat],
                    props,
                    typedict_sub,
                    origin,
                    scale_factor,
                ),
            )
    else:
        anns = anns_from_hoverdict(data, props, typedict, origin, scale_factor)

    logger.info("Added %d annotations.", len(anns))
    store.append_many(anns)


def dict_to_store(
    patch_output: dict,
    scale_factor: tuple[int, int],
    class_dict: dict | None = None,
    save_path: Path | None = None,
) -> AnnotationStore | Path:
    """Converts (and optionally saves) output of TIAToolbox engines as AnnotationStore.

    Args:
        patch_output (dict):
            A dictionary in the TIAToolbox Engines output format. Important
            keys are "probabilities", "predictions", "coordinates", and "labels".
        scale_factor (tuple[int, int]):
            The scale factor to use when loading the
            annotations. All coordinates will be multiplied by this factor to allow
            conversion of annotations saved at non-baseline resolution to baseline.
            Should be model_mpp/slide_mpp.
        class_dict (dict):
            Optional dictionary mapping class indices to class names.
        save_path (str or Path):
            Optional Output directory to save the Annotation
            Store results.

    Returns:
        (SQLiteStore or Path):
            An SQLiteStore containing Annotations for each patch
            or Path to file storing SQLiteStore containing Annotations
            for each patch.

    """
    if "coordinates" not in patch_output:
        # we cant create annotations without coordinates
        msg = "Patch output must contain coordinates."
        raise ValueError(msg)
    # get relevant keys
    class_probs = patch_output.get("probabilities", [])
    preds = patch_output.get("predictions", [])
    patch_coords = np.array(patch_output.get("coordinates", []))
    if not np.all(np.array(scale_factor) == 1):
        patch_coords = patch_coords * (np.tile(scale_factor, 2))  # to baseline mpp
    labels = patch_output.get("labels", [])
    # get classes to consider
    if len(class_probs) == 0:
        classes_predicted = np.unique(preds).tolist()
    else:
        classes_predicted = range(len(class_probs[0]))
    if class_dict is None:
        # if no class dict create a default one
        class_dict = {i: i for i in np.unique(preds + labels).tolist()}

    # find what keys we need to save
    keys = ["predictions"]
    keys = keys + [key for key in ["probabilities", "labels"] if key in patch_output]

    # put patch predictions into a store
    annotations = []
    for i, pred in enumerate(preds):
        if "probabilities" in keys:
            props = {
                f"prob_{class_dict[j]}": class_probs[i][j] for j in classes_predicted
            }
        else:
            props = {}
        if "labels" in keys:
            props["label"] = class_dict[labels[i]]
        props["type"] = class_dict[pred]
        annotations.append(Annotation(Polygon.from_bounds(*patch_coords[i]), props))
    store = SQLiteStore()
    keys = store.append_many(annotations, [str(i) for i in range(len(annotations))])

    # if a save director is provided, then dump store into a file
    if save_path:
        # ensure parent directory exisits
        save_path.parent.absolute().mkdir(parents=True, exist_ok=True)
        # ensure proper db extension
        save_path = save_path.parent.absolute() / (save_path.stem + ".db")
        store.dump(save_path)
        return save_path

    return store


def dict_to_zarr(
    raw_predictions: dict,
    save_path: Path,
    **kwargs: dict,
) -> Path:
    """Saves the output of TIAToolbox engines to a zarr file.

    Args:
        raw_predictions (dict):
            A dictionary in the TIAToolbox Engines output format.
        save_path (str or Path):
            Path to save the zarr file.
        **kwargs (dict):
            Keyword Args to update patch_pred_store_zarr attributes.


    Returns:
        Path to zarr file storing the patch predictor output

    """
    # Default values for Compressor and Chunks set if not received from kwargs.
    compressor = (
        kwargs["compressor"] if "compressor" in kwargs else numcodecs.Zstd(level=1)
    )
    chunks = kwargs.get("chunks", 10000)

    # ensure proper zarr extension
    save_path = save_path.parent.absolute() / (save_path.stem + ".zarr")

    # save to zarr
    predictions_array = np.array(raw_predictions["predictions"])
    z = zarr.open(
        save_path,
        mode="w",
        shape=predictions_array.shape,
        chunks=chunks,
        compressor=compressor,
    )
    z[:] = predictions_array

    return save_path
