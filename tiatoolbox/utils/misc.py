"""Miscellaneous small functions repeatedly used in tiatoolbox"""
import os
import cv2
import pathlib
import yaml


def split_path_name_ext(full_path):
    """Split path of a file to directory path, file name and extension

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
    input_dir, file_name = os.path.split(full_path)
    file_name, ext = os.path.splitext(file_name)
    return input_dir, file_name, ext


def grab_files_from_dir(input_path, file_types=("*.jpg", "*.png", "*.tif")):
    """Grabs file paths specified by file extensions

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
    """Save dictionary as yaml

    Args:
        input_dict (dict): A variable of type 'dict'
        output_path (str, pathlib.Path): Path to save the output file

    Returns:

    Examples:
        >>> from tiatoolbox import utils
        >>> input_dict = {'hello': 'Hello World!'}
        >>> utils.misc.save_yaml(input_dict, './hello.yaml')


    """
    with open(pathlib.Path(output_path), "w") as yaml_file:
        yaml.dump(input_dict, yaml_file)


def imwrite(image_path, img):
    """Write a numpy array to an image

    Args:
        image_path (str, pathlib.Path): file path (including extension) to save image
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


def imresize(img, scale_factor, interpolation=cv2.INTER_CUBIC):
    """Resize input image

    Args:
        img (ndarray): input image
        scale_factor (float): scaling factor to resize the input image
        interpolation (int): interpolation, default=cv2.INTER_CUBIC

    Returns:
        ndarray: resized image

    Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> from tiatoolbox.utils import misc
            >>> wsi_obj = wsireader.WSIReader(input_dir="./",
            ...     file_name="CMU-1.ndpi")
            >>> slide_thumbnail = wsi_obj.slide_thumbnail()
            >>> # Resize the image to half size using scale_factor 0.5
            >>> misc.imresize(slide_thumbnail, scale_factor=0.5)

    """
    # Estimate new dimension
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    # Resize image
    resized_img = cv2.resize(img, dim, interpolation=interpolation)

    return resized_img
