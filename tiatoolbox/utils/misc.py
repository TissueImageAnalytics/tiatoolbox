"""
This file contains miscellaneous small functions repeatedly required and used in the repo
"""
import os
import pathlib


def split_path_name_ext(full_path):
    """
    Split path of a file to directory path, file name and extension

    Args:
        full_path: Path to a file

    Returns:
        input_dir: directory path
        file_name: name of the file without extension
        ext: file extension
    """
    input_dir, file_name = os.path.split(full_path)
    file_name, ext = os.path.splitext(file_name)
    return input_dir, file_name, ext


def grab_files_from_dir(input_path, file_types=("*.jpg", "*.png", "*.tif")):
    """
    Grabs file paths specified by file extensions

    Args:
        input_path: path to the directory where files need to be searched
        file_types: file types (extensions) to be searched

    Returns:
        list: file paths as a python list
    """
    input_path = pathlib.Path(input_path)

    if type(file_types) == str:
        if len(file_types.split(",")) > 1:
            file_types = tuple(file_types.split(","))
        else:
            file_types = (file_types,)

    files_grabbed = []
    for files in file_types:
        files_grabbed.extend(input_path.glob(files))

    return list(files_grabbed)
