"""
Miscellaneous small functions repeatedly required and used in the repo
"""
import os
import pathlib
import yaml


def split_path_name_ext(full_path):
    """
    Split path of a file to directory path, file name and extension

    Args:
        full_path: Path to a file

    Returns:
        tuple: Three sections of the input file path
        (input directory path, file name, file extension)

    """
    input_dir, file_name = os.path.split(full_path)
    file_name, ext = os.path.splitext(file_name)
    return input_dir, file_name, ext


def grab_files_from_dir(input_path, file_types=("*.jpg", "*.png", "*.tif")):
    """
    Grabs file paths specified by file extensions

    Args:
        input_path (str, pathlib.Path): path to the directory where files need to be searched
        file_types (str, tuple): file types (extensions) to be searched

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


def save_yaml(input_dict, output_path="output.yaml"):
    """
    Save dictionary as yaml
    Args:
        input_dict: A variable of type 'dict'
        output_path: Path to save the output file

    Returns:

    Examples:
        >>> input_dict = {'hello': 'Hello World!'}
        >>> save_yaml(input_dict, './hello.yaml')


    """
    with open(pathlib.Path(output_path), "w") as yaml_file:
        yaml.dump(input_dict, yaml_file)
