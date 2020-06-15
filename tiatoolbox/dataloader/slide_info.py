"""
This file contains code to output or save slide information using python multiprocessing
"""
from tiatoolbox.dataloader import wsireader
from tiatoolbox.decorators.multiproc import TIAMultiProcess

import os


@TIAMultiProcess(iter_on="input_path")
def slide_info(input_path, output_dir=None):
    """
    Single file run to output or save WSI meta data. Multiprocessing uses this function to run slide_info in parallel
    Args:
        input_path: Path to whole slide image
        output_dir: Path to output directory to save the output
    Returns:
        displays or saves WSI meta information

    """

    input_dir, file_name = os.path.split(input_path)

    print(file_name, flush=True)
    _, file_type = os.path.splitext(file_name)

    if file_type == ".svs" or file_type == ".ndpi" or file_type == ".mrxs":
        wsi_reader = wsireader.WSIReader(
            input_dir=input_dir, file_name=file_name, output_dir=output_dir
        )
        info = wsi_reader.slide_info()
        return info
    else:
        print("File type not supported")
        return None
