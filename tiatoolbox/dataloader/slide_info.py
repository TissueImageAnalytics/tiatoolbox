"""
This file contains code to output or save slide information using python multiprocessing
"""
from tiatoolbox.dataloader import wsireader
from tiatoolbox.utils import misc_utils as misc
from tiatoolbox.decorators.multiproc import TIAMultiProcess

import os


@TIAMultiProcess(iter_on='input_path')
def slide_info(input_path, output_dir=None, mode="show"):
    """
    Single file run to output or save WSI meta data. Multiprocessing uses this function to run slide_info in parallel
    Args:
        input_path: Path to whole slide image
        output_dir: Path to output directory to save the output
        mode: "show" to display meta information only or "save" to save the meta information

    Returns:
        displays or saves WSI meta information

    """

    input_dir, file_name = os.path.split(input_path)

    if output_dir is None:
        output_dir = os.path.join(input_dir, "..", "meta")

    if mode is None:
        mode = "show"

    if not os.path.isdir(output_dir) and mode == "save":
        os.makedirs(output_dir, exist_ok=True)

    print(file_name, flush=True)
    _, file_type = os.path.splitext(file_name)

    if file_type == ".svs" or file_type == ".ndpi" or file_type == ".mrxs":
        wsi_reader = wsireader.WSIReader(
            input_dir=input_dir, file_name=file_name, output_dir=output_dir
        )
        if mode == "show":
            info = wsi_reader.slide_info(save_mode=False)
            print(info)
            return info
        else:
            wsi_reader.slide_info(output_dir=output_dir, output_name=file_name + ".yaml")
            return os.path.join(output_dir, file_name+".yaml")
