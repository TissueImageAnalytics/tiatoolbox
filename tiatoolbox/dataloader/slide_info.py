"""
This file contains code to output or save slide information using python multiprocessing
"""
from tiatoolbox.dataloader import wsireader
from tiatoolbox.utils import misc_utils as misc
import os
import multiprocessing
from multiprocessing import Pool
from functools import partial


def single_file_run(file_name, input_dir, output_dir=None, mode="show"):
    """
    Single file run to output or save WSI meta data. Multiprocessing uses this function to run slide_info in parallel
    Args:
        file_name: WSI file name
        input_dir: Path to input directory
        output_dir: Path to output directory to save the output
        mode: "show" to display meta information only or "save" to save the meta information

    Returns:
        displays or saves WSI meta information

    """
    print(file_name, flush=True)
    _, file_type = os.path.splitext(file_name)

    if file_type == ".svs" or file_type == ".ndpi" or file_type == ".mrxs":
        wsi_reader = wsireader.WSIReader(
            input_dir=input_dir, file_name=file_name, output_dir=output_dir
        )
        if mode == "show":
            print(wsi_reader.slide_info(save_mode=False))
        else:
            _, name, _ = misc.split_path_name_ext(file_name)
            wsi_reader.slide_info(output_dir=output_dir, output_name=name + ".yaml")


def slide_info(
    wsi_input,
    output_dir=None,
    file_types=("*.ndpi", "*.svs", "*.mrxs"),
    mode="show",
    num_cpu=None,
):
    """
    Displays or saves WSI metadata
    Args:
        wsi_input: input path to WSI file or directory path
        output_dir: Path to output directory to save the output, default=wsi_input/../meta
        file_types: file types to capture from directory, default=("*.ndpi", "*.svs", "*.mrxs")
        mode: "show" to display meta information only or "save" to save the meta information, default=show
        num_cpu: num of cpus to use for multiprocessing, default=multiprocessing.cpu_count()

    Returns:
        displays or saves WSI meta information

    """

    if output_dir is None:
        if os.path.isfile(wsi_input):
            dir_path, _ = os.path.split(wsi_input)
            output_dir = os.path.join(dir_path, "..", "meta")
        elif os.path.isdir(wsi_input):
            output_dir = os.path.join(wsi_input, "..", "meta")

    if num_cpu is None:
        num_cpu = multiprocessing.cpu_count()

    if file_types is None:
        file_types = ("*.ndpi", "*.svs", "*.mrxs")

    if mode is None:
        mode = "show"

    if not os.path.isdir(output_dir) and mode == "save":
        os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(wsi_input):
        files_all = misc.grab_files_from_dir(
            input_path=wsi_input, file_types=file_types
        )
        with Pool(num_cpu) as p:
            p.map(
                partial(
                    single_file_run,
                    output_dir=output_dir,
                    input_dir=wsi_input,
                    mode=mode,
                ),
                files_all,
            )

    if os.path.isfile(wsi_input):
        input_dir, file_name = os.path.split(wsi_input)
        single_file_run(
            file_name=file_name, output_dir=output_dir, input_dir=input_dir, mode=mode
        )
