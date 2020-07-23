"""Get Slide Meta Data information"""
from tiatoolbox.dataloader import wsireader
from tiatoolbox.decorators.multiproc import TIAMultiProcess

import os


def slide_info_single(input_path, output_dir=None, verbose=True):
    """Wrapper for multi core slide_info
    Args:
        input_path (str): Path to whole slide image
        output_dir (str): Path to output directory to save the output
        verbose(bool): Print output, default=True

    Returns:
        dict: dictionary with Whole Slide meta information

    Examples:
        >>> from tiatoolbox.dataloader.slide_info import slide_info
        >>> from tiatoolbox import utils
        >>> file_types = ("*.ndpi", "*.svs", "*.mrxs")
        >>> files_all = utils.misc.grab_files_from_dir(input_path,
        ...     file_types=file_types)
        >>> for i in range(len(files_all)):
        ...     slide_param = slide_info_single(input_path=files_all[i])
        ...     utils.misc.save_yaml(slide_param,
        ...             slide_param["file_name"] + ".yaml")
        ...        print(slide_param)

    """
    input_dir, file_name = os.path.split(input_path)

    if verbose:
        print(file_name, flush=True)
    _, file_type = os.path.splitext(file_name)

    if file_type in (".svs", ".ndpi", ".mrxs"):
        wsi_reader = wsireader.OpenSlideWSIReader(
            input_dir=input_dir, file_name=file_name, output_dir=output_dir
        )
        info = wsi_reader.slide_info
        if verbose:
            print(info)
    else:
        print("File type not supported")
        info = None

    return info


@TIAMultiProcess(iter_on="input_path")
def slide_info(input_path, output_dir=None, verbose=True):
    """Single file run to output or save WSI meta data.

    Multiprocessing uses this function to run slide_info in parallel

    Args:
        input_path (str): Path to whole slide image
        output_dir (str): Path to output directory to save the output
        verbose(bool): Print output, default=True
        workers (int): num of cpu cores to use for multiprocessing
    Returns:
        list: list of dictionary Whole Slide meta information

    Examples:
        >>> from tiatoolbox.dataloader.slide_info import slide_info
        >>> from tiatoolbox import utils
        >>> file_types = ("*.ndpi", "*.svs", "*.mrxs")
        >>> files_all = utils.misc.grab_files_from_dir(input_path,
        ...     file_types=file_types)
        >>> slide_params = slide_info(input_path=files_all, workers=2)
        >>> for slide_param in slide_params:
        ...        utils.misc.save_yaml(slide_param,
        ...             slide_param["file_name"] + ".yaml")
        ...        print(slide_param)

        """
    return slide_info_single(
        input_path=input_path, output_dir=output_dir, verbose=verbose
    )
