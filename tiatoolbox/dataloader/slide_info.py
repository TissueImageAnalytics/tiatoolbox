"""
Get Slide Meta Data information
"""
from tiatoolbox.dataloader import wsireader
from tiatoolbox.decorators.multiproc import TIAMultiProcess

import os


@TIAMultiProcess(iter_on="input_path")
def slide_info(input_path, output_dir=None):
    """
    slide_info()
    Single file run to output or save WSI meta data. Multiprocessing uses this function to run slide_info in parallel
    Args:
        input_path: Path to whole slide image
        output_dir: Path to output directory to save the output
    Returns:
        displays or saves WSI meta information

    Examples:
        >>> from tiatoolbox.dataloader.slide_info import slide_info
        >>> from tiatoolbox import utils
        >>> file_types = ("*.ndpi", "*.svs", "*.mrxs")
        >>> files_all = utils.misc.grab_files_from_dir(input_path, file_types=file_types)
        >>> slide_params = slide_info(input_path=files_all, workers=2)
        >>> for slide_param in slide_params:
        >>>        utils.misc.save_yaml(slide_param, slide_param["file_name"] + ".yaml")
        >>>        print(type(slide_param))

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
