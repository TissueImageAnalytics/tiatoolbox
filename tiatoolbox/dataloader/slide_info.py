"""Get Slide Meta Data information"""
from tiatoolbox.dataloader import wsireader
from tiatoolbox.decorators.multiproc import TIAMultiProcess
from tiatoolbox.utils import misc


@TIAMultiProcess(iter_on="input_path")
def slide_info(input_path, output_dir=None):
    """Whole slide image meta data for single file. Multiprocessing decorator runs this
    function in parallel using the number of specified cpu cores.

    Args:
        input_path (str): Path to whole slide image
        output_dir (str): Path to output directory to save the output
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

    input_dir, file_name, ext = misc.split_path_name_ext(input_path)

    print(file_name+ext, flush=True)

    if ext in (".svs", ".ndpi", ".mrxs"):
        wsi_reader = wsireader.WSIReader(
            input_dir=input_dir, file_name=file_name+ext, output_dir=output_dir
        )
        info = wsi_reader.slide_info()
    else:
        print("File type not supported")
        info = None

    return info
