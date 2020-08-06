"""Get Slide Meta Data information"""
from tiatoolbox.dataloader import wsireader
from tiatoolbox.decorators.multiproc import TIAMultiProcess
from tiatoolbox.utils.exceptions import FileNotSupported
from tiatoolbox.utils.misc import split_path_name_ext


@TIAMultiProcess(iter_on="input_path")
def slide_info(input_path, output_dir=None, verbose=True):
    """Returns WSI meta data.
    Multiprocessing decorator runs this function in parallel.

    Args:
        input_path (str): Path to whole slide image
        output_dir (str): Path to output directory to save the output
        verbose (bool): Print output, default=True
        workers (int): num of cpu cores to use for multiprocessing

    Returns:
        list: list of dictionary Whole Slide meta information

    Examples:
        >>> from tiatoolbox.dataloader.slide_info import slide_info
        >>> from tiatoolbox import utils
        >>> file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
        >>> files_all = utils.misc.grab_files_from_dir(input_path,
        ...     file_types=file_types)
        >>> slide_params = slide_info(input_path=files_all, workers=2)
        >>> for slide_param in slide_params:
        ...        utils.misc.save_yaml(slide_param.as_dict(),
        ...             slide_param.file_name + ".yaml")
        ...        print(slide_param.as_dict())

    """

    input_dir, file_name, file_type = split_path_name_ext(input_path)

    if verbose:
        print(file_name + file_type, flush=True)

    if file_type in (".svs", ".ndpi", ".mrxs"):
        wsi_reader = wsireader.OpenSlideWSIReader(
            input_dir=input_dir,
            file_name=file_name + file_type,
            output_dir=output_dir
        )
        info = wsi_reader.slide_info
        if verbose:
            print(info.as_dict())
    elif file_type in (".jp2",):
        wsi_reader = wsireader.OmnyxJP2WSIReader(
            input_dir=input_dir,
            file_name=file_name + file_type,
            output_dir=output_dir,
        )
        info = wsi_reader.slide_info
        if verbose:
            print(info.as_dict())
    else:
        raise FileNotSupported(
            file_type + " file format is not supported.")

    return info
