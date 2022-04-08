"""Get Slide Meta Data information."""
import pathlib

from tiatoolbox.wsicore.wsireader import WSIReader


def slide_info(input_path, verbose=True):
    """Return WSI meta data.

    Args:
        input_path (str or pathlib.Path):
            Path to whole slide image.
        verbose (bool):
            Print output, default=True.

    Returns:
        WSIMeta:
            Metadata information.

    Examples:
        >>> from tiatoolbox.wsicore.slide_info import slide_info
        >>> from tiatoolbox import utils
        >>> file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
        >>> files_all = utils.misc.grab_files_from_dir(input_path,
        ...     file_types=file_types)
        >>> for curr_file in files_all:
        ...     slide_param = slide_info(input_path=curr_file)
        ...     utils.misc.save_yaml(slide_param.as_dict(),
        ...           str(slide_param.file_path) + ".yaml")
        ...     print(slide_param.as_dict())

    """
    input_path = pathlib.Path(input_path)
    if verbose:
        print(input_path.name, flush=True)

    wsi = WSIReader.open(input_img=input_path)
    info = wsi.info
    if verbose:
        print(info.as_dict())

    return info
