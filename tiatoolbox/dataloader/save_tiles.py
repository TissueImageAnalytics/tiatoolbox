"""Save image tiles from the whole slide image"""
from tiatoolbox.dataloader import wsireader
from tiatoolbox.decorators.multiproc import TIAMultiProcess
from tiatoolbox.utils import misc


@TIAMultiProcess(iter_on="input_path")
def save_tiles(
    input_path,
    output_dir="tiles",
    tile_objective_value=20,
    tile_read_size_w=5000,
    tile_read_size_h=5000,
):
    """Save image tiles for whole slide image. Default file format for tiles is jpg.
    Multiprocessing decorator runs this function in parallel using the number of
    specified cpu cores.

    Args:
        input_path (str): Path to whole slide image
        output_dir (str): Path to output directory to save the output
        tile_objective_value (int): objective value at which tile is generated,
                default=20
        tile_read_size_w (int): tile width, default=5000
        tile_read_size_h (int): tile height, default=5000
        workers (int): num of cpu cores to use for multiprocessing

    Returns:

    Examples:
        >>> from tiatoolbox.dataloader.save_tiles import save_tiles
        >>> from tiatoolbox.utils import misc
        >>> file_types = ("*.ndpi", "*.svs", "*.mrxs")
        >>> files_all = misc.grab_files_from_dir(input_path,
        ...     file_types=file_types)
        >>> save_tiles(input_path=files_all,
        ...     output_dir="tiles",
        ...     tile_objective_value=10,
        ...     tile_read_size_w=5000,
        ...     tile_read_size_h=5000
        ...     )

    """

    input_dir, file_name, ext = misc.split_path_name_ext(input_path)

    print(file_name + ext, flush=True)

    if ext in (".svs", ".ndpi", ".mrxs"):
        wsi_reader = wsireader.WSIReader(
            input_dir=input_dir,
            file_name=file_name + ext,
            output_dir=output_dir,
            tile_objective_value=tile_objective_value,
            tile_read_size_w=tile_read_size_w,
            tile_read_size_h=tile_read_size_h,
        )
        wsi_reader.save_tiles()
    else:
        raise Exception("File type not supported")
