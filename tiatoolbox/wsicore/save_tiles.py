"""Save image tiles from the whole slide image."""
import pathlib

from tiatoolbox.wsicore.wsireader import WSIReader


def save_tiles(
    input_path,
    output_dir="tiles",
    tile_objective_value=20,
    tile_read_size=(5000, 5000),
    verbose=True,
):
    """Save image tiles for whole slide image.

    Default file format for tiles is jpg.

    Args:
        input_path (str or pathlib.Path):
            Path to whole slide image
        output_dir (str or pathlib.Path):
            Path to output directory to save the output
        tile_objective_value (int):
            objective value at which tile is generated, default=20.
        tile_read_size (tuple(int)):
            Tile (width, height), default=(5000, 5000).
        verbose (bool):
            Print output, default=True

    Examples:
        >>> from tiatoolbox.wsicore.save_tiles import save_tiles
        >>> from tiatoolbox.utils import misc
        >>> file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
        >>> files_all = misc.grab_files_from_dir(input_path,
        ...     file_types=file_types)
        >>> for curr_file in files_all:
        ...     save_tiles(input_path=curr_file,
        ...         output_dir="tiles",
        ...         tile_objective_value=10,
        ...         tile_read_size = (5000,5000))

    """
    input_path = pathlib.Path(input_path)
    if verbose:
        print(input_path.name, flush=True)

    wsi = WSIReader.open(input_img=input_path)
    wsi.save_tiles(
        output_dir=output_dir,
        tile_objective_value=tile_objective_value,
        tile_read_size=tile_read_size,
        verbose=verbose,
    )
