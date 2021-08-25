"""Tile pyrmaid generation in standard formats.

Included methods are DeepZoom and Zoomify in addition to a generic
method.

These are generally intended for serialisation or streaming via a web
UI. The `get_tile` method returns a Pillow Image object which can
be easily serialised via the use of an io.BytesIO object or saved
directly to disk.
"""

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Union, Dict, Iterable

import numpy as np
from PIL import Image
from xml.etree import ElementTree as ET
import defusedxml

from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils.transforms import imresize

defusedxml.defuse_stdlib()


class TilePyramidGenerator:
    r"""Generic tile pyramid generator with sensible defaults.

    Args:
        wsi (WSIReader): The WSI reader object. Must implement
            `tiatoolbox.wsicore.wsi_Reader.WSIReader.read_rect`.
        tile_size (int): The size of tiles to generate. Default is
            256. Note that the output tile size will be
            :math:`\text{tile size} + 2 \times\text{overlap}`.
        downsample (int): The downsample factor between levels.
            Default is 2.
        tile_overlap (int): The number of extra pixel to add to each
            edge of the tile. Default is 0.

    """

    def __init__(
        self,
        wsi: WSIReader,
        tile_size: int = 256,
        downsample: int = 2,
        overlap: int = 0,
    ):
        self.wsi = wsi
        self.tile_size = tile_size
        self.overlap = overlap
        self.downsample = downsample

    @property
    def output_tile_size(self) -> int:
        r"""The size of the tile which will be returned.

        This is eqivalent to :math:`\text{tile size} + 2*\text{overlay}`.

        """
        return self.tile_size + 2 * self.overlap

    @lru_cache(maxsize=None)
    def level_downsample(self, level: int) -> float:
        """Find the downsample factor for a level."""
        return 2 ** (self.level_count - level - 1)

    @lru_cache(maxsize=None)
    def level_dimensions(self, level: int) -> Tuple[int, int]:
        """The total pixel dimensions of the tile pyramid at a given level.

        Args:
            level (int): The level to calculate the dimensions for.

        """
        baseline_dims = self.wsi.info.slide_dimensions
        level_dims = np.ceil(
            np.divide(baseline_dims, self.level_downsample(level))
        ).astype(int)
        return tuple(level_dims)

    @lru_cache(maxsize=None)
    def tile_grid_size(self, level: int) -> Tuple[int, int]:
        """Width and height of the minimal grid of tiles to cover the slide.

        Args:
            level (int): The level to calculate the grid size for.

        """
        if level >= self.level_count:
            raise IndexError
        return tuple(
            np.ceil(np.divide(self.level_dimensions(level), self.tile_size)).astype(int)
        )

    @property
    def sub_tile_level_count(self):
        return 0

    @property
    def level_count(self) -> int:
        """Number of levels in the tile pyramid.

        The number of levels is such that level_count - 1 is a 1:1 of
        the slide baseline resolution (level 0 of the WSI).

        """
        return int(
            np.ceil(
                np.log2(np.divide(self.wsi.info.slide_dimensions, self.tile_size))
            ).max()
            + self.sub_tile_level_count
        )

    def get_tile_thumb(self) -> Image:
        """Return a thumbnail which fits the whole slide in one tile.

        The thumbnail output size has the longest edge equal to the
        tile size. The other edge preserves the orignal aspect ratio.

        """
        slide_dims = np.array(self.wsi.info.slide_dimensions)
        tile_dim = self.tile_size + self.overlap
        out_dims = np.round(slide_dims / slide_dims.max() * tile_dim).astype(int)
        bounds = (0, 0, *slide_dims)
        thumb = self.wsi.read_bounds(
            bounds, resolution=self.wsi.info.level_count - 1, units="level"
        )
        thumb = imresize(thumb, output_size=out_dims)
        return Image.fromarray(thumb)

    def get_tile(
        self,
        level: int,
        x: int,
        y: int,
        pad_mode: str = "constant",
        interpolation: str = "optimise",
    ) -> Image:
        """Get a tile at a given level and coordinate.

        Note that levels are in the reverse order of those in WSIReader.
        I.E. level 0 here corresponds to the lowest resolution whereas
        level 0 in WSIReader corresponds to the maximum resolution
        (baseline).

        Args:
            level (int):
                The pyramid level of the tile starting from 0
                (the whole slide in one tile, 0-0-0).
            x (int):
                The tile index in the x direction.
            y (int):
                The tile index in the y direction.
            pad_mode (str):
                Method for padding when reading areas outside of
                the input image. Default is constant (0 padding). This
                is passed to `read_func` which defaults to
                :func:`safe_padded_read`. See :func:`safe_padded_read`
                for supported pad modes. Setting to "none" or None will
                result in no padding being applied.
            interpolation (str):
                Interpolation mode to use.
                Defaults to optimise.
                Possible values are: linear, cubic, lanczos, nearest,
                area, optimise.
                Linear most closely matches OpenSlide.

        """
        if level < 0:
            raise IndexError
        if level > self.level_count:
            raise IndexError("Invalid level")

        scale = self.level_downsample(level)
        baseline_x = (x * self.tile_size * scale) - (self.overlap * scale)
        baseline_y = (y * self.tile_size * scale) - (self.overlap * scale)
        output_size = [self.output_tile_size] * 2
        coord = [baseline_x, baseline_y]
        if level < self.sub_tile_level_count:
            output_size = [2 ** level] * 2
            thumb = self.get_tile_thumb()
            thumb.thumbnail(output_size)
            return thumb
        slide_dimensions = np.array(self.wsi.info.slide_dimensions)
        if all(slide_dimensions < [baseline_x, baseline_y]):
            raise IndexError

        # Don't print out any warnings about interpolation etc.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rgb = self.wsi.read_rect(
                coord,
                size=output_size,
                resolution=1 / scale,
                units="baseline",
                pad_mode=pad_mode,
                interpolation=interpolation,
            )
        return Image.fromarray(rgb)

    def tile_path(self, level: int, x: int, y: int) -> Path:
        """Generate the path for a specified tile.

        Args:
            level (int):
                The pyramid level of the tile starting from 0
                (the whole slide in one tile, 0-0-0).
            x (int):
                The tile index in the x direction.
            y (int):
                The tile index in the y direction.

        Returns:
            Path: A pathlib path object with two parts.

        """
        raise NotImplementedError

    def __len__(self) -> int:
        return sum(
            np.prod(self.tile_grid_size(level)) for level in range(self.level_count)
        )

    def __iter__(self) -> Iterable:
        for level in range(self.level_count):
            for x, y in np.ndindex(self.tile_grid_size(level)):
                yield self.get_tile(level=level, x=x, y=y)


class DeepZoomGenerator(TilePyramidGenerator):
    r"""Pyramid tile generator following the DeepZoom format.

    Args:
        wsi (WSIReader):
            The WSI reader object. Must implement
            `tiatoolbox.wsicore.wsi_Reader.WSIReader.read_rect`.
        tile_size (int):
            The size of tiles to generate.
            Default is 256.
            Note that the output tile size will be
            :math:`\text{tile size} + 2 \times\text{overlap}`.
        downsample (int):
            The downsample factor between levels.
            Default is 2.
        tile_overlap (int):
            The number of extra pixel to add to each edge of the tile.
            Default is 0.

    """

    def __init__(
        self,
        wsi: WSIReader,
        tile_size: int = 254,
        downsample: int = 2,
        overlap: int = 1,
    ):
        super().__init__(wsi, tile_size, downsample, overlap)

    def dzi(
        self, dzi_format="xml", tile_format="jpg"
    ) -> Union[ET.Element, Dict[str, dict]]:
        """Generate and return DeepZoom XML metadata (.dzi).

        Arguments:
            format (str): Format of DZI file. Defaults to "XML" which
                returns an instance of ElementTree. Specifying "json",
                will return a dictionary which can be serialised to
                a valid DZI JSON file.
            tile_format (str): Image format file extension for tiles.
                Defaults to "jpg".

        Returns:
            ElementTree: XML DZI metadata.

        Example:
            >>> from xml.etree import ElementTree as ET
            >>> from tiatoolbox.wsicore.wsireader import get_wsireader
            >>> from tiatoolbox.tools.pyramid import DeepZoomGenerator
            >>>
            >>> slide = get_wsireader("CMU-1.svs")
            >>> dz = DeepZoomGenerator(slide)
            >>> dzi = dz.get_dzi()
            >>> print(ET.tostring(dzi, encoding="utf8").decode("utf8))
            <?xml version='1.0' encoding='utf8'?>
            <Image Format="jpg" Overlap="1" TileSize="256"
                xmlns="http://schemas.microsoft.com/deepzoom/2008">
            <Size Height="512" Width="512" /></Image>
        """
        width, height = self.wsi.info.slide_dimensions
        if dzi_format == "xml":
            root = ET.Element(
                "Image",
                {
                    "xmlns": "http://schemas.microsoft.com/deepzoom/2008",
                    "Format": tile_format,
                    "Overlap": str(self.overlap),
                    "TileSize": str(self.output_tile_size),
                },
            )
            ET.SubElement(root, "Size", {"Height": str(width), "Width": str(height)})
            return root
        if dzi_format == "json":
            json_dict = {
                "Image": {
                    "xmlns": "http://schemas.microsoft.com/deepzoom/2008",
                    "Format": tile_format,
                    "Overlap": str(self.overlap),
                    "TileSize": str(self.output_tile_size),
                    "Size": {
                        "Height": str(width),
                        "Width": str(height),
                    },
                }
            }
            return json_dict
        raise ValueError("Invalid format.")

    @property
    def sub_tile_level_count(self) -> int:
        """The number of levels which are a downsample of the whole image tile 0-0-0.

        Deepzoom levels start at 0 with a 1x1 pixel representing the
        whole image. The levels double in size until the region size is
        larger than a single tile.

        """
        return int(np.ceil(np.log2(self.output_tile_size)))

    @lru_cache(maxsize=None)
    def tile_path(self, level: int, x: int, y: int) -> Path:
        """Generate the DeepZoom path for a specified tile.

        Args:
            level (int):
                The pyramid level of the tile starting from 0
                (the whole slide in one tile, 0-0-0).
            x (int):
                The tile index in the x direction.
            y (int):
                The tile index in the y direction.

        Returns:
            Path: A pathlib path object with two parts.

        """
        path = Path(f"{level}") / f"{x}_{y}.jpg"
        return path

    def get_tile(
        self,
        level: int,
        x: int,
        y: int,
        pad_mode: str = "none",
        interpolation: str = "optimise",
    ) -> Image:
        return super().get_tile(
            level, x, y, pad_mode=pad_mode, interpolation=interpolation
        )


class ZoomifyGenerator(TilePyramidGenerator):
    r"""Pyramid tile generator with extra Zoomify specific methods.

    Zoomify splits tiles into groups of 256 (due to old file system
    limitations). The extra `tile_group` method here is for calculating
    these tile groups when generating tile paths.

    An old description of the Zoomify format can be found `here`_.

    .. _here:
        https://ecommons.cornell.edu/bitstream/handle/1813/5410/Introducing_Zoomify_Image.pdf

    Args:
        wsi (WSIReader):
            The WSI reader object. Must implement
            `tiatoolbox.wsicore.wsi_Reader.WSIReader.read_rect`.
        tile_size (int):
            The size of tiles to generate. Default is
            256. Note that the output tile size will be
            :math:`\text{tile size} + 2 \times\text{overlap}`.
        downsample (int):
            The downsample factor between levels.
            Default is 2.
        tile_overlap (int):
            The number of extra pixel to add to each
            edge of the tile. Default is 0.

    """

    @lru_cache(maxsize=None)
    def tile_group(self, level: int, x: int, y: int):
        """Find the tile group for a tile index.

        Tile groups are numbered from level 0 (tile 0-0-0) and increment
        every 256 tiles in ZXY axis order.

        Args:
            level (int):
                The pyramid level of the tile starting from 0
                (the whole slide in one tile, 0-0-0).
            x (int):
                The tile index in the x direction.
            y (int):
                The tile index in the y direction.

        Returns:
            int: The tile group for the specified tile.

        """
        grid_size = np.array(self.tile_grid_size(level))
        if any(grid_size <= [x, y]):
            raise IndexError
        cumsum = sum(np.prod(self.tile_grid_size(n)) for n in range(level))
        index_in_level = np.ravel_multi_index((y, x), self.tile_grid_size(level)[::-1])
        tile_index = cumsum + index_in_level
        tile_group = tile_index // 256
        return tile_group

    @lru_cache(maxsize=None)
    def tile_path(self, level: int, x: int, y: int) -> Path:
        """Generate the Zoomify path for a specified tile.

        Args:
            level (int): The pyramid level of the tile starting from 0
                (the whole slide in one tile, 0-0-0).
            x (int): The tile index in the x direction.
            y (int): The tile index in the y direction.

        Returns:
            Path: A pathlib path object with two parts.

        """
        g = self.tile_group(level, x, y)
        z = level
        path = Path(f"TileGroup{g}") / f"{z}-{x}-{y}.jpg"
        return path
