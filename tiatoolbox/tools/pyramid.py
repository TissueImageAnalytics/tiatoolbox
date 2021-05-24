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
from xml.etree import ElementTree
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from tiatoolbox.wsicore.wsireader import WSIReader


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
        cache_size (int): The maximum number of recent tiles to
            cache. Default is 1024.
    """

    def __init__(
        self,
        wsi: WSIReader,
        tile_size: int = 256,
        downsample: int = 2,
        overlap: int = 0,
        cache_size=1024,
    ):
        self.wsi = wsi
        self.tile_size = tile_size
        self.overlap = overlap
        self.cache_size = cache_size
        self.downsample = downsample

        # Set up LRU cache for the tiles themselves
        get_tile = self.get_tile

        @lru_cache(maxsize=self.cache_size)
        def _get_tile(*args, **kwargs):
            """Thin chached wrapper around get_tile."""
            return get_tile(*args, **kwargs)

        self.get_tile = _get_tile

    @property
    def output_tile_size(self) -> int:
        r"""The size of the tile which will be returned.

        This is eqivalent to :math:`\text{tile size} + 2*\text{overlay}`.
        """
        return self.tile_size + 2 * self.overlap

    @lru_cache(maxsize=None)
    def level_downsample(self, level: int) -> float:
        """Find the downsample factor for a level."""
        return 2 ** (self.level_count - level)

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
        """The width and height of the minimal grid of tiles to cover the slide.

        Args:
            level (int): The level to calculate the grid size for.
        """
        return tuple(
            np.ceil(np.divide(self.level_dimensions(level), self.tile_size)).astype(int)
        )

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
        )

    def get_tile(self, level: int, x: int, y: int) -> Image:
        """Get a tile at a given level and coordinate.

        Note that levels are in the reverse order of those in WSIReader.
        I.E. level 0 here corresponds to the lowest resolution whereas
        level 0 in WSIReader corresponds to the maximum resolution
        (baseline).

        Args:
            level (int): The pyramid level of the tile starting from 0
                (the whole slide in one tile, 0-0-0).
            x (int): The tile index in the x direction.
            y (int): The tile index in the y direction.
        """
        if level < 0:
            raise IndexError
        if level >= self.levels:
            raise IndexError
        scale = self.level_downsample(level)
        baseline_x = (x * self.tile_size * scale) - (self.overlap * scale)
        baseline_y = (y * self.tile_size * scale) - (self.overlap * scale)
        slide_dimensions = np.array(self.wsi.info.slide_dimensions)
        if all(slide_dimensions < [baseline_x, baseline_y]):
            raise IndexError
        # Don't print out any warnings about interpolation etc.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rgb = self.wsi.read_rect(
                (baseline_x, baseline_y),
                size=[self.output_tile_size] * 2,
                resolution=1 / scale,
                units="baseline",
            )
        return Image.fromarray(rgb)

    def tile_path(self, level: int, x: int, y: int) -> Path:
        """Generate the path for a specified tile.

        Args:
            level (int): The pyramid level of the tile starting from 0
                (the whole slide in one tile, 0-0-0).
            x (int): The tile index in the x direction.
            y (int): The tile index in the y direction.

        Returns:
            Path: A pathlib path object with two parts.
        """
        raise NotImplementedError


class DeepZoomGenerator(TilePyramidGenerator):
    r"""Pyramid tile generator following the DeepZoom format.

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
        cache_size (int): The maximum number of recent tiles to
            cache. Default is 1024.
    """

    def __init__(
        self,
        wsi: WSIReader,
        tile_size: int = 254,
        downsample: int = 2,
        overlap: int = 1,
        cache_size=1024,
    ):
        super().__init__(wsi, tile_size, downsample, overlap, cache_size)

    def get_dzi(self) -> ElementTree:
        """Generate and return DeepZoom XML metadata (.dzi).

        Returns:
            ElementTree: XML DZI metadata.
        """
        # TODO: Add DZI metadata generation
        raise NotImplementedError

    def tile_path(self, level: int, x: int, y: int) -> Path:
        """Generate the DeepZoom path for a specified tile.

        Args:
            level (int): The pyramid level of the tile starting from 0
                (the whole slide in one tile, 0-0-0).
            x (int): The tile index in the x direction.
            y (int): The tile index in the y direction.

        Returns:
            Path: A pathlib path object with two parts.
        """
        # TODO: Add DeepZoom path generation
        raise NotImplementedError


class ZoomifyGenerator(TilePyramidGenerator):
    r"""Pyramid tile generator with extra Zoomify specific methods.

    Zoomify splits tiles into groups of 256 (due to old file system
    limitations). The extra `tile_group` method here is for calculating
    these tile groups when generating tile paths.

    An old description of the Zoomify format can be found `here`_.

    .. _here:
        https://ecommons.cornell.edu/bitstream/handle/1813/5410/Introducing_Zoomify_Image.pdf

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
        cache_size (int): The maximum number of recent tiles to
            cache. Default is 1024.
    """

    @lru_cache(maxsize=None)
    def tile_group(self, level: int, x: int, y: int):
        """Find the tile group for a tile index.

        Tile groups are numbered from level 0 (tile 0-0-0) and increment
        every 256 tiles in ZXY axis order.

        Args:
            level (int): The pyramid level of the tile starting from 0
                (the whole slide in one tile, 0-0-0).
            x (int): The tile index in the x direction.
            y (int): The tile index in the y direction.

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
