"""Tile pyramid generation in standard formats.

Included methods are DeepZoom and Zoomify in addition to a generic
method.

These are generally intended for serialisation or streaming via a web
UI. The `get_tile` method returns a Pillow Image object which can be
easily serialised via the use of an io.BytesIO object or saved directly
to disk.

"""

import tarfile
import time
import warnings
import zipfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Iterable, Tuple, Union

import defusedxml
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from tiatoolbox.utils.transforms import imresize, locsize2bounds
from tiatoolbox.wsicore.wsireader import WSIReader, WSIMeta
from tiatoolbox.annotation.storage import AnnotationStore
from tiatoolbox.utils.visualization import AnnotationRenderer

defusedxml.defuse_stdlib()


class TilePyramidGenerator:
    r"""Generic tile pyramid generator with sensible defaults.

    Args:
        wsi (WSIReader):
            The WSI reader object. Must implement
            `tiatoolbox.wsicore.wsi_Reader.WSIReader.read_rect`.
        tile_size (int):
            The size of tiles to generate. Default is 256. Note that the
            output tile size will be :math:`\text{tile size} + 2
            \times\text{overlap}`.
        downsample (int):
            The downsample factor between levels. Default is 2.
        overlap (int):
            The number of extra pixel to add to each edge of the tile.
            Default is 0.

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

        This is equivalent to :math:`\text{tile size} + 2*\text{overlay}`.

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
            level (int):
                The level to calculate the dimensions for.

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
            level (int):
                The level to calculate the grid size for.

        """
        if level < 0 or level >= self.level_count:
            raise IndexError("Invalid level.")
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
        wsi_to_tile_ratio = np.divide(self.wsi.info.slide_dimensions, self.tile_size)
        # Levels where a tile contains only part of the wsi
        super_level_count = np.ceil(np.log2(wsi_to_tile_ratio)).max()
        total_level_count = super_level_count + 1 + self.sub_tile_level_count
        return int(total_level_count)

    def get_thumb_tile(self) -> Image:
        """Return a thumbnail which fits the whole slide in one tile.

        The thumbnail output size has the longest edge equal to the tile
        size. The other edge preserves the original aspect ratio.

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
                The pyramid level of the tile starting from 0 (the whole
                slide in one tile, 0-0-0).
            x (int):
                The tile index in the x direction.
            y (int):
                The tile index in the y direction.
            pad_mode (str):
                Method for padding when reading areas outside of the
                input image. Default is constant (0 padding). This is
                passed to `read_func` which defaults to
                :func:`safe_padded_read`. See :func:`safe_padded_read`
                for supported pad modes. Setting to "none" or None will
                result in no padding being applied.
            interpolation (str):
                Interpolation mode to use. Defaults to optimise.
                Possible values are: linear, cubic, lanczos, nearest,
                area, optimise. Linear most closely matches OpenSlide.

        Returns:
            Image:
                Pillow image of the tile.

        Example:
            >>> from tiatoolbox.tools.pyramid import TilePyramidGenerator
            >>> from tiatoolbox.wsicore.wsireader import get_wsireader
            >>> wsi = get_wsireader("sample.svs")
            >>> tile_generator = TilePyramidGenerator(
            ...   wsi=reader,
            ...   tile_size=256,
            ... )
            >>> tile_0_0_0 = tile_generator.get_tile(level=0, x=0, y=0)

        """
        if level < 0:
            raise IndexError
        if level > self.level_count:
            raise IndexError("Invalid level.")

        scale = self.level_downsample(level)
        baseline_x = (x * self.tile_size * scale) - (self.overlap * scale)
        baseline_y = (y * self.tile_size * scale) - (self.overlap * scale)
        output_size = [self.output_tile_size] * 2
        coord = [baseline_x, baseline_y]
        if level < self.sub_tile_level_count:
            output_size = self.output_tile_size // 2 ** (
                self.sub_tile_level_count - level
            )
            output_size = np.repeat(output_size, 2).astype(int)
            thumb = self.get_thumb_tile()
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
        alph=255-np.all(rgb==255,axis=2).astype('uint8')*255
        rgb=np.dstack((rgb,alph))
        return Image.fromarray(rgb)

    def tile_path(self, level: int, x: int, y: int) -> Path:
        """Generate the path for a specified tile.

        Args:
            level (int):
                The pyramid level of the tile starting from 0 (the whole
                slide in one tile, 0-0-0).
            x (int):
                The tile index in the x direction.
            y (int):
                The tile index in the y direction.

        Returns:
            Path:
                A pathlib path object with two parts.

        """
        raise NotImplementedError

    def dump(  # noqa: CCR001
        self, path: Union[str, Path], container=None, compression=None
    ):
        """Write all tiles to disk.

        Arguments:
            path (str or Path):
                The path to write the tiles to.
            container (str):
                Container to use. Defaults to None which saves to a
                directory. Possible values are "zip", "tar".
            compression (str):
                Compression method. Defaults to None. Possible values
                are None, "deflate", "gzip", "bz2", "lzma". Note that
                tar does not support deflate and zip does not support
                gzip.

        Examples:
            >>> from tiatoolbox.tools.pyramid import TilePyramidGenerator
            >>> from tiatoolbox.wsicore.wsireader import get_wsireader
            >>> wsi = get_wsireader("sample.svs")
            >>> tile_generator = TilePyramidGenerator(
            ...   wsi=reader,
            ...   tile_size=256,
            ... )
            >>>  tile_generator.dump(
            ...    path="sample.gz.zip",
            ...    container="zip",
            ...    compression="gzip",
            ...  )

        """
        path = Path(path)
        if container not in [None, "zip", "tar"]:
            raise ValueError("Unsupported container.")

        if container is None:
            path.mkdir(parents=False)
            if compression is not None:
                raise ValueError("Unsupported compression for container None.")

            def save_tile(tile_path: Path, tile: Image.Image) -> None:
                """Write the tile to the output directory."""
                full_path = path / tile_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                tile.save(full_path)

        elif container == "zip":
            compression2enum = {
                None: zipfile.ZIP_STORED,
                "deflate": zipfile.ZIP_DEFLATED,
                "bz2": zipfile.ZIP_BZIP2,
                "lzma": zipfile.ZIP_LZMA,
            }
            if compression not in compression2enum:
                raise ValueError("Unsupported compression for zip.")

            archive = zipfile.ZipFile(
                path, mode="w", compression=compression2enum[compression]
            )

            def save_tile(tile_path: Path, tile: Image.Image) -> None:
                """Write the tile to the output tar."""
                bio = BytesIO()
                tile.save(bio, format="jpeg")
                bio.seek(0)
                data = bio.read()
                archive.writestr(
                    str(tile_path),
                    data,
                    compress_type=compression2enum[compression],
                )

        else:  # container == "tar":
            compression2mode = {
                None: "w",
                "gzip": "w:gz",
                "bz2": "w:bz2",
                "lzma": "w:xz",
            }
            if compression not in compression2mode:
                raise ValueError("Unsupported compression for tar.")

            archive = tarfile.TarFile.open(path, mode=compression2mode[compression])

            def save_tile(tile_path: Path, tile: Image.Image) -> None:
                """Write the tile to the output zip."""
                bio = BytesIO()
                tile.save(bio, format="jpeg")
                bio.seek(0)
                tar_info = tarfile.TarInfo(name=str(tile_path))
                tar_info.mtime = time.time()
                tar_info.size = bio.tell()
                archive.addfile(tarinfo=tar_info, fileobj=bio)

        for level in range(self.level_count):
            for x, y in np.ndindex(self.tile_grid_size(level)):
                tile = self.get_tile(level=level, x=x, y=y)
                tile_path = self.tile_path(level, x, y)
                save_tile(tile_path, tile)

        if container is not None:
            archive.close()

    def __len__(self) -> int:
        return sum(
            np.prod(self.tile_grid_size(level)) for level in range(self.level_count)
        )

    def __iter__(self) -> Iterable:
        for level in range(self.level_count):
            for x, y in np.ndindex(self.tile_grid_size(level)):
                yield self.get_tile(level=level, x=x, y=y)


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
            The size of tiles to generate. Default is 256. Note that the
            output tile size will be :math:`\text{tile size} + 2
            \times\text{overlap}`.
        downsample (int):
            The downsample factor between levels. Default is 2.
        tile_overlap (int):
            The number of extra pixel to add to each edge of the tile.
            Default is 0.

    """

    @lru_cache(maxsize=None)
    def tile_group(self, level: int, x: int, y: int):
        """Find the tile group for a tile index.

        Tile groups are numbered from level 0 (tile 0-0-0) and increment
        every 256 tiles in ZXY axis order.

        Args:
            level (int):
                The pyramid level of the tile starting from 0 (the whole
                slide in one tile, 0-0-0).
            x (int):
                The tile index in the x direction.
            y (int):
                The tile index in the y direction.

        Returns:
            int:
                The tile group for the specified tile.

        """
        grid_size = np.array(self.tile_grid_size(level))
        if any(grid_size <= [x, y]):
            raise IndexError
        cumsum = sum(np.prod(self.tile_grid_size(n)) for n in range(level))
        index_in_level = np.ravel_multi_index((y, x), self.tile_grid_size(level)[::-1])
        tile_index = cumsum + index_in_level
        return tile_index // 256  # the tile group

    @lru_cache(maxsize=None)
    def tile_path(self, level: int, x: int, y: int) -> Path:
        """Generate the Zoomify path for a specified tile.

        Args:
            level (int):
                The pyramid level of the tile starting from 0 (the whole
                slide in one tile, 0-0-0).
            x (int):
                The tile index in the x direction.
            y (int):
                The tile index in the y direction.

        Returns:
            Path:
                A pathlib path object with two parts.

        """
        g = self.tile_group(level, x, y)
        z = level
        return Path(f"TileGroup{g}") / f"{z}-{x}-{y}.jpg"


class AnnotationTileGenerator(ZoomifyGenerator):
    r"""Tile generator using an AnnotationRenderer to render tiles
    showing annotations held in an AnnotationStore

    Args:
        info (WSIMeta):
            An WSIMeta Object storing the metadata of the slide this
            generator is rendering tiles for
        Store (AnnotationStore):
            An AnnotationStore Object containing annotations to be rendered
            for given slide
        renderer (AnnotationRenderer):
            An AnnotationRenderer Object which will render annotations belonging
            to a tile according to specified parameters
        tile_size (int):
            The size of tiles to generate. Default is 256. Note that the
            output tile size will be :math:`\text{tile size} + 2
            \times\text{overlap}`.
        downsample (int):
            The downsample factor between levels. Default is 2.
        overlap (int):
            The number of extra pixel to add to each edge of the tile.
            Default is 0.

    """

    def __init__(
        self,
        info: WSIMeta,
        store: AnnotationStore,
        renderer: AnnotationRenderer = None,
        tile_size: int = 256,
        downsample: int = 2,
        overlap: int = 0,
    ):
        self.info = info
        self.store = store
        if renderer is None:
            renderer = AnnotationRenderer()
        self.renderer = renderer
        self.tile_size = tile_size
        self.overlap = overlap
        self.downsample = downsample

    def get_thumb_tile(self) -> Image:
        """Return a thumbnail which fits the whole slide in one tile.

        The thumbnail output size has the longest edge equal to the tile
        size. The other edge preserves the original aspect ratio.

        """
        slide_dims = np.array(self.info.slide_dimensions)
        tile_dim = self.tile_size + self.overlap
        scale = self.level_downsample(self.info.level_count - 1)
        out_dims = np.round(slide_dims / slide_dims.max() * tile_dim).astype(int)
        bounds = (0, 0, *slide_dims)
        rgb = np.zeros((*out_dims, 4), dtype=np.uint8)
        bound_geom = Polygon.from_bounds(*bounds)
        rgb = self.render_annotations(rgb, bound_geom, scale, (0, 0))
        return Image.fromarray(rgb)

    @lru_cache(maxsize=None)
    def level_dimensions(self, level: int) -> Tuple[int, int]:
        """The total pixel dimensions of the tile pyramid at a given level.

        Args:
            level (int):
                The level to calculate the dimensions for.

        """
        baseline_dims = self.info.slide_dimensions
        level_dims = np.ceil(
            np.divide(baseline_dims, self.level_downsample(level))
        ).astype(int)
        return tuple(level_dims)

    @property
    def level_count(self) -> int:
        """Number of levels in the tile pyramid.

        The number of levels is such that level_count - 1 is a 1:1 of
        the slide baseline resolution (level 0 of the WSI).

        """
        wsi_to_tile_ratio = np.divide(self.info.slide_dimensions, self.tile_size)
        # Levels where a tile contains only part of the wsi
        super_level_count = np.ceil(np.log2(wsi_to_tile_ratio)).max()
        total_level_count = super_level_count + 1 + self.sub_tile_level_count
        return int(total_level_count)

    def get_tile(
        self,
        level: int,
        x: int,
        y: int,
    ) -> Image:
        """Render a tile at a given level and coordinate.

        Note that levels are in the reverse order of those in WSIReader.
        I.E. level 0 here corresponds to the lowest resolution whereas
        level 0 in WSIReader corresponds to the maximum resolution
        (baseline).

        Args:
            level (int):
                The pyramid level of the tile starting from 0 (the whole
                slide in one tile, 0-0-0).
            x (int):
                The tile index in the x direction.
            y (int):
                The tile index in the y direction.

        Returns:
            Image:
                Pillow image of the tile.

        Example:
            >>> from tiatoolbox.tools.pyramid import AnnotationTileGenerator
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> from tiatoolbox.annotation.storage import SQLiteStore
            >>> wsi = WSIReader.open("sample.svs")
            >>> SQ=SQLiteStore.from_geojson(geo_path)
            >>> tile_generator = AnnotationTileGenerator(
            ...   info=wsi.info,
            ...   store=SQ,
            ... )
            >>> tile_0_0_0 = tile_generator.get_tile(level=0, x=0, y=0)

        """
        if level < 0:
            raise IndexError
        if level > self.level_count:
            raise IndexError("Invalid level.")

        scale = self.level_downsample(level)
        baseline_x = (x * self.tile_size * scale) - (self.overlap * scale)
        baseline_y = (y * self.tile_size * scale) - (self.overlap * scale)
        output_size = [self.output_tile_size] * 2
        coord = [baseline_x, baseline_y]
        if level < self.sub_tile_level_count:
            output_size = self.output_tile_size // 2 ** (
                self.sub_tile_level_count - level
            )
            output_size = np.repeat(output_size, 2).astype(int)
            thumb = self.get_thumb_tile()
            thumb.thumbnail(output_size)
            return thumb
        slide_dimensions = np.array(self.info.slide_dimensions)
        if all(slide_dimensions < [baseline_x, baseline_y]):
            raise IndexError

        rgb = np.zeros((*output_size, 4), dtype=np.uint8)
        bounds = locsize2bounds(coord, [self.tile_size * scale] * 2)
        bound_geom = Polygon.from_bounds(*bounds)
        rgb = self.render_annotations(rgb, bound_geom, scale, coord)

        return Image.fromarray(rgb)

    def render_annotations(self, rgb, bound_geom, scale, tl):
        """get annotations as bbox or geometry according to zoom level,
        and decimate large collections of small annotations if appropriate"""
        r = self.renderer
        big_thresh = 0.0025 * (self.tile_size * scale) ** 2
        decimate = int(scale / self.renderer.max_scale) + 1
        if scale > 100:
            decimate = decimate * 2

        if scale > self.renderer.max_scale:
            anns_dict = self.store.bquery(
                bound_geom.bounds, 
                self.renderer.where,
            )
            print(f'len annotations dict is: {len(anns_dict)}')
            if len(anns_dict) < 40:
                decimate = int(len(anns_dict) / 20) + 1
            i = 0
            for key, ann in anns_dict.items():
                i += 1
                if ann.geometry.area > big_thresh:
                    ann = self.store[key]
                    ann_bounded = r.get_bounded(ann, bound_geom)
                    if ann_bounded.is_empty:
                        #only bbox not actual geom was inside the tile, so ignore
                        continue
                    if ann_bounded.geom_type == "Polygon":
                        r.render_poly(rgb, ann, ann_bounded, tl, scale)
                    elif ann_bounded.geom_type == "LineString":
                        r.render_line(rgb, ann, ann_bounded, tl, scale)
                    else:
                        print(f"unknown geometry: {ann_bounded.geom_type}")
                    continue
                if i % decimate == 0:
                    if ann.geometry.geom_type == "Point":
                        r.render_pt(rgb, ann, tl, scale)
                        continue
                    ann_bounded = r.get_bounded(ann, bound_geom)
                    if ann_bounded.geom_type == "Polygon":
                        r.render_rect(rgb, ann, ann_bounded, tl, scale)
                    elif ann_bounded.geom_type == "LineString":
                        #ann_bounded = ann.geometry.intersection(bound_geom)
                        r.render_line(rgb, ann, ann_bounded, tl, scale)
                    else:
                        print(f"unknown geometry: {ann_bounded.geom_type}")
        else:
            anns = self.store.query(bound_geom.bounds, self.renderer.where)
            for ann in anns:
                if ann.geometry.geom_type == "Point":
                    r.render_pt(rgb, ann, tl, scale)
                    continue
                ann_bounded = r.get_bounded(ann, bound_geom)
                if ann_bounded.geom_type == "Polygon":
                    r.render_poly(rgb, ann, ann_bounded, tl, scale)
                elif ann_bounded.geom_type == "LineString":
                    r.render_line(rgb, ann, ann_bounded, tl, scale)
                elif ann_bounded.geom_type == "MultiPolygon":
                    r.render_multipoly(rgb, ann, ann_bounded, tl, scale)
                elif ann_bounded.geom_type == "GeometryCollection":
                    #print(f"unknown geometry: {ann_bounded.geom_type}: {[g.geom_type for g in ann_bounded.geoms]}")
                    pass
                else:
                    print(f"unknown geometry: {ann_bounded.geom_type}")

        return rgb
