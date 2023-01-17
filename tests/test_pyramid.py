"""Tests for tile pyramid generation."""
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from skimage import data
from skimage.metrics import peak_signal_noise_ratio

from tiatoolbox.tools import pyramid
from tiatoolbox.utils.image import imresize
from tiatoolbox.wsicore import wsireader


def test_zoomify_tile_path():
    """Test Zoomify tile path generation."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi)
    path = dz.tile_path(0, 0, 0)
    assert isinstance(path, Path)
    assert len(path.parts) == 2
    assert "TileGroup" in path.parts[0]
    assert re.match(pattern=r"TileGroup\d+", string=path.parts[0]) is not None
    assert re.match(pattern=r"\d+-\d+-\d+\.jpg", string=path.parts[1]) is not None


def test_zoomify_len():
    """Test __len__ for ZoomifyGenerator."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi, tile_size=256)
    assert len(dz) == (4 * 4) + (2 * 2) + 1


def test_zoomify_iter():
    """Test __iter__ for ZoomifyGenerator."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi, tile_size=256)
    for tile in dz:
        assert isinstance(tile, Image.Image)
        assert tile.size == (256, 256)


def test_tile_grid_size_invalid_level():
    """Test tile_grid_size for IndexError on invalid levels."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi, tile_size=256)
    with pytest.raises(IndexError):
        dz.tile_grid_size(level=-1)
    with pytest.raises(IndexError):
        dz.tile_grid_size(level=100)
    dz.tile_grid_size(level=0)


def test_get_tile_negative_level():
    """Test for IndexError on negative levels."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi, tile_size=256)
    with pytest.raises(IndexError):
        dz.get_tile(-1, 0, 0)


def test_get_tile_large_level():
    """Test for IndexError on too large a level."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi, tile_size=256)
    with pytest.raises(IndexError):
        dz.get_tile(100, 0, 0)


def test_get_tile_large_xy():
    """Test for IndexError on too large an xy index."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi, tile_size=256)
    with pytest.raises(IndexError):
        dz.get_tile(0, 100, 100)


def test_zoomify_tile_group_index_error():
    """Test IndexError for Zoomify tile groups."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi, tile_size=256)
    with pytest.raises(IndexError):
        dz.tile_group(0, 100, 100)


def test_zoomify_dump_options_combinations(tmp_path):  # noqa: CCR001
    """Test for no fatal errors on all option combinations for dump."""
    array = data.camera()
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi, tile_size=64)

    for container in [None, "zip", "tar"]:
        compression_methods = [None, "deflate", "gzip", "bz2", "lzma"]
        if container == "zip":
            compression_methods.remove("gzip")
        if container == "tar":
            compression_methods.remove("deflate")
        if container is None:
            compression_methods = [None]
        for compression in compression_methods:
            out_path = tmp_path / f"{compression}-pyramid"
            if container is not None:
                out_path = out_path.with_suffix(f".{container}")
            dz.dump(out_path, container=container, compression=compression)
            assert out_path.exists()


def test_zoomify_dump_compression_error(tmp_path):
    """Test ValueError is raised on invalid compression modes."""
    array = data.camera()
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi, tile_size=64)
    out_path = tmp_path / "pyramid_dump"

    with pytest.raises(ValueError, match="Unsupported compression for container None"):
        dz.dump(out_path, container=None, compression="deflate")

    with pytest.raises(ValueError, match="Unsupported compression for zip"):
        dz.dump(out_path, container="zip", compression="gzip")

    with pytest.raises(ValueError, match="Unsupported compression for tar"):
        dz.dump(out_path, container="tar", compression="deflate")


def test_zoomify_dump_container_error(tmp_path):
    """Test ValueError is raised on invalid containers."""
    array = data.camera()
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi, tile_size=64)
    out_path = tmp_path / "pyramid_dump"

    with pytest.raises(ValueError, match="Unsupported container"):
        dz.dump(out_path, container="foo")


def test_zoomify_dump(tmp_path):
    """Test dumping to directory."""
    array = data.camera()
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi, tile_size=64)
    out_path = tmp_path / "pyramid_dump"
    dz.dump(out_path)
    assert out_path.exists()
    assert len(list((out_path / "TileGroup0").glob("0-*"))) == 1
    assert Image.open(out_path / "TileGroup0" / "0-0-0.jpg").size == (64, 64)


def test_get_thumb_tile():
    """Test getting a thumbnail tile (whole WSI in one tile)."""
    array = data.camera()
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi, tile_size=224)
    thumb = dz.get_thumb_tile()
    assert thumb.size == (224, 224)
    cv2_thumb = imresize(array, output_size=(224, 224))
    psnr = peak_signal_noise_ratio(cv2_thumb, np.array(thumb.convert("L")))
    assert np.isinf(psnr) or psnr < 40


def test_sub_tile_levels():
    """Test sub-tile level generation."""
    array = data.camera()
    wsi = wsireader.VirtualWSIReader(array)

    class MockTileGenerator(pyramid.TilePyramidGenerator):
        def tile_path(self, level: int, x: int, y: int) -> Path:  # skipcq: PYL-R0201
            return Path(level, x, y)

        @property
        def sub_tile_level_count(self):
            return 1

    dz = MockTileGenerator(wsi, tile_size=224)

    tile = dz.get_tile(0, 0, 0)
    assert tile.size == (112, 112)
