"""Tests for tile pyramid generation."""
from pathlib import Path
import re

import numpy as np
from openslide import ImageSlide
from openslide.deepzoom import DeepZoomGenerator as ODeepZoomGenerator
from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from skimage import data

from tiatoolbox.wsicore import wsireader
from tiatoolbox.tools import pyramid


def test_tilepyramidgenerator_overlap():
    """Test DeepZoomGenerator overlap with default parameters."""
    array = np.random.randint(0, 255, size=(512, 512), dtype=np.uint8)
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.DeepZoomGenerator(wsi)
    x0 = np.array(dz.get_tile(dz.level_count - 1, 0, 0))
    x1 = np.array(dz.get_tile(dz.level_count - 1, 1, 0))
    # Check the overlapping pixel columns match
    assert mean_squared_error(x1[:, 0], x0[:, -2]) < 1
    assert mean_squared_error(x1[:, 1], x0[:, -1]) < 1


def test_tilepyramidgenerator_openslide_consistency():
    """ "Check DeepZoomGenerator is consistent with OpenSlide."""
    array = data.camera()

    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.DeepZoomGenerator(wsi)

    img = Image.fromarray(array)
    osr = ImageSlide(img)
    odz = ODeepZoomGenerator(osr)

    for level in range(8, dz.level_count):
        w, h = dz.tile_grid_size(level)
        for i in range(w):
            for j in range(h):
                ox = odz.get_tile(level, (i, j))
                ox = np.array(ox)
                x = np.array(dz.get_tile(level, i, j, interpolation="optimise"))
                assert ox.shape == x.shape
                psnr = peak_signal_noise_ratio(ox, x)
                assert psnr == np.inf or psnr < 45
                assert mean_squared_error(ox, x) < 15


def test_deepzoomgenerator_dzi_xml():
    """Test DeepZoom DZI XML generation."""
    from xml.etree import ElementTree as ET
    from defusedxml.ElementTree import fromstring

    array = data.camera()

    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.DeepZoomGenerator(wsi)
    dzi = dz.dzi()
    assert isinstance(dzi, ET.Element)
    xml_string = ET.tostring(dzi, encoding="utf8")
    # Check for the xml decleration
    assert xml_string.startswith(b"<?xml")

    namespaces = {"dzi": "http://schemas.microsoft.com/deepzoom/2008"}
    parsed = fromstring(xml_string)
    assert namespaces["dzi"] in parsed.tag
    assert len(parsed.findall("dzi:Size", namespaces)) == 1


def test_deepzoomgenerator_dzi_json():
    """Test DeepZoom DZI JSON generation."""
    array = data.camera()

    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.DeepZoomGenerator(wsi)
    dzi = dz.dzi(dzi_format="json")

    assert isinstance(dzi, dict)
    assert len(dzi.keys()) == 1
    assert "Image" in dzi
    for key in ["Format", "Overlap", "TileSize", "xmlns", "Size"]:
        assert key in dzi["Image"]
    for key in ["Width", "Height"]:
        assert key in dzi["Image"]["Size"]


def test_deepzoom_tile_path():
    """Test DeepZooom tile path generation."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.DeepZoomGenerator(wsi)
    path = dz.tile_path(0, 1, 2)
    assert isinstance(path, Path)
    assert len(path.parts) == 2
    assert re.match(pattern=r"\d+", string=path.parts[0]) is not None
    assert re.match(pattern=r"\d+_\d+\.jpg", string=path.parts[1]) is not None


def test_zoomify_tile_path():
    """Test Zoomify tile path generation."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi)
    path = dz.tile_path(0, 0, 1)
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
    assert len(dz) == (4 * 4) + (2 * 2)


def test_zoomify_iter():
    """Test __iter__ for ZoomifyGenerator."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    dz = pyramid.ZoomifyGenerator(wsi, tile_size=256)
    for tile in dz:
        assert isinstance(tile, Image.Image)
        assert tile.size == (256, 256)
