"""Tests for tile pyramid generation."""
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
    # array = np.random.randint(0, 255, size=(256, 512), dtype=np.uint8)
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


# %%
