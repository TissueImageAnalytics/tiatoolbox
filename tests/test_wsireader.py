"""Tests for reading whole-slide images."""

import os
import pathlib
import random
import re
import shutil
from time import time

# When no longer supporting Python <3.9 this should be collections.abc.Iterable
from typing import Iterable

import cv2
import numpy as np
import pytest
import zarr
from click.testing import CliRunner
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, disk, remove_small_objects
from skimage.registration import phase_cross_correlation

from tiatoolbox import cli, rcParam, utils
from tiatoolbox.utils.exceptions import FileNotSupported
from tiatoolbox.utils.misc import imread
from tiatoolbox.utils.transforms import imresize, locsize2bounds
from tiatoolbox.wsicore import wsireader
from tiatoolbox.wsicore.wsireader import (
    ArrayView,
    OmnyxJP2WSIReader,
    OpenSlideWSIReader,
    TIFFWSIReader,
    VirtualWSIReader,
)

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------

NDPI_TEST_TISSUE_BOUNDS = (30400, 11810, 30912, 12322)
NDPI_TEST_TISSUE_LOCATION = (30400, 11810)
NDPI_TEST_TISSUE_SIZE = (512, 512)

SVS_TEST_TISSUE_BOUNDS = (1000, 2000, 2000, 3000)
SVS_TEST_TISSUE_LOCATION = (1000, 2000)
SVS_TEST_TISSUE_SIZE = (1000, 1000)

JP2_TEST_TISSUE_BOUNDS = (32768, 42880, 33792, 43904)
JP2_TEST_TISSUE_LOCATION = (32768, 42880)
JP2_TEST_TISSUE_SIZE = (1024, 1024)

# -------------------------------------------------------------------------------------
# Generate Parameterized Tests
# -------------------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    """Generate (parameterize) test scenarios.

    Adapted from pytest documentation. For more information on
    parameterized tests see:
    https://docs.pytest.org/en/6.2.x/example/parametrize.html#a-quick-port-of-testscenarios

    """
    # Return if the test is not part of a class
    if metafunc.cls is None:
        return
    idlist = []
    argvalues = []
    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append([x[1] for x in items])
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


# -------------------------------------------------------------------------------------
# Utility Test Functions
# -------------------------------------------------------------------------------------


def _get_temp_folder_path(prefix="temp"):
    """Return unique temp folder path"""
    new_dir = os.path.join(rcParam["TIATOOLBOX_HOME"], f"{prefix}-{int(time())}")
    return new_dir


def strictly_increasing(sequence: Iterable) -> bool:
    """Return True if sequence is strictly increasing.

    Args:
        sequence (Iterable): Sequence to check.

    Returns:
        bool: True if strictly increasing.
    """
    return all(a < b for a, b in zip(sequence, sequence[1:]))


def strictly_decreasing(sequence: Iterable) -> bool:
    """Return True if sequence is strictly decreasing.

    Args:
        sequence (Iterable): Sequence to check.


    Returns:
        bool: True if strictly decreasing.
    """
    return all(a > b for a, b in zip(sequence, sequence[1:]))


def read_rect_objective_power(wsi, location, size):
    """Read rect objective helper."""
    for objective_power in [20, 10, 5, 2.5, 1.25]:
        im_region = wsi.read_rect(
            location, size, resolution=objective_power, units="power"
        )

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert im_region.shape == (*size[::-1], 3)


def read_bounds_mpp(wsi, bounds, size, jp2=False):
    """Read bounds mpp helper."""
    slide_mpp = wsi.info.mpp
    for factor in range(1, 10):
        mpp = slide_mpp * factor
        downsample = mpp / slide_mpp

        im_region = wsi.read_bounds(bounds, resolution=mpp, units="mpp")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        expected_output_shape = tuple(
            np.round((np.array(size[::-1]) / downsample)).astype(int)
        )
        if jp2:
            assert im_region.shape[:2] == pytest.approx(expected_output_shape, abs=1)
        else:
            assert im_region.shape[:2] == expected_output_shape
        assert im_region.shape[2] == 3


def read_bounds_objective_power(wsi, slide_power, bounds, size, jp2=False):
    """Read bounds objective power helper."""
    for objective_power in [20, 10, 5, 2.5, 1.25]:
        downsample = slide_power / objective_power

        im_region = wsi.read_bounds(
            bounds,
            resolution=objective_power,
            units="power",
        )

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        expected_output_shape = tuple(
            np.round((np.array(size[::-1]) / downsample)).astype(int)
        )
        if jp2:
            assert im_region.shape[:2] == pytest.approx(
                expected_output_shape[:2], abs=1
            )
        else:
            assert im_region.shape[:2] == expected_output_shape
        assert im_region.shape[2] == 3


def read_bounds_level_consistency(wsi, bounds):
    """Read bounds level consistency helper.

    Reads the same region at each stored resolution level and compares
    the resulting image using phase cross correlation to check that they
    are aligned.
    """
    # Avoid testing very small levels (e.g. as in Omnyx JP2) becuase
    # MSE for very small levels is noisy.
    levels_to_test = [
        n for n, downsample in enumerate(wsi.info.level_downsamples) if downsample <= 32
    ]
    imgs = [wsi.read_bounds(bounds, level, "level") for level in levels_to_test]
    smallest_size = imgs[-1].shape[:2][::-1]
    resized = [cv2.resize(img, smallest_size) for img in imgs]
    # Some blurring applied to account for changes in sharpness arising
    # from interpolation when calculating the downsampled levels. This
    # adds some tolerance for the comparison.
    blurred = [cv2.GaussianBlur(img, (5, 5), cv2.BORDER_REFLECT) for img in resized]
    as_float = [img.astype(np.float) for img in blurred]

    # Pair-wise check resolutions for mean squared error
    for i, a in enumerate(as_float):
        for b in as_float[i + 1 :]:
            _, error, phase_diff = phase_cross_correlation(a, b)
            assert phase_diff < 0.125
            assert error < 0.125


def command_line_slide_thumbnail(runner, sample, tmp_path, mode="save"):
    """Command line slide thumbnail helper."""
    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--img-input",
            str(pathlib.Path(sample)),
            "--mode",
            mode,
            "--output-path",
            str(pathlib.Path(tmp_path).joinpath("slide_thumb.jpg")),
        ],
    )

    assert slide_thumb_result.exit_code == 0
    if mode == "save":
        assert pathlib.Path(tmp_path).joinpath("slide_thumb.jpg").is_file()


# -------------------------------------------------------------------------------------
# Utility Test Classes
# -------------------------------------------------------------------------------------


class DummyMutableOpenSlideObject:
    """Dummy OpenSlide object with mutable properties."""

    def __init__(self, openslide_obj) -> None:
        self.openslide_obj = openslide_obj
        self._properties = dict(openslide_obj.properties)

    def __getattr__(self, name: str):
        return getattr(self.openslide_obj, name)

    @property
    def properties(self):
        """Return the fake properties."""
        return self._properties


# -------------------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------------------


def test_wsireader_slide_info(sample_svs, tmp_path):
    """Test for slide_info in WSIReader class as a python function."""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(sample_svs).parent),
        file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    slide_param = wsi.info
    out_path = tmp_path / slide_param.file_path.with_suffix(".yaml").name
    utils.misc.save_yaml(slide_param.as_dict(), out_path)


def test_wsireader_slide_info_cache(sample_svs):
    """Test for caching slide_info in WSIReader class as a python function."""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(sample_svs).parent),
        file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    info = wsi.info
    cached_info = wsi.info
    assert info.as_dict() == cached_info.as_dict()


def relative_level_scales_baseline(wsi):
    """Relative level scales for pixels per baseline pixel."""
    level_scales = wsi._relative_level_scales(0.125, "baseline")
    level_scales = np.array(level_scales)
    downsamples = np.array(wsi.info.level_downsamples)
    expected = downsamples * 0.125
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test__relative_level_scales_openslide_baseline(sample_ndpi):
    """Test openslide relative level scales for pixels per baseline pixel."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    relative_level_scales_baseline(wsi)


def test__relative_level_scales_jp2_baseline(sample_jp2):
    """Test jp2 relative level scales for pixels per baseline pixel."""
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    relative_level_scales_baseline(wsi)


def test__relative_level_scales_openslide_mpp(sample_ndpi):
    """Test openslide calculation of relative level scales for mpp."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    level_scales = wsi._relative_level_scales(0.5, "mpp")
    level_scales = np.array(level_scales)
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert all(level_scales[0] == wsi.info.mpp / 0.5)


def test__relative_level_scales_jp2_mpp(sample_jp2):
    """Test jp2 calculation of relative level scales for mpp."""
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    level_scales = wsi._relative_level_scales(0.5, "mpp")
    level_scales = np.array(level_scales)
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert all(level_scales[0] == wsi.info.mpp / 0.5)


def relative_level_scales_power(wsi):
    """Calculation of relative level scales for objective power."""
    level_scales = wsi._relative_level_scales(wsi.info.objective_power, "power")
    level_scales = np.array(level_scales)
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert np.array_equal(level_scales[0], [1, 1])
    downsamples = np.array(wsi.info.level_downsamples)
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], downsamples)


def test__relative_level_scales_openslide_power(sample_ndpi):
    """Test openslide calculation of relative level scales for objective power."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    relative_level_scales_power(wsi)


def test__relative_level_scales_jp2_power(sample_jp2):
    """Test jp2 calculation of relative level scales for objective power."""
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    relative_level_scales_power(wsi)


def relative_level_scales_level(wsi):
    """Calculation of relative level scales for level."""
    level_scales = wsi._relative_level_scales(3, "level")
    level_scales = np.array(level_scales)
    assert np.array_equal(level_scales[3], [1, 1])
    downsamples = np.array(wsi.info.level_downsamples)
    expected = downsamples / downsamples[3]
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test__relative_level_scales_openslide_level(sample_ndpi):
    """Test openslide calculation of relative level scales for level."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    relative_level_scales_level(wsi)


def test__relative_level_scales_jp2_level(sample_jp2):
    """Test jp2 calculation of relative level scales for level."""
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    relative_level_scales_level(wsi)


def relative_level_scales_float(wsi):
    """Calculation of relative level scales for fractional level."""
    level_scales = wsi._relative_level_scales(1.5, "level")
    level_scales = np.array(level_scales)
    assert level_scales[0] == pytest.approx([1 / 3, 1 / 3])
    downsamples = np.array(wsi.info.level_downsamples)
    expected = downsamples / downsamples[0] * (1 / 3)
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test__relative_level_scales_openslide_level_float(sample_ndpi):
    """Test openslide calculation of relative level scales for fractional level."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    relative_level_scales_float(wsi)


def test__relative_level_scales_jp2_level_float(sample_jp2):
    """Test jp2 calculation of relative level scales for fractional level."""
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    relative_level_scales_float(wsi)


def test__relative_level_scales_invalid_units(sample_svs):
    """Test _relative_level_scales with invalid units."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    with pytest.raises(ValueError):
        wsi._relative_level_scales(1.0, "gibberish")


def test__relative_level_scales_no_mpp():
    """Test _relative_level_scales objective when mpp is None."""

    class DummyWSI:
        """Mock WSIReader for testing."""

        _relative_level_scales = wsireader.WSIReader._relative_level_scales

        @property
        def info(self):
            return wsireader.WSIMeta((100, 100), axes="YXS")

    wsi = DummyWSI()
    with pytest.raises(ValueError):
        wsi._relative_level_scales(1.0, "mpp")


def test__relative_level_scales_no_objective_power():
    """Test _relative_level_scales objective when objective power is None."""

    class DummyWSI:
        """Mock WSIReader for testing."""

        _relative_level_scales = wsireader.WSIReader._relative_level_scales

        @property
        def info(self):
            return wsireader.WSIMeta((100, 100), axes="YXS")

    wsi = DummyWSI()
    with pytest.raises(ValueError):
        wsi._relative_level_scales(10, "power")


def test__relative_level_scales_level_too_high(sample_svs):
    """Test _relative_level_scales levels set too high."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    with pytest.raises(ValueError):
        wsi._relative_level_scales(100, "level")


def test_find_optimal_level_and_downsample_openslide_interpolation_warning(
    sample_ndpi,
):
    """Test finding optimal level for mpp read with scale > 1.

    This tests the case where the scale is found to be > 1 and interpolation
    will be applied to the output. A UserWarning should be raised in this case.
    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    with pytest.warns(UserWarning):
        _, _ = wsi._find_optimal_level_and_downsample(0.1, "mpp")


def test_find_optimal_level_and_downsample_jp2_interpolation_warning(sample_jp2):
    """Test finding optimal level for mpp read with scale > 1.

    This tests the case where the scale is found to be > 1 and interpolation
    will be applied to the output. A UserWarning should be raised in this case.
    """
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    with pytest.warns(UserWarning):
        _, _ = wsi._find_optimal_level_and_downsample(0.1, "mpp")


def test_find_optimal_level_and_downsample_mpp(sample_ndpi):
    """Test finding optimal level for mpp read."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)

    mpps = [0.5, 10]
    expected_levels = [0, 4]
    expected_scales = [[0.91282519, 0.91012514], [0.73026016, 0.72810011]]

    for mpp, expected_level, expected_scale in zip(
        mpps, expected_levels, expected_scales
    ):
        read_level, post_read_scale_factor = wsi._find_optimal_level_and_downsample(
            mpp, "mpp"
        )

        assert read_level == expected_level
        assert post_read_scale_factor == pytest.approx(expected_scale)


def test_find_optimal_level_and_downsample_power(sample_ndpi):
    """Test finding optimal level for objective power read."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)

    objective_powers = [20, 10, 5, 2.5, 1.25]
    expected_levels = [0, 1, 2, 3, 4]
    for objective_power, expected_level in zip(objective_powers, expected_levels):
        read_level, post_read_scale_factor = wsi._find_optimal_level_and_downsample(
            objective_power, "power"
        )

        assert read_level == expected_level
        assert np.array_equal(post_read_scale_factor, [1.0, 1.0])


def test_find_optimal_level_and_downsample_level(sample_ndpi):
    """Test finding optimal level for level read.

    For integer levels, the returned level should always be the same as
    the input level.
    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)

    for level in range(wsi.info.level_count):
        read_level, post_read_scale_factor = wsi._find_optimal_level_and_downsample(
            level, "level"
        )

        assert read_level == level
        assert np.array_equal(post_read_scale_factor, [1.0, 1.0])


def test_find_read_rect_params_power(sample_ndpi):
    """Test finding read rect parameters for objective power."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)

    location = NDPI_TEST_TISSUE_LOCATION
    size = NDPI_TEST_TISSUE_SIZE
    # Test a range of objective powers
    for target_scale in [1.25, 2.5, 5, 10, 20]:
        (level, _, read_size, post_read_scale, _) = wsi.find_read_rect_params(
            location=location,
            size=size,
            resolution=target_scale,
            units="power",
        )
        assert level >= 0
        assert level < wsi.info.level_count
        # Check that read_size * scale == size
        post_read_downscaled_size = np.round(read_size * post_read_scale).astype(int)
        assert np.array_equal(post_read_downscaled_size, np.array(size))


def test_find_read_rect_params_mpp(sample_ndpi):
    """Test finding read rect parameters for objective mpp."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)

    location = NDPI_TEST_TISSUE_LOCATION
    size = NDPI_TEST_TISSUE_SIZE
    # Test a range of MPP
    for target_scale in range(1, 10):
        (level, _, read_size, post_read_scale, _) = wsi.find_read_rect_params(
            location=location,
            size=size,
            resolution=target_scale,
            units="mpp",
        )
        assert level >= 0
        assert level < wsi.info.level_count
        # Check that read_size * scale == size
        post_read_downscaled_size = np.round(read_size * post_read_scale).astype(int)
        assert np.array_equal(post_read_downscaled_size, np.array(size))


def test_read_rect_openslide_baseline(sample_ndpi):
    """Test openslide read rect at baseline.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    location = NDPI_TEST_TISSUE_LOCATION
    size = NDPI_TEST_TISSUE_SIZE
    im_region = wsi.read_rect(location, size, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)


def test_read_rect_jp2_baseline(sample_jp2):
    """Test jp2 read rect at baseline.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    location = JP2_TEST_TISSUE_LOCATION
    size = JP2_TEST_TISSUE_SIZE
    im_region = wsi.read_rect(location, size, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)


def test_read_rect_tiffreader_svs_baseline(sample_svs):
    """Test TIFFWSIReader.read_rect with an SVS file at baseline."""
    wsi = wsireader.TIFFWSIReader(sample_svs)
    location = SVS_TEST_TISSUE_LOCATION
    size = SVS_TEST_TISSUE_SIZE
    im_region = wsi.read_rect(location, size, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)


def test_read_rect_tiffreader_ome_tiff_baseline(sample_ome_tiff):
    """Test TIFFWSIReader.read_rect with an OME-TIFF file at baseline."""
    wsi = wsireader.TIFFWSIReader(sample_ome_tiff)
    location = SVS_TEST_TISSUE_LOCATION
    size = SVS_TEST_TISSUE_SIZE
    im_region = wsi.read_rect(location, size, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)


def test_read_rect_openslide_levels(sample_ndpi):
    """Test openslide read rect with resolution in levels.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    location = NDPI_TEST_TISSUE_LOCATION
    size = NDPI_TEST_TISSUE_SIZE
    for level in range(wsi.info.level_count):
        im_region = wsi.read_rect(location, size, resolution=level, units="level")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert im_region.shape == (*size[::-1], 3)


def test_read_rect_jp2_levels(sample_jp2):
    """Test jp2 read rect with resolution in levels.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    location = (0, 0)
    size = JP2_TEST_TISSUE_SIZE
    width, height = size
    for level in range(wsi.info.level_count):
        level_width, level_height = wsi.info.level_dimensions[level]
        im_region = wsi.read_rect(location, size, resolution=level, units="level")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert pytest.approx(
            im_region.shape,
            (
                min(height, level_height),
                min(width, level_width),
                3,
            ),
            abs=1,
        )


def read_rect_mpp(wsi, location, size):
    """Read rect with resolution in microns per pixel."""
    for factor in range(1, 10):
        mpp = wsi.info.mpp * factor
        im_region = wsi.read_rect(location, size, resolution=mpp, units="mpp")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert im_region.shape == (*size[::-1], 3)


def test_read_rect_openslide_mpp(sample_ndpi):
    """Test openslide read rect with resolution in microns per pixel.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    location = NDPI_TEST_TISSUE_LOCATION
    size = NDPI_TEST_TISSUE_SIZE
    read_rect_mpp(wsi, location, size)


def test_read_rect_jp2_mpp(sample_jp2):
    """Test jp2 read rect with resolution in microns per pixel.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    location = JP2_TEST_TISSUE_LOCATION
    size = JP2_TEST_TISSUE_SIZE
    read_rect_mpp(wsi, location, size)


def test_read_rect_openslide_objective_power(sample_ndpi):
    """Test openslide read rect with resolution in objective power.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    location = NDPI_TEST_TISSUE_LOCATION
    size = NDPI_TEST_TISSUE_SIZE

    read_rect_objective_power(wsi, location, size)


def test_read_rect_jp2_objective_power(sample_jp2):
    """Test jp2 read rect with resolution in objective power.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    location = JP2_TEST_TISSUE_LOCATION
    size = JP2_TEST_TISSUE_SIZE

    read_rect_objective_power(wsi, location, size)


def test_read_bounds_openslide_baseline(sample_ndpi):
    """Test openslide read bounds at baseline.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    bounds = NDPI_TEST_TISSUE_BOUNDS
    size = NDPI_TEST_TISSUE_SIZE
    im_region = wsi.read_bounds(bounds, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)


def test_read_bounds_jp2_baseline(sample_jp2):
    """Test jp2 read bounds at baseline.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    bounds = JP2_TEST_TISSUE_BOUNDS
    size = JP2_TEST_TISSUE_SIZE
    im_region = wsi.read_bounds(bounds, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)

    bounds = (32768, 42880, 33792, 50000)
    im_region = wsi.read_bounds(bounds, resolution=2.5, units="power")
    assert im_region.dtype == "uint8"
    assert im_region.shape == (445, 64, 3)


def test_read_bounds_openslide_levels(sample_ndpi):
    """Test openslide read bounds with resolution in levels.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    bounds = NDPI_TEST_TISSUE_BOUNDS
    width, height = NDPI_TEST_TISSUE_SIZE
    for level, downsample in enumerate(wsi.info.level_downsamples):
        im_region = wsi.read_bounds(bounds, resolution=level, units="level")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        expected_output_shape = tuple(
            np.round([height / downsample, width / downsample, 3]).astype(int)
        )
        assert im_region.shape == expected_output_shape


def test_read_bounds_jp2_levels(sample_jp2):
    """Test jp2 read bounds with resolution in levels.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    bounds = JP2_TEST_TISSUE_BOUNDS
    width, height = JP2_TEST_TISSUE_SIZE
    for level, downsample in enumerate(wsi.info.level_downsamples):
        im_region = wsi.read_bounds(bounds, resolution=level, units="level")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        expected_output_shape = tuple(
            np.round([height / downsample, width / downsample])
        )
        assert im_region.shape[:2] == pytest.approx(expected_output_shape, abs=1)
        assert im_region.shape[2] == 3


def test_read_bounds_openslide_mpp(sample_ndpi):
    """Test openslide read bounds with resolution in microns per pixel.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    bounds = NDPI_TEST_TISSUE_BOUNDS
    size = NDPI_TEST_TISSUE_SIZE

    read_bounds_mpp(wsi, bounds, size)


def test_read_bounds_jp2_mpp(sample_jp2):
    """Test jp2 read bounds with resolution in microns per pixel.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    bounds = JP2_TEST_TISSUE_BOUNDS
    size = JP2_TEST_TISSUE_SIZE

    read_bounds_mpp(wsi, bounds, size, jp2=True)


def test_read_bounds_openslide_objective_power(sample_ndpi):
    """Test openslide read bounds with resolution in objective power.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    bounds = NDPI_TEST_TISSUE_BOUNDS
    size = NDPI_TEST_TISSUE_SIZE
    slide_power = wsi.info.objective_power

    read_bounds_objective_power(wsi, slide_power, bounds, size)


def test_read_bounds_jp2_objective_power(sample_jp2):
    """Test jp2 read bounds with resolution in objective power.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    bounds = JP2_TEST_TISSUE_BOUNDS
    size = JP2_TEST_TISSUE_SIZE
    slide_power = wsi.info.objective_power

    read_bounds_objective_power(wsi, slide_power, bounds, size, jp2=True)


def test_read_bounds_interpolated(sample_svs):
    """Test openslide read bounds with interpolated output.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    bounds = SVS_TEST_TISSUE_BOUNDS
    size = SVS_TEST_TISSUE_SIZE
    im_region = wsi.read_bounds(
        bounds,
        resolution=0.1,
        units="mpp",
    )

    assert 0.1 < wsi.info.mpp[0]
    assert 0.1 < wsi.info.mpp[1]
    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape[2] == 3
    assert all(np.array(im_region.shape[:2]) > size)


def test_read_bounds_level_consistency_openslide(sample_ndpi):
    """Test read_bounds produces the same visual field across resolution levels."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    bounds = NDPI_TEST_TISSUE_BOUNDS

    read_bounds_level_consistency(wsi, bounds)


def test_read_bounds_level_consistency_jp2(sample_jp2):
    """Test read_bounds produces the same visual field across resolution levels."""
    bounds = JP2_TEST_TISSUE_BOUNDS
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)

    read_bounds_level_consistency(wsi, bounds)


def test_wsireader_get_thumbnail_openslide(sample_svs):
    """Test for get_thumbnail as a python function."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    slide_thumbnail = wsi.slide_thumbnail()
    assert isinstance(slide_thumbnail, np.ndarray)
    assert slide_thumbnail.dtype == "uint8"


def test_wsireader_get_thumbnail_jp2(sample_jp2):
    """Test for get_thumbnail as a python function."""
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    slide_thumbnail = wsi.slide_thumbnail()
    assert isinstance(slide_thumbnail, np.ndarray)
    assert slide_thumbnail.dtype == "uint8"


def test_wsireader_save_tiles(sample_svs, tmp_path):
    """Test for save_tiles in wsireader as a python function."""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(sample_svs).parent),
        file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    wsi.save_tiles(
        output_dir=str(pathlib.Path(tmp_path).joinpath("test_wsireader_save_tiles")),
        tile_objective_value=5,
        tile_read_size=(5000, 5000),
        verbose=True,
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("test_wsireader_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("test_wsireader_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("test_wsireader_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )


def test_incompatible_objective_value(sample_svs, tmp_path):
    """Test for incompatible objective value."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    with pytest.raises(ValueError):
        wsi.save_tiles(
            output_dir=str(
                pathlib.Path(tmp_path).joinpath("test_wsireader_save_tiles")
            ),
            tile_objective_value=3,
            tile_read_size=(5000, 5000),
            verbose=True,
        )


def test_incompatible_level(sample_svs, tmp_path):
    """Test for incompatible objective value."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    with pytest.warns(UserWarning):
        wsi.save_tiles(
            output_dir=str(
                pathlib.Path(tmp_path).joinpath("test_wsireader_save_tiles2")
            ),
            tile_objective_value=1,
            tile_read_size=(500, 500),
            verbose=True,
        )


def test_wsireader_jp2_save_tiles(sample_jp2, tmp_path):
    """Test for save_tiles in wsireader as a python function."""
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    wsi.save_tiles(
        output_dir=str(
            pathlib.Path(tmp_path).joinpath("test_wsireader_jp2_save_tiles")
        ),
        tile_objective_value=5,
        tile_read_size=(5000, 5000),
        verbose=True,
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("test_wsireader_jp2_save_tiles")
        .joinpath("test1.jp2")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("test_wsireader_jp2_save_tiles")
        .joinpath("test1.jp2")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("test_wsireader_jp2_save_tiles")
        .joinpath("test1.jp2")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )


def test_openslide_objective_power_from_mpp(sample_svs):
    """Test OpenSlideWSIReader approximation of objective power from mpp."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    wsi.openslide_wsi = DummyMutableOpenSlideObject(wsi.openslide_wsi)
    props = wsi.openslide_wsi._properties

    del props["openslide.objective-power"]  # skipcq: PTC-W0043
    with pytest.warns(UserWarning, match=r"Objective power inferred"):
        _ = wsi.info

    del props["openslide.mpp-x"]  # skipcq: PTC-W0043
    del props["openslide.mpp-y"]  # skipcq: PTC-W0043
    with pytest.warns(UserWarning, match=r"Unable to determine objective power"):
        _ = wsi._info()


def test_openslide_mpp_from_tiff_resolution(sample_svs):
    """Test OpenSlideWSIReader mpp from TIFF resolution tags."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    wsi.openslide_wsi = DummyMutableOpenSlideObject(wsi.openslide_wsi)
    props = wsi.openslide_wsi._properties

    del props["openslide.mpp-x"]  # skipcq: PTC-W0043
    del props["openslide.mpp-y"]  # skipcq: PTC-W0043
    props["tiff.ResolutionUnit"] = "centimeter"
    props["tiff.XResolution"] = 1e4  # Pixels per cm
    props["tiff.YResolution"] = 1e4  # Pixels per cm
    with pytest.warns(UserWarning, match=r"Falling back to TIFF resolution"):
        _ = wsi.info

    assert np.array_equal(wsi.info.mpp, [1, 1])


def test_VirtualWSIReader(source_image):
    """Test VirtualWSIReader"""
    wsi = wsireader.VirtualWSIReader(pathlib.Path(source_image))
    with pytest.warns(UserWarning, match=r"Unknown scale"):
        _ = wsi._info()
    with pytest.warns(UserWarning, match=r"Raw data is None"):
        _ = wsi._info()

    assert wsi.img.shape == (256, 256, 3)

    img = wsi.read_rect(location=(0, 0), size=(100, 50))
    assert img.shape == (50, 100, 3)

    img = wsi.read_region(location=(0, 0), size=(100, 50), level=0)
    assert img.shape == (50, 100, 3)


def test_VirtualWSIReader_invalid_mode(source_image):
    """Test creating a VritualWSIReader with an invalid mode."""
    with pytest.raises(ValueError):
        wsireader.VirtualWSIReader(pathlib.Path(source_image), mode="foo")


def test_VirtualWSIReader_read_bounds(source_image):
    """Test VirtualWSIReader read bounds"""
    wsi = wsireader.VirtualWSIReader(pathlib.Path(source_image))
    img = wsi.read_bounds(bounds=(0, 0, 50, 100))
    assert img.shape == (100, 50, 3)

    img = wsi.read_bounds(bounds=(0, 0, 50, 100), resolution=1.5, units="baseline")
    assert img.shape == (150, 75, 3)

    img = wsi.read_bounds(bounds=(0, 0, 50, 100), resolution=0.5, units="baseline")
    assert img.shape == (50, 25, 3)

    with pytest.raises(IndexError):
        _ = wsi.read_bounds(bounds=(0, 0, 50, 100), resolution=0.5, units="level")

    with pytest.raises(ValueError):
        _ = wsi.read_bounds(bounds=(0, 0, 50, 100), resolution=1, units="level")


def test_VirtualWSIReader_read_rect(source_image):
    """Test VirtualWSIReader read rect."""
    wsi = wsireader.VirtualWSIReader(pathlib.Path(source_image))
    info = wsi.info

    img = wsi.read_rect(location=(0, 0), size=(50, 100))
    assert img.shape == (100, 50, 3)

    img = wsi.read_rect(
        location=(0, 0), size=(50, 100), resolution=1.5, units="baseline"
    )
    assert img.shape == (100, 50, 3)

    img = wsi.read_rect(
        location=(0, 0), size=(50, 100), resolution=0.5, units="baseline"
    )
    assert img.shape == (100, 50, 3)

    with pytest.raises(IndexError):
        _ = wsi.read_rect(
            location=(0, 0), size=(50, 100), resolution=0.5, units="level"
        )

    with pytest.raises(ValueError):
        _ = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=1, units="level")

    wsi = wsireader.VirtualWSIReader(pathlib.Path(source_image), info=info)

    assert info.as_dict() == wsi.info.as_dict()


def test_VirtualWSIReader_read_bounds_virtual_baseline(source_image):
    """Test VirtualWSIReader read bounds with virtual baseline."""
    image_path = pathlib.Path(source_image)
    img_array = utils.misc.imread(image_path)
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(slide_dimensions=double_size, axes="YXS")
    wsi = wsireader.VirtualWSIReader(image_path, info=meta)
    location = (0, 0)
    size = (50, 100)
    bounds = utils.transforms.locsize2bounds(location, size)
    region = wsi.read_bounds(bounds, pad_mode="constant", interpolation="cubic")
    target_size = tuple(np.round(np.array([25, 50]) * 2).astype(int))
    target = cv2.resize(
        img_array[:50, :25, :], target_size, interpolation=cv2.INTER_CUBIC
    )

    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.1


def test_VirtualWSIReader_read_rect_virtual_baseline(source_image):
    """Test VirtualWSIReader read rect with virtual baseline.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image.

    """
    img_array = utils.misc.imread(pathlib.Path(source_image))
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(slide_dimensions=double_size, axes="YXS")
    wsi = wsireader.VirtualWSIReader(pathlib.Path(source_image), info=meta)
    region = wsi.read_rect(location=(0, 0), size=(50, 100))
    target = cv2.resize(
        img_array[:50, :25, :], (50, 100), interpolation=cv2.INTER_CUBIC
    )
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2


def test_VirtualWSIReader_read_rect_virtual_levels(source_image):
    """Test VirtualWSIReader read rect with vritual levels.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image and the pyramid/resolution levels.

    Checks that the regions read at each level line up with expected values.

    """
    img_array = utils.misc.imread(pathlib.Path(source_image))
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(
        slide_dimensions=double_size, level_downsamples=[1, 2, 4], axes="YXS"
    )
    wsi = wsireader.VirtualWSIReader(pathlib.Path(source_image), info=meta)
    region = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=1, units="level")
    target = img_array[:100, :50, :]
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2

    region = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=2, units="level")
    target = cv2.resize(img_array[:200, :100, :], (50, 100))
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2


def test_VirtualWSIReader_read_bounds_virtual_levels(source_image):
    """Test VirtualWSIReader read bounds with vritual levels.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image and the pyramid/resolution levels.

    Checks that the regions read at each level line up with expected values.

    """
    img_array = utils.misc.imread(pathlib.Path(source_image))
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(
        slide_dimensions=double_size, level_downsamples=[1, 2, 4], axes="YXS"
    )
    wsi = wsireader.VirtualWSIReader(pathlib.Path(source_image), info=meta)
    location = (0, 0)
    size = (50, 100)
    bounds = utils.transforms.locsize2bounds(location, size)

    region = wsi.read_bounds(bounds, resolution=1, units="level")
    target = img_array[:50, :25, :]
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2

    region = wsi.read_bounds(bounds, resolution=2, units="level")
    target_size = tuple(np.round(np.array([25, 50]) / 2).astype(int))
    target = cv2.resize(
        img_array[:50, :25, :], target_size, interpolation=cv2.INTER_CUBIC
    )
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2


def test_VirtualWSIReader_read_rect_virtual_levels_mpp(source_image):
    """Test VirtualWSIReader read rect with vritual levels and MPP.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image and the pyramid/resolution levels and
    a baseline MPP specified.

    Checks that the regions read with specified MPP for each level lines up
    with expected values.
    """
    img_array = utils.misc.imread(pathlib.Path(source_image))
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(
        slide_dimensions=double_size,
        axes="YXS",
        level_downsamples=[1, 2, 4],
        mpp=(0.25, 0.25),
    )
    wsi = wsireader.VirtualWSIReader(pathlib.Path(source_image), info=meta)
    region = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=0.5, units="mpp")
    target = img_array[:100, :50, :]
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2

    region = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=1, units="mpp")
    target = cv2.resize(
        img_array[:200, :100, :], (50, 100), interpolation=cv2.INTER_CUBIC
    )
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2


def test_VirtualWSIReader_read_bounds_virtual_levels_mpp(source_image):
    """Test VirtualWSIReader read bounds with vritual levels and MPP.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image and the pyramid/resolution levels.

    Checks that the regions read at each level line up with expected values.

    """
    img_array = utils.misc.imread(pathlib.Path(source_image))
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(
        slide_dimensions=double_size,
        axes="YXS",
        level_downsamples=[1, 2, 4],
        mpp=(0.25, 0.25),
    )
    wsi = wsireader.VirtualWSIReader(pathlib.Path(source_image), info=meta)
    location = (0, 0)
    size = (50, 100)
    bounds = utils.transforms.locsize2bounds(location, size)

    region = wsi.read_bounds(bounds, resolution=0.5, units="mpp")
    target = img_array[:50, :25, :]
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2

    region = wsi.read_bounds(bounds, resolution=1, units="mpp")
    target_size = tuple(np.round(np.array([25, 50]) / 2).astype(int))
    target = cv2.resize(
        img_array[:50, :25, :], target_size, interpolation=cv2.INTER_CUBIC
    )
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2


def test_tissue_mask_otsu(sample_svs):
    """Test wsi.tissue_mask with Otsu's method."""

    wsi = wsireader.OpenSlideWSIReader(sample_svs)

    tissue_thumb = wsi.slide_thumbnail()
    grey_thumb = cv2.cvtColor(tissue_thumb, cv2.COLOR_RGB2GRAY)

    otsu_threhold = threshold_otsu(grey_thumb)
    otsu_mask = grey_thumb < otsu_threhold

    mask = wsi.tissue_mask(method="otsu")
    mask_thumb = mask.slide_thumbnail()

    assert np.mean(np.logical_xor(mask_thumb, otsu_mask)) < 0.05


def test_tissue_mask_morphological(sample_svs):
    """Test wsi.tissue_mask with morphological method."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    resolutions = [5, 10]
    units = ["power", "mpp"]
    scale_fns = [lambda x: x * 2, lambda x: 32 / x]
    for unit, scaler in zip(units, scale_fns):
        for resolution in resolutions:
            mask = wsi.tissue_mask(
                method="morphological", resolution=resolution, units=unit
            )

            tissue_thumb = wsi.slide_thumbnail(resolution, unit)
            grey_thumb = tissue_thumb.mean(axis=-1)
            mask_thumb = mask.slide_thumbnail(resolution, unit)

            otsu_threhold = threshold_otsu(grey_thumb)
            otsu_mask = grey_thumb < otsu_threhold
            morpho_mask = binary_dilation(otsu_mask, disk(scaler(resolution)))
            morpho_mask = remove_small_objects(morpho_mask, 100 * scaler(resolution))

    assert np.mean(np.logical_xor(mask_thumb, morpho_mask)) < 0.1


def test_tissue_mask_morphological_levels(sample_svs):
    """Test wsi.tissue_mask with morphological method and resolution in level."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    thumb = wsi.slide_thumbnail(0, "level")
    grey_thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    threshold = threshold_otsu(grey_thumb)
    reference = grey_thumb < threshold
    # Using kernel_size of 1
    mask = wsi.tissue_mask("morphological", 0, "level")
    mask_thumb = mask.slide_thumbnail(0, "level")
    assert np.mean(mask_thumb == reference) > 0.99
    # Custom kernel_size (should still be close to reference)
    reference = binary_dilation(reference, disk(3))
    mask = wsi.tissue_mask("morphological", 0, "level", kernel_size=3)
    mask_thumb = mask.slide_thumbnail(0, "level")
    assert np.mean(mask_thumb == reference) > 0.95


def test_tissue_mask_read_bounds_none_interpolation(sample_svs):
    """Test reading a mask using read_bounds with no interpolation."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    mask = wsi.tissue_mask("otsu")
    mask_region = mask.read_bounds((0, 0, 512, 512), interpolation="none")
    assert mask_region.shape[0] == 32
    assert mask_region.shape[1] == 33


def test_tissue_mask_read_rect_none_interpolation(sample_svs):
    """Test reading a mask using read_rect with no interpolation."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    mask = wsi.tissue_mask("otsu")
    mask_region = mask.read_rect((0, 0), (512, 512), interpolation="none")
    assert mask_region.shape[0] == 32
    assert mask_region.shape[1] == 33


def test_invalid_masker_method(sample_svs):
    """Test that an invalid masker method string raises a ValueError."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    with pytest.raises(ValueError):
        wsi.tissue_mask(method="foo")


def test_get_wsireader(
    sample_svs, sample_ndpi, sample_jp2, sample_ome_tiff, source_image
):
    """Test get_wsireader to return correct object."""
    with pytest.raises(FileNotSupported):
        _ = wsireader.get_wsireader("./sample.csv")

    with pytest.raises(TypeError):
        _ = wsireader.get_wsireader([1, 2])

    wsi = wsireader.get_wsireader(sample_svs)
    assert isinstance(wsi, wsireader.OpenSlideWSIReader)

    wsi = wsireader.get_wsireader(sample_ndpi)
    assert isinstance(wsi, wsireader.OpenSlideWSIReader)

    wsi = wsireader.get_wsireader(sample_jp2)
    assert isinstance(wsi, wsireader.OmnyxJP2WSIReader)

    wsi = wsireader.get_wsireader(sample_ome_tiff)
    assert isinstance(wsi, wsireader.TIFFWSIReader)

    wsi = wsireader.get_wsireader(pathlib.Path(source_image))
    assert isinstance(wsi, wsireader.VirtualWSIReader)

    img = utils.misc.imread(str(pathlib.Path(source_image)))
    wsi = wsireader.get_wsireader(input_img=img)
    assert isinstance(wsi, wsireader.VirtualWSIReader)

    # test if get_wsireader can accept a wsireader instance
    wsi_type = type(wsi)
    wsi_out = wsireader.get_wsireader(input_img=wsi)
    assert isinstance(wsi_out, wsi_type)

    # test loading .npy
    temp_dir = _get_temp_folder_path()
    os.mkdir(temp_dir)
    temp_file = f"{temp_dir}/sample.npy"
    np.save(temp_file, np.random.randint(1, 255, [5, 5, 5]))
    wsi_out = wsireader.get_wsireader(temp_file)
    assert isinstance(wsi_out, VirtualWSIReader)
    shutil.rmtree(temp_dir)


def test_jp2_missing_cod(sample_jp2):
    """Test for warning if JP2 is missing COD segment."""
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    wsi.glymur_wsi.codestream.segment = []
    with pytest.warns(UserWarning, match="missing COD"):
        _ = wsi.info


def test_read_rect_at_resolution(sample_wsi_dict):
    """Test for read rect using location at requested."""
    mini_wsi2_svs = pathlib.Path(sample_wsi_dict["wsi1_8k_8k_svs"])
    mini_wsi2_jpg = pathlib.Path(sample_wsi_dict["wsi1_8k_8k_jpg"])
    mini_wsi2_jp2 = pathlib.Path(sample_wsi_dict["wsi1_8k_8k_jp2"])

    # * check sync read between Virtual Reader and WSIReader (openslide) (reference)
    reader_list = [
        VirtualWSIReader(mini_wsi2_jpg),
        OpenSlideWSIReader(mini_wsi2_svs),
        OmnyxJP2WSIReader(mini_wsi2_jp2),
    ]

    for reader_idx, reader in enumerate(reader_list):
        roi1 = reader.read_rect(
            np.array([500, 500]),
            np.array([2000, 2000]),
            coord_space="baseline",
            resolution=1.00,
            units="baseline",
        )
        roi2 = reader.read_rect(
            np.array([1000, 1000]),
            np.array([4000, 4000]),
            coord_space="resolution",
            resolution=2.00,
            units="baseline",
        )
        roi2 = imresize(roi2, output_size=[2000, 2000])
        cc = np.corrcoef(roi1[..., 0].flatten(), roi2[..., 0].flatten())
        # this control the harshness of similarity test, how much should be?
        assert np.min(cc) > 0.90, reader_idx


def test_read_bounds_location_in_requested_resolution(sample_wsi_dict):
    """Actually a duel test for sync read and read at requested."""
    # """Test synchronize read for VirtualReader"""
    # convert to pathlib Path to prevent wsireader complaint
    mini_wsi1_msk = pathlib.Path(sample_wsi_dict["wsi2_4k_4k_msk"])
    mini_wsi2_svs = pathlib.Path(sample_wsi_dict["wsi1_8k_8k_svs"])
    mini_wsi2_jpg = pathlib.Path(sample_wsi_dict["wsi1_8k_8k_jpg"])
    mini_wsi2_jp2 = pathlib.Path(sample_wsi_dict["wsi1_8k_8k_jp2"])

    def compare_reader(reader1, reader2, read_coord, read_cfg, check_content=True):
        """Correlation test to compare output of 2 readers."""
        requested_size = read_coord[2:] - read_coord[:2]
        requested_size = requested_size[::-1]  # XY to YX
        roi1 = reader1.read_bounds(
            read_coord,
            coord_space="resolution",
            pad_constant_values=255,
            **read_cfg,
        )
        roi2 = reader2.read_bounds(
            read_coord,
            coord_space="resolution",
            pad_constant_values=255,
            **read_cfg,
        )
        # using only reader 1 because it is reference reader
        shape1 = reader1.slide_dimensions(**read_cfg)
        # shape2 = reader2.slide_dimensions(**read_cfg)
        # print(read_cfg, shape1, shape2)
        assert roi1.shape[0] == requested_size[0], (
            read_cfg,
            requested_size,
            roi1.shape,
        )
        assert roi1.shape[1] == requested_size[1], (
            read_cfg,
            requested_size,
            roi1.shape,
        )
        assert roi1.shape[0] == roi2.shape[0], (read_cfg, roi1.shape, roi2.shape)
        assert roi1.shape[1] == roi2.shape[1], (read_cfg, roi1.shape, roi2.shape)
        if check_content:
            cc = np.corrcoef(roi1[..., 0].flatten(), roi2[..., 0].flatten())
            # this control the harshness of similarity test, how much should be?
            assert np.min(cc) > 0.90, (cc, read_cfg, read_coord, shape1)

    # * now check sync read by comparing the RoI with different base
    # the output should be at same resolution even if source is of different base
    msk = imread(mini_wsi1_msk)
    msk_reader = VirtualWSIReader(msk)

    bigger_msk = cv2.resize(
        msk, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST
    )
    bigger_msk_reader = VirtualWSIReader(bigger_msk)
    # * must set mpp metadata to not None else wont work
    ref_metadata = bigger_msk_reader.info
    ref_metadata.mpp = np.array([1.0, 1.0])
    ref_metadata.objective_power = 1.0
    msk_reader.info = ref_metadata

    shape2 = bigger_msk_reader.slide_dimensions(resolution=0.75, units="mpp")
    shape1 = msk_reader.slide_dimensions(resolution=0.75, units="mpp")
    assert shape1[0] - shape2[0] < 10  # offset may happen if shape is not multiple
    assert shape1[1] - shape2[1] < 10  # offset may happen if shape is not multiple
    shape2 = bigger_msk_reader.slide_dimensions(resolution=0.75, units="power")
    shape1 = msk_reader.slide_dimensions(resolution=0.75, units="power")
    assert shape1[0] - shape2[0] < 10  # offset may happen if shape is not multiple
    assert shape1[1] - shape2[1] < 10  # offset may happen if shape is not multiple

    # * check sync read between Virtual Reader
    requested_coords = np.array([3500, 3000, 5500, 7000])  # XY, manually pick
    # baseline value can be think of as scaling factor wrt baseline
    read_cfg_list = [
        # read at strange resolution value so that if it fails,
        # normal scale will also fail
        ({"resolution": 0.75, "units": "mpp"}, np.array([10000, 10000, 15000, 15000])),
        (
            {"resolution": 1.56, "units": "power"},
            np.array([10000, 10000, 15000, 15000]),
        ),
        ({"resolution": 3.00, "units": "mpp"}, np.array([2500, 2500, 4000, 4000])),
        ({"resolution": 0.30, "units": "baseline"}, np.array([2000, 2000, 3000, 3000])),
    ]
    for _, (read_cfg, read_coord) in enumerate(read_cfg_list):
        read_coord = requested_coords if read_coord is None else read_coord
        compare_reader(msk_reader, bigger_msk_reader, read_coord, read_cfg)

    # * check sync read between Virtual Reader and WSIReader (openslide) (reference)
    requested_coords = np.array([3500, 3000, 4500, 4000])  # XY, manually pick
    read_cfg_list = [
        # read at strange resolution value so that if it fails,
        # it means normal scale will also fail
        ({"resolution": 0.35, "units": "mpp"}, np.array([1000, 1000, 2000, 2000])),
        ({"resolution": 23.5, "units": "power"}, None),
        ({"resolution": 0.35, "units": "baseline"}, np.array([1000, 1000, 2000, 2000])),
        ({"resolution": 1.35, "units": "baseline"}, np.array([8000, 8000, 9000, 9000])),
        ({"resolution": 1.00, "units": "level"}, np.array([1000, 1000, 2000, 2000])),
    ]

    wsi_reader = OpenSlideWSIReader(mini_wsi2_svs)
    tile = imread(mini_wsi2_jpg)
    tile = imresize(tile, scale_factor=0.76)
    vrt_reader = VirtualWSIReader(tile)
    vrt_reader.info = wsi_reader.info

    for _, (read_cfg, read_coord) in enumerate(read_cfg_list):
        read_coord = requested_coords if read_coord is None else read_coord
        compare_reader(wsi_reader, vrt_reader, read_coord, read_cfg, check_content=True)

    # * check sync read between Virtual Reader and WSIReader (jp2) (reference)
    requested_coords = np.array([2500, 2500, 4000, 4000])  # XY, manually pick
    read_cfg_list = [
        # read at strange resolution value so that if it fails,
        # normal scale will also fail
        ({"resolution": 0.35, "units": "mpp"}, None),
        ({"resolution": 23.5, "units": "power"}, None),
        ({"resolution": 0.65, "units": "baseline"}, np.array([3000, 3000, 4000, 4000])),
        ({"resolution": 1.35, "units": "baseline"}, np.array([4000, 4000, 5000, 5000])),
        ({"resolution": 1.00, "units": "level"}, np.array([1500, 1500, 2000, 2000])),
    ]
    wsi_reader = OmnyxJP2WSIReader(mini_wsi2_jp2)
    wsi_thumb = wsi_reader.slide_thumbnail(resolution=0.85, units="mpp")
    vrt_reader = VirtualWSIReader(wsi_thumb)
    vrt_reader.info = wsi_reader.info

    for _, (read_cfg, read_coord) in enumerate(read_cfg_list):
        read_coord = requested_coords if read_coord is None else read_coord
        compare_reader(wsi_reader, vrt_reader, read_coord, read_cfg)


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_read_bounds(sample_ndpi, tmp_path):
    """Test OpenSlide read_bounds CLI."""
    runner = CliRunner()
    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--img-input",
            str(pathlib.Path(sample_ndpi)),
            "--resolution",
            "0",
            "--units",
            "level",
            "--mode",
            "save",
            "--region",
            "0",
            "0",
            "2000",
            "2000",
            "--output-path",
            str(pathlib.Path(tmp_path).joinpath("im_region.jpg")),
        ],
    )

    assert read_bounds_result.exit_code == 0
    assert pathlib.Path(tmp_path).joinpath("im_region.jpg").is_file()

    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--img-input",
            str(pathlib.Path(sample_ndpi)),
            "--resolution",
            "0",
            "--units",
            "level",
            "--mode",
            "save",
            "--output-path",
            str(pathlib.Path(tmp_path).joinpath("im_region2.jpg")),
        ],
    )

    assert read_bounds_result.exit_code == 0
    assert pathlib.Path(tmp_path).joinpath("im_region2.jpg").is_file()


def test_command_line_jp2_read_bounds(sample_jp2, tmp_path):
    """Test JP2 read_bounds."""
    runner = CliRunner()
    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--img-input",
            str(pathlib.Path(sample_jp2)),
            "--resolution",
            "0",
            "--units",
            "level",
            "--mode",
            "save",
        ],
    )

    assert read_bounds_result.exit_code == 0
    assert pathlib.Path(tmp_path).joinpath("../im_region.jpg").is_file()


def test_command_line_jp2_read_bounds_show(sample_jp2, tmp_path):
    """Test JP2 read_bounds with mode as 'show'."""
    runner = CliRunner()
    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--img-input",
            str(pathlib.Path(sample_jp2)),
            "--resolution",
            "0",
            "--units",
            "level",
            "--mode",
            "show",
        ],
    )

    assert read_bounds_result.exit_code == 0


def test_command_line_unsupported_file_read_bounds(sample_svs, tmp_path):
    """Test unsupported file read bounds."""
    runner = CliRunner()
    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--img-input",
            str(pathlib.Path(sample_svs))[:-1],
            "--resolution",
            "0",
            "--units",
            "level",
            "--mode",
            "save",
        ],
    )

    assert read_bounds_result.output == ""
    assert read_bounds_result.exit_code == 1
    assert isinstance(read_bounds_result.exception, FileNotSupported)


def test_command_line_slide_thumbnail(sample_ndpi, tmp_path):
    """Test for the slide_thumbnail CLI."""
    runner = CliRunner()

    command_line_slide_thumbnail(runner, sample=sample_ndpi, tmp_path=tmp_path)


def test_command_line_slide_thumbnail_output_none(sample_svs, tmp_path):
    """Test cli slide thumbnail with output dir None."""
    runner = CliRunner()
    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--img-input",
            str(pathlib.Path(sample_svs)),
            "--mode",
            "save",
        ],
    )

    assert slide_thumb_result.exit_code == 0
    assert pathlib.Path(tmp_path).joinpath("../slide_thumb.jpg").is_file()


def test_command_line_jp2_slide_thumbnail(sample_jp2, tmp_path):
    """Test for the jp2 slide_thumbnail CLI."""
    runner = CliRunner()

    command_line_slide_thumbnail(runner, sample=sample_jp2, tmp_path=tmp_path)


def test_command_line_jp2_slide_thumbnail_mode_show(sample_jp2, tmp_path):
    """Test for the jp2 slide_thumbnail CLI mode='show'."""
    runner = CliRunner()

    command_line_slide_thumbnail(
        runner, sample=sample_jp2, tmp_path=tmp_path, mode="show"
    )


def test_command_line_jp2_slide_thumbnail_file_not_supported(sample_jp2, tmp_path):
    """Test for the jp2 slide_thumbnail CLI."""
    runner = CliRunner()

    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--img-input",
            str(pathlib.Path(sample_jp2))[:-1],
            "--mode",
            "save",
            "--output-path",
            str(pathlib.Path(tmp_path).joinpath("slide_thumb.jpg")),
        ],
    )

    assert slide_thumb_result.output == ""
    assert slide_thumb_result.exit_code == 1
    assert isinstance(slide_thumb_result.exception, FileNotSupported)


def test_openslide_read_rect_edge_reflect_padding(sample_svs):
    """Test openslide edge reflect padding for read_rect."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    region = wsi.read_rect((-64, -64), (128, 128), pad_mode="reflect")
    assert 0 not in region.min(axis=-1)


def test_openslide_read_bounds_edge_reflect_padding(sample_svs):
    """Test openslide edge reflect padding for read_bounds."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    region = wsi.read_bounds((-64, -64, 64, 64), pad_mode="reflect")
    assert 0 not in region.min(axis=-1)


def test_tiffwsireader_invalid_tiff(sample_ndpi):
    """Test for TIFF which is not supported by TIFFWSIReader."""
    with pytest.raises(ValueError, match="Unsupported TIFF"):
        _ = wsireader.TIFFWSIReader(sample_ndpi)


def test_tiffwsireader_invalid_svs_metadata(sample_svs, monkeypatch):
    """Test for invalid SVS key-value pairs in TIFF escription tag."""
    wsi = wsireader.TIFFWSIReader(sample_svs)
    monkeypatch.setattr(
        wsi.tiff.pages[0],
        "description",
        wsi.tiff.pages[0].description.replace("=", "=="),
    )
    with pytest.raises(ValueError, match="key=value"):
        _ = wsi._info()


def test_tiffwsireader_invalid_ome_metadata(sample_ome_tiff, monkeypatch):
    """Test exception raised for invalid OME-XML metadata instrument."""
    wsi = wsireader.TIFFWSIReader(sample_ome_tiff)
    monkeypatch.setattr(
        wsi.tiff.pages[0],
        "description",
        wsi.tiff.pages[0].description.replace(
            '<Objective ID="Objective:0:0" NominalMagnification="20.0"/>', ""
        ),
    )
    with pytest.raises(KeyError, match="No matching Instrument"):
        _ = wsi._info()


def test_tiffwsireader_ome_metadata_missing_one_mppy(sample_ome_tiff, monkeypatch):
    """Test no exception raised for missing x/y mpp but warning given."""
    for dim in "XY":
        wsi = wsireader.TIFFWSIReader(sample_ome_tiff)
        monkeypatch.setattr(
            wsi.tiff.pages[0],
            "description",
            re.sub(f'PhysicalSize{dim}="[^"]*"', "", wsi.tiff.pages[0].description),
        )
        with pytest.warns(UserWarning, match="Only one MPP"):
            _ = wsi._info()


def test_arrayview_unsupported_axes():
    """Test unsupported axes in ArrayView."""
    array = zarr.ones((128, 128, 3))
    array_view = ArrayView(array=array, axes="FOO")
    with pytest.raises(Exception, match="Unsupported axes"):
        array_view[:64, :64, :]


def test_arrayview_unsupported_axes_shape(sample_ome_tiff, monkeypatch):
    """Test accessing an unspported axes in TIFFWSIReader._shape_channels_last."""
    wsi = wsireader.TIFFWSIReader(sample_ome_tiff)
    monkeypatch.setattr(wsi, "_axes", "FOO")
    with pytest.raises(Exception, match="Unsupported axes"):
        _ = wsi._info()


def test_arrayview_incomplete_index():
    """Test reading from ArrayView without specifying all axes slices."""
    array = zarr.array(np.random.rand(128, 128, 3))
    array_view = ArrayView(array=array, axes="YXS")
    view_1 = array_view[:64, :64, :]
    view_2 = array_view[:64, :64]
    assert np.array_equal(view_1, view_2)


def test_arrayview_single_number_index():
    """Test reading a column from ArrayView.

    I'm not sure why you would want to do this but it is implemented for
    consistency with other __getitem__ objects.
    """
    array = zarr.array(np.random.rand(128, 128, 3))
    array_view = ArrayView(array=array, axes="YXS")
    view_1 = array_view[0]
    view_2 = array_view[0]
    assert np.array_equal(view_1, view_2)


class TestReader:
    scenarios = [
        ("TIFFReader", {"reader_class": TIFFWSIReader}),
    ]

    @staticmethod
    def test_wsimeta_attrs(sample_ome_tiff, reader_class):
        """Check for expected attrs in .info / WSIMeta.

        Checks for existence of expected attrs but not their contents.
        """
        wsi = reader_class(sample_ome_tiff)
        info = wsi.info
        expected_attrs = [
            "slide_dimensions",
            "axes",
            "level_dimensions",
            "level_count",
            "level_downsamples",
            "vendor",
            "mpp",
            "objective_power",
            "file_path",
        ]
        for attr in expected_attrs:
            assert hasattr(info, attr)

    @staticmethod
    def test_read_rect_level_consistency(sample_ome_tiff, reader_class):
        """Compare the same region at each stored resolution level.

        Read the same region at each stored resolution level and compare
        the resulting image using phase cross correlation to check that
        they are aligned.

        """
        wsi = reader_class(sample_ome_tiff)
        location = (0, 0)
        size = np.array([1024, 1024])

        # Avoid testing very small levels (e.g. as in Omnyx JP2) because
        # MSE for very small levels is noisy.
        level_downsamples = [
            downsample for downsample in wsi.info.level_downsamples if downsample <= 32
        ]
        imgs = [
            wsi.read_rect(location, size // downsample, level, "level")
            for level, downsample in enumerate(level_downsamples)
        ]
        smallest_size = imgs[-1].shape[:2][::-1]
        resized = [imresize(img, output_size=smallest_size) for img in imgs]
        # Some blurring applied to account for changes in sharpness arising
        # from interpolation when calculating the downsampled levels. This
        # adds some tolerance for the comparison.
        blurred = [cv2.GaussianBlur(img, (5, 5), cv2.BORDER_REFLECT) for img in resized]
        as_float = [img.astype(np.float) for img in blurred]

        # Pair-wise check resolutions for mean squared error
        for i, a in enumerate(as_float):
            for b in as_float[i + 1 :]:
                _, error, phase_diff = phase_cross_correlation(a, b)
                assert phase_diff < 0.125
                assert error < 0.125

    @staticmethod
    def test_read_bounds_level_consistency(sample_ome_tiff, reader_class):
        """Compare the same region at each stored resolution level.

        Read the same region at each stored resolution level and compare
        the resulting image using phase cross correlation to check that
        they are aligned.

        """
        wsi = reader_class(sample_ome_tiff)
        bounds = (0, 0, 1024, 1024)
        # This logic can be moved from the helper to here when other
        # reader classes have been parameterised into scenarios also.
        read_bounds_level_consistency(wsi, bounds)

    @staticmethod
    def test_fuzz_read_region_baseline_size(sample_ome_tiff, reader_class):
        """Fuzz test for `read_bounds` output size at level 0 (baseline).

        - Tests that the output image size matches the input bounds size.
        - 50 random seeded reads are performed.
        - All test bounds are within the the slide dimensions.
        - Bounds sizes are randomised between 1 and 512 in width and height.
        """
        random.seed(123)
        wsi = reader_class(sample_ome_tiff)
        width, height = wsi.info.slide_dimensions
        iterations = 50
        for _ in range(iterations):
            size = (random.randint(1, 512), random.randint(1, 512))
            location = (
                random.randint(0, width - size[0]),
                random.randint(0, height - size[1]),
            )
            bounds = locsize2bounds(location, size)
            region = wsi.read_bounds(bounds, resolution=0, units="level")
            assert region.shape[:2][::-1] == size

    @staticmethod
    def test_read_rect_coord_space_consistency(sample_ome_tiff, reader_class):
        """Test that read_rect coord_space modes are consistent.

        Using `read_rect` with `coord_space="baseline"` and
        `coord_space="resolution"` should produce the same output when
        the bounds are a multiple of the scale difference between the two
        modes. I.E. reading at baseline with a set of coordinates should
        yield the same region as reading at half the resolution and
        with coordinates which are half the size. Note that the output
        will not be of the same size, but the field of view will match.

        """
        reader = reader_class(sample_ome_tiff)
        roi1 = reader.read_rect(
            np.array([500, 500]),
            np.array([2000, 2000]),
            coord_space="baseline",
            resolution=1.00,
            units="baseline",
        )
        roi2 = reader.read_rect(
            np.array([250, 250]),
            np.array([1000, 1000]),
            coord_space="resolution",
            resolution=0.5,
            units="baseline",
        )
        # Make the regions the same size for comparison of content
        roi2 = imresize(roi2, output_size=[2000, 2000])
        cc = np.corrcoef(roi1[..., 0].flatten(), roi2[..., 0].flatten())
        # This control the harshness of similarity test, how much should be?
        assert np.min(cc) > 0.95

    @staticmethod
    def test_region_dump(sample_ome_tiff, reader_class):
        from matplotlib import pyplot as plt

        wsi = reader_class(sample_ome_tiff)
        _, axs = plt.subplots(
            nrows=1,
            ncols=wsi.info.level_count,
            figsize=(wsi.info.level_count, 3),
            squeeze=False,
        )
        for level, ax in zip(range(wsi.info.level_count), axs[0]):
            bounds = (0, 0, 1024, 1024)
            region = wsi.read_bounds(bounds, resolution=level, units="level")
            ax.imshow(region)
        plt.savefig("tiff_level_check.png")
