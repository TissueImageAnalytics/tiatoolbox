"""Tests for reading whole-slide images."""

import json
import os
import pathlib
import random
import re
import shutil
from copy import deepcopy
from time import time

# When no longer supporting Python <3.9 this should be collections.abc.Iterable
from typing import Iterable

import cv2
import numpy as np
import pytest
import zarr
from click.testing import CliRunner
from skimage.filters import threshold_otsu
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.morphology import binary_dilation, disk, remove_small_objects
from skimage.registration import phase_cross_correlation

from tiatoolbox import cli, rcParam, utils
from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.data import _fetch_remote_sample
from tiatoolbox.utils.exceptions import FileNotSupported
from tiatoolbox.utils.misc import imread
from tiatoolbox.utils.transforms import imresize, locsize2bounds
from tiatoolbox.utils.visualization import AnnotationRenderer
from tiatoolbox.wsicore import wsireader
from tiatoolbox.wsicore.wsireader import (
    AnnotationStoreReader,
    ArrayView,
    DICOMWSIReader,
    NGFFWSIReader,
    OmnyxJP2WSIReader,
    OpenSlideWSIReader,
    TIFFWSIReader,
    VirtualWSIReader,
    WSIReader,
    is_ngff,
    is_zarr,
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

COLOR_DICT = {
    0: (200, 0, 0, 255),
    1: (0, 200, 0, 255),
    2: (0, 0, 200, 255),
    3: (155, 155, 0, 255),
    4: (155, 0, 155, 255),
    5: (0, 155, 155, 255),
}


# -------------------------------------------------------------------------------------
# Utility Test Functions
# -------------------------------------------------------------------------------------


def _get_temp_folder_path(prefix="temp"):
    """Return unique temp folder path"""
    return os.path.join(rcParam["TIATOOLBOX_HOME"], f"{prefix}-{int(time())}")


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
        bounds_shape = np.array(size[::-1])
        expected_output_shape = tuple((bounds_shape / downsample).round().astype(int))
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
        bounds_shape = np.array(size[::-1])
        expected_output_shape = tuple((bounds_shape / downsample).round().astype(int))
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
    # Avoid testing very small levels (e.g. as in Omnyx JP2) because
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
            _, error, phase_diff = phase_cross_correlation(a, b, normalization=None)
            assert phase_diff < 0.125
            assert error < 0.125


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
    level_scales = wsi.info.relative_level_scales(0.125, "baseline")
    level_scales = np.array(level_scales)
    downsamples = np.array(wsi.info.level_downsamples)
    expected = downsamples * 0.125
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test_relative_level_scales_openslide_baseline(sample_ndpi):
    """Test openslide relative level scales for pixels per baseline pixel."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    relative_level_scales_baseline(wsi)


def test_relative_level_scales_jp2_baseline(sample_jp2):
    """Test jp2 relative level scales for pixels per baseline pixel."""
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    relative_level_scales_baseline(wsi)


def test_relative_level_scales_openslide_mpp(sample_ndpi):
    """Test openslide calculation of relative level scales for mpp."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    level_scales = wsi.info.relative_level_scales(0.5, "mpp")
    level_scales = np.array(level_scales)
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert all(level_scales[0] == wsi.info.mpp / 0.5)


def test_relative_level_scales_jp2_mpp(sample_jp2):
    """Test jp2 calculation of relative level scales for mpp."""
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    level_scales = wsi.info.relative_level_scales(0.5, "mpp")
    level_scales = np.array(level_scales)
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert all(level_scales[0] == wsi.info.mpp / 0.5)


def relative_level_scales_power(wsi):
    """Calculation of relative level scales for objective power."""
    level_scales = wsi.info.relative_level_scales(wsi.info.objective_power, "power")
    level_scales = np.array(level_scales)
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert np.array_equal(level_scales[0], [1, 1])
    downsamples = np.array(wsi.info.level_downsamples)
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], downsamples)


def test_relative_level_scales_openslide_power(sample_ndpi):
    """Test openslide calculation of relative level scales for objective power."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    relative_level_scales_power(wsi)


def test_relative_level_scales_jp2_power(sample_jp2):
    """Test jp2 calculation of relative level scales for objective power."""
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    relative_level_scales_power(wsi)


def relative_level_scales_level(wsi):
    """Calculation of relative level scales for level."""
    level_scales = wsi.info.relative_level_scales(3, "level")
    level_scales = np.array(level_scales)
    assert np.array_equal(level_scales[3], [1, 1])
    downsamples = np.array(wsi.info.level_downsamples)
    expected = downsamples / downsamples[3]
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test_relative_level_scales_openslide_level(sample_ndpi):
    """Test openslide calculation of relative level scales for level."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    relative_level_scales_level(wsi)


def test_relative_level_scales_jp2_level(sample_jp2):
    """Test jp2 calculation of relative level scales for level."""
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    relative_level_scales_level(wsi)


def relative_level_scales_float(wsi):
    """Calculation of relative level scales for fractional level."""
    level_scales = wsi.info.relative_level_scales(1.5, "level")
    level_scales = np.array(level_scales)
    assert level_scales[0] == pytest.approx([1 / 3, 1 / 3])
    downsamples = np.array(wsi.info.level_downsamples)
    expected = downsamples / downsamples[0] * (1 / 3)
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test_relative_level_scales_openslide_level_float(sample_ndpi):
    """Test openslide calculation of relative level scales for fractional level."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    relative_level_scales_float(wsi)


def test_relative_level_scales_jp2_level_float(sample_jp2):
    """Test jp2 calculation of relative level scales for fractional level."""
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    relative_level_scales_float(wsi)


def test_relative_level_scales_invalid_units(sample_svs):
    """Test relative_level_scales with invalid units."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    with pytest.raises(ValueError, match="Invalid units"):
        wsi.info.relative_level_scales(1.0, "gibberish")


def test_relative_level_scales_no_mpp():
    """Test relative_level_scales objective when mpp is None."""

    class DummyWSI:
        """Mock WSIReader for testing."""

        @property
        def info(self):
            return wsireader.WSIMeta((100, 100), axes="YXS")

    wsi = DummyWSI()
    with pytest.raises(ValueError, match="MPP is None"):
        wsi.info.relative_level_scales(1.0, "mpp")


def test_relative_level_scales_no_objective_power():
    """Test relative_level_scales objective when objective power is None."""

    class DummyWSI:
        """Mock WSIReader for testing."""

        @property
        def info(self):
            return wsireader.WSIMeta((100, 100), axes="YXS")

    wsi = DummyWSI()
    with pytest.raises(ValueError, match="Objective power is None"):
        wsi.info.relative_level_scales(10, "power")


def test_relative_level_scales_level_too_high(sample_svs):
    """Test relative_level_scales levels set too high."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    with pytest.raises(ValueError, match="levels"):
        wsi.info.relative_level_scales(100, "level")


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


def test_convert_resolution_units(sample_ndpi):
    """Test the resolution unit conversion code."""
    wsi = wsireader.WSIReader.open(sample_ndpi)

    # test for invalid input and output units
    with pytest.raises(ValueError, match=r".*Invalid input_unit.*"):
        wsi.convert_resolution_units(0, input_unit="invalid")
    with pytest.raises(ValueError, match=r".*Invalid output_unit.*"):
        wsi.convert_resolution_units(0, input_unit="level", output_unit="level")

    # Test functionality: Assuming a baseline_mpp to be 0.25 at 40x mag
    gt_mpp = wsi.info.mpp
    gt_power = wsi.info.objective_power
    gt_dict = {"mpp": 2 * gt_mpp, "power": gt_power / 2, "baseline": 0.5}

    # convert input_unit == "mpp" to other formats
    output = wsi.convert_resolution_units(2 * gt_mpp, input_unit="mpp")
    assert output["power"] == gt_dict["power"]

    # convert input_unit == "power" to other formats
    output = wsi.convert_resolution_units(
        gt_power / 2, input_unit="power", output_unit="mpp"
    )
    assert all(output == gt_dict["mpp"])

    # convert input_unit == "level" to other formats
    output = wsi.convert_resolution_units(1, input_unit="level")
    assert all(output["mpp"] == gt_dict["mpp"])

    # convert input_unit == "baseline" to other formats
    output = wsi.convert_resolution_units(0.5, input_unit="baseline")
    assert output["power"] == gt_dict["power"]

    # Test for missing information
    org_info = wsi.info
    # test when mpp is missing
    _info = deepcopy(org_info)
    _info.mpp = None
    wsi._m_info = _info
    with pytest.raises(ValueError, match=r".*Missing 'mpp'.*"):
        wsi.convert_resolution_units(0, input_unit="mpp")
    _ = wsi.convert_resolution_units(0, input_unit="power")

    # test when power is missing
    _info = deepcopy(org_info)
    _info.objective_power = None
    wsi._m_info = _info
    with pytest.raises(ValueError, match=r".*Missing 'objective_power'.*"):
        wsi.convert_resolution_units(0, input_unit="power")
    _ = wsi.convert_resolution_units(0, input_unit="mpp")

    # test when power and mpp are missing
    _info = deepcopy(org_info)
    _info.objective_power = None
    _info.mpp = None
    wsi._m_info = _info
    _ = wsi.convert_resolution_units(0, input_unit="baseline")
    with pytest.warns(UserWarning, match=r".*output_unit is returned as None.*"):
        _ = wsi.convert_resolution_units(0, input_unit="level", output_unit="mpp")


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


def test_is_tiled_tiff(source_image):
    source_image.replace(source_image.with_suffix(".tiff"))
    assert wsireader.is_tiled_tiff(source_image.with_suffix(".tiff")) is False
    source_image.with_suffix(".tiff").replace(source_image)


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
        im_region = wsi.read_rect(
            location,
            size,
            resolution=level,
            units="level",
            pad_mode=None,
        )

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert im_region.shape == pytest.approx(
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


def test_wsireader_save_tiles(sample_svs, tmp_path):
    """Test for save_tiles in wsireader as a python function."""
    tmp_path = pathlib.Path(tmp_path)
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(sample_svs).parent),
        file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    wsi.save_tiles(
        output_dir=str(tmp_path / ("test_wsireader_save_tiles")),
        tile_objective_value=5,
        tile_read_size=(5000, 5000),
        verbose=True,
    )
    assert (
        tmp_path / "test_wsireader_save_tiles" / "CMU-1-Small-Region.svs" / "Output.csv"
    ).exists()
    assert (
        tmp_path
        / "test_wsireader_save_tiles"
        / "CMU-1-Small-Region.svs"
        / "slide_thumbnail.jpg"
    ).exists()
    assert (
        tmp_path
        / "test_wsireader_save_tiles"
        / "CMU-1-Small-Region.svs"
        / "Tile_5_0_0.jpg"
    ).exists()


def test_incompatible_objective_value(sample_svs, tmp_path):
    """Test for incompatible objective value."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    with pytest.raises(ValueError, match="objective power"):
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
    tmp_path = pathlib.Path(tmp_path)
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    wsi.save_tiles(
        output_dir=str(tmp_path / "test_wsireader_jp2_save_tiles"),
        tile_objective_value=5,
        tile_read_size=(5000, 5000),
        verbose=True,
    )
    assert (
        tmp_path / "test_wsireader_jp2_save_tiles" / "test1.jp2" / "Output.csv"
    ).exists()
    assert (
        tmp_path / "test_wsireader_jp2_save_tiles" / "test1.jp2" / "slide_thumbnail.jpg"
    ).exists()
    assert (
        tmp_path / "test_wsireader_jp2_save_tiles" / "test1.jp2" / "Tile_5_0_0.jpg"
    ).exists()


def test_openslide_objective_power_from_mpp(sample_svs):
    """Test OpenSlideWSIReader approximation of objective power from mpp."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    wsi.openslide_wsi = DummyMutableOpenSlideObject(wsi.openslide_wsi)
    props = wsi.openslide_wsi._properties

    del props["openslide.objective-power"]  # skipcq
    with pytest.warns(UserWarning, match=r"Objective power inferred"):
        _ = wsi.info

    del props["openslide.mpp-x"]  # skipcq
    del props["openslide.mpp-y"]  # skipcq
    with pytest.warns(UserWarning, match=r"Unable to determine objective power"):
        _ = wsi._info()


def test_openslide_mpp_from_tiff_resolution(sample_svs):
    """Test OpenSlideWSIReader mpp from TIFF resolution tags."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    wsi.openslide_wsi = DummyMutableOpenSlideObject(wsi.openslide_wsi)
    props = wsi.openslide_wsi._properties

    del props["openslide.mpp-x"]  # skipcq
    del props["openslide.mpp-y"]  # skipcq
    props["tiff.ResolutionUnit"] = "centimeter"
    props["tiff.XResolution"] = 1e4  # Pixels per cm
    props["tiff.YResolution"] = 1e4  # Pixels per cm
    with pytest.warns(UserWarning, match=r"Falling back to TIFF resolution"):
        _ = wsi.info

    assert np.array_equal(wsi.info.mpp, [1, 1])


def test_virtual_wsi_reader(source_image):
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


def test_virtual_wsi_reader_invalid_mode(source_image):
    """Test creating a VritualWSIReader with an invalid mode."""
    with pytest.raises(ValueError, match="Invalid mode"):
        wsireader.VirtualWSIReader(pathlib.Path(source_image), mode="foo")


def test_virtual_wsi_reader_read_bounds(source_image):
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

    with pytest.raises(ValueError, match="level"):
        _ = wsi.read_bounds(bounds=(0, 0, 50, 100), resolution=1, units="level")


def test_virtual_wsi_reader_read_rect(source_image):
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

    with pytest.raises(ValueError, match="level"):
        _ = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=1, units="level")

    wsi = wsireader.VirtualWSIReader(pathlib.Path(source_image), info=info)

    assert info.as_dict() == wsi.info.as_dict()


def test_virtual_wsi_reader_read_bounds_virtual_baseline(source_image):
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
    region = wsi.read_bounds(bounds, pad_mode="reflect", interpolation="cubic")
    target_size = tuple(np.round(np.array([25, 50]) * 2).astype(int))
    target = cv2.resize(
        img_array[:50, :25, :], target_size, interpolation=cv2.INTER_CUBIC
    )

    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.1


def test_virtual_wsi_reader_read_rect_virtual_baseline(source_image):
    """Test VirtualWSIReader read rect with virtual baseline.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image.

    """
    img_array = utils.misc.imread(pathlib.Path(source_image))
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(slide_dimensions=double_size, axes="YXS")
    wsi = wsireader.VirtualWSIReader(pathlib.Path(source_image), info=meta)
    region = wsi.read_rect(location=(0, 0), size=(50, 100), pad_mode="reflect")
    target = cv2.resize(
        img_array[:50, :25, :], (50, 100), interpolation=cv2.INTER_CUBIC
    )
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2


def test_virtual_wsi_reader_read_rect_virtual_levels(source_image):
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
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 1

    region = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=2, units="level")
    target = cv2.resize(img_array[:200, :100, :], (50, 100))
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 1


def test_virtual_wsi_reader_read_bounds_virtual_levels(source_image):
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
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 1

    region = wsi.read_bounds(bounds, resolution=2, units="level")
    target_size = tuple(np.round(np.array([25, 50]) / 2).astype(int))
    target = cv2.resize(
        img_array[:50, :25, :], target_size, interpolation=cv2.INTER_CUBIC
    )
    offset, error, _ = phase_cross_correlation(target, region, normalization=None)
    assert all(offset == 0)
    assert error < 0.1
    psnr = peak_signal_noise_ratio(target, region)
    assert psnr < 50


def test_virtual_wsi_reader_read_rect_virtual_levels_mpp(source_image):
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
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 1

    region = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=1, units="mpp")
    target = cv2.resize(
        img_array[:200, :100, :], (50, 100), interpolation=cv2.INTER_CUBIC
    )
    offset, error, _ = phase_cross_correlation(target, region, normalization=None)
    assert all(offset == 0)
    assert error < 0.1
    psnr = peak_signal_noise_ratio(target, region)
    assert psnr < 50


def test_virtual_wsi_reader_read_bounds_virtual_levels_mpp(source_image):
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
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 1

    region = wsi.read_bounds(bounds, resolution=1, units="mpp")
    target_size = tuple(np.round(np.array([25, 50]) / 2).astype(int))
    target = cv2.resize(
        img_array[:50, :25, :], target_size, interpolation=cv2.INTER_CUBIC
    )
    offset, error, _ = phase_cross_correlation(target, region, normalization=None)
    assert all(offset == 0)
    assert error < 0.1
    psnr = peak_signal_noise_ratio(target, region)
    assert psnr < 50


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
    assert mask_region.shape[1] == 32


def test_tissue_mask_read_rect_none_interpolation(sample_svs):
    """Test reading a mask using read_rect with no interpolation."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    mask = wsi.tissue_mask("otsu")
    mask_region = mask.read_rect((0, 0), (512, 512), interpolation="none")
    assert mask_region.shape[0] == 32
    assert mask_region.shape[1] == 32


def test_invalid_masker_method(sample_svs):
    """Test that an invalid masking method string raises a ValueError."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    with pytest.raises(ValueError, match="masking method"):
        wsi.tissue_mask(method="foo")


def test_wsireader_open(
    sample_svs, sample_ndpi, sample_jp2, sample_ome_tiff, source_image
):
    """Test WSIReader.open() to return correct object."""
    with pytest.raises(FileNotSupported):
        _ = WSIReader.open("./sample.csv")

    with pytest.raises(TypeError):
        _ = WSIReader.open([1, 2])

    wsi = WSIReader.open(sample_svs)
    assert isinstance(wsi, wsireader.OpenSlideWSIReader)

    wsi = WSIReader.open(sample_ndpi)
    assert isinstance(wsi, wsireader.OpenSlideWSIReader)

    wsi = WSIReader.open(sample_jp2)
    assert isinstance(wsi, wsireader.OmnyxJP2WSIReader)

    wsi = WSIReader.open(sample_ome_tiff)
    assert isinstance(wsi, wsireader.TIFFWSIReader)

    wsi = WSIReader.open(pathlib.Path(source_image))
    assert isinstance(wsi, wsireader.VirtualWSIReader)

    img = utils.misc.imread(str(pathlib.Path(source_image)))
    wsi = WSIReader.open(input_img=img)
    assert isinstance(wsi, wsireader.VirtualWSIReader)

    # test if WSIReader.open can accept a wsireader instance
    wsi_type = type(wsi)
    wsi_out = WSIReader.open(input_img=wsi)
    assert isinstance(wsi_out, wsi_type)

    # test loading .npy
    temp_dir = _get_temp_folder_path()
    os.mkdir(temp_dir)
    temp_file = f"{temp_dir}/sample.npy"
    np.save(temp_file, np.random.randint(1, 255, [5, 5, 5]))
    wsi_out = WSIReader.open(temp_file)
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

    # * now check sync read by comparing the ROI with different base
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


@pytest.mark.skipif(
    utils.env_detection.running_on_ci(),
    reason="No need to display image on travis.",
)
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


def test_tiffwsireader_invalid_tiff(remote_sample):
    """Test for TIFF which is not supported by TIFFWSIReader."""
    with pytest.raises(ValueError, match="Unsupported TIFF"):
        _ = wsireader.TIFFWSIReader(remote_sample("two-tiled-pages"))


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


def test_manual_mpp_tuple(sample_svs):
    """Test setting a manual mpp for a WSI."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs, mpp=(0.123, 0.123))
    assert tuple(wsi.info.mpp) == (0.123, 0.123)


def test_manual_mpp_float(sample_svs):
    """Test setting a manual mpp for a WSI."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs, mpp=0.123)
    assert tuple(wsi.info.mpp) == (0.123, 0.123)


def test_manual_mpp_invalid(sample_svs):
    """Test setting a manual mpp for a WSI."""
    with pytest.raises(TypeError, match="mpp"):
        _ = wsireader.OpenSlideWSIReader(sample_svs, mpp=(0.5,))
    with pytest.raises(TypeError, match="mpp"):
        _ = wsireader.OpenSlideWSIReader(sample_svs, mpp="foo")


def test_manual_power_tuple(sample_svs):
    """Test setting a manual power for a WSI."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs, power=42)
    assert wsi.info.objective_power == 42


def test_manual_power_invalid(sample_svs):
    """Test setting a manual power for a WSI."""
    with pytest.raises(TypeError, match="power"):
        _ = wsireader.OpenSlideWSIReader(sample_svs, power=(42,))


def test_tiled_tiff_openslide(remote_sample):
    """Test reading a tiled TIFF file with OpenSlide."""
    sample_path = remote_sample("tiled-tiff-1-small-jpeg")
    wsi = wsireader.WSIReader.open(sample_path)
    assert isinstance(wsi, wsireader.OpenSlideWSIReader)


def test_tiled_tiff_tifffile(remote_sample):
    """Test fallback to tifffile for files which openslide cannot read.

    E.G. tiled tiffs with JPEG XL compression.

    """
    sample_path = remote_sample("tiled-tiff-1-small-jp2k")
    wsi = wsireader.WSIReader.open(sample_path)
    assert isinstance(wsi, wsireader.TIFFWSIReader)


def test_is_zarr_empty_dir(tmp_path):
    """Test is_zarr is false for an empty .zarr directory."""
    zarr_dir = tmp_path / "zarr.zarr"
    zarr_dir.mkdir()
    assert not is_zarr(zarr_dir)


def test_is_zarr_array(tmp_path):
    """Test is_zarr is true for a .zarr directory with an array."""
    zarr_dir = tmp_path / "zarr.zarr"
    zarr_dir.mkdir()
    _zarray_path = zarr_dir / ".zarray"
    minimal_zarray = {
        "shape": [1, 1, 1],
        "dtype": "uint8",
        "compressor": {
            "id": "lz4",
        },
        "chunks": [1, 1, 1],
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "zarr_format": 2,
    }
    with open(_zarray_path, "w") as f:
        json.dump(minimal_zarray, f)
    assert is_zarr(zarr_dir)


def test_is_zarr_group(tmp_path):
    """Test is_zarr is true for a .zarr directory with an group."""
    zarr_dir = tmp_path / "zarr.zarr"
    zarr_dir.mkdir()
    _zgroup_path = zarr_dir / ".zgroup"
    minimal_zgroup = {
        "zarr_format": 2,
    }
    with open(_zgroup_path, "w") as f:
        json.dump(minimal_zgroup, f)
    assert is_zarr(zarr_dir)


def test_is_ngff_regular_zarr(tmp_path):
    """Test is_ngff is false for a regular zarr."""
    zarr_path = tmp_path / "zarr.zarr"
    zarr.open(zarr_path, "w")
    assert is_zarr(zarr_path)
    assert not is_ngff(zarr_path)

    # check we get the appropriate error message if we open it
    with pytest.raises(FileNotSupported, match="does not appear to be a v0.4"):
        WSIReader.open(zarr_path)


def test_store_reader_no_info(tmp_path):
    """Test AnnotationStoreReader with no info."""
    SQLiteStore(tmp_path / "store.db")
    with pytest.raises(ValueError, match="No metadata found"):
        AnnotationStoreReader(tmp_path / "store.db")


def test_store_reader_explicit_info(remote_sample, tmp_path):
    """Test AnnotationStoreReader with explicit info."""
    SQLiteStore(tmp_path / "store.db")
    wsi_reader = WSIReader.open(remote_sample("svs-1-small"))
    reader = AnnotationStoreReader(tmp_path / "store.db", wsi_reader.info)
    assert reader._info().as_dict() == wsi_reader.info.as_dict()


def test_store_reader_from_store(remote_sample, tmp_path):
    """Test AnnotationStoreReader from an AnnotationStore object."""
    store = SQLiteStore(remote_sample("annotation_store_svs_1"))
    reader = AnnotationStoreReader(store)
    assert isinstance(reader.store, SQLiteStore)


def test_store_reader_base_wsi_str(remote_sample, tmp_path):
    """Test AnnotationStoreReader with base_wsi as a string."""
    store = SQLiteStore(remote_sample("annotation_store_svs_1"))
    reader = AnnotationStoreReader(store, base_wsi=remote_sample("svs-1-small"))
    assert isinstance(reader.store, SQLiteStore)
    assert isinstance(reader.base_wsi, WSIReader)


def test_store_reader_alpha(remote_sample):
    """Test AnnotationStoreReader with alpha channel."""
    wsi_reader = WSIReader.open(remote_sample("svs-1-small"))
    store_reader = AnnotationStoreReader(
        remote_sample("annotation_store_svs_1"),
        wsi_reader.info,
        base_wsi=wsi_reader,
    )
    wsi_thumb = wsi_reader.slide_thumbnail()
    wsi_tile = wsi_reader.read_rect((500, 500), (1000, 1000))
    store_thumb = store_reader.slide_thumbnail()
    store_tile = store_reader.read_rect((500, 500), (1000, 1000))
    store_reader.alpha = 0.2
    store_thumb_alpha = store_reader.slide_thumbnail()
    store_tile_alpha = store_reader.read_rect((500, 500), (1000, 1000))
    # the thumbnail with low alpha should be closer to wsi_thumb
    assert np.mean(np.abs(store_thumb_alpha - wsi_thumb)) < np.mean(
        np.abs(store_thumb - wsi_thumb)
    )
    # the tile with low alpha should be closer to wsi_tile
    assert np.mean(np.abs(store_tile_alpha - wsi_tile)) < np.mean(
        np.abs(store_tile - wsi_tile)
    )


def test_store_reader_no_types(tmp_path, remote_sample):
    """Test AnnotationStoreReader with no types."""
    SQLiteStore(tmp_path / "store.db")
    wsi_reader = WSIReader.open(remote_sample("svs-1-small"))
    reader = AnnotationStoreReader(tmp_path / "store.db", wsi_reader.info)
    # shouldn't try to color by type if not present
    assert reader.renderer.score_prop is None


def test_store_reader_info_from_base(tmp_path, remote_sample):
    """Test that AnnotationStoreReader will correctly get metadata
    from a provided base_wsi if the store has no wsi metadata."""
    SQLiteStore(tmp_path / "store.db")
    wsi_reader = WSIReader.open(remote_sample("svs-1-small"))
    store_reader = AnnotationStoreReader(tmp_path / "store.db", base_wsi=wsi_reader)
    # the store reader should have the same metadata as the base wsi
    assert store_reader.info.mpp[0] == wsi_reader.info.mpp[0]


def test_ngff_zattrs_non_micrometer_scale_mpp(tmp_path):
    """Test that mpp is None if scale is not in micrometers."""
    sample = _fetch_remote_sample("ngff-1")
    # Create a copy of the sample with a non-micrometer scale
    sample_copy = tmp_path / "ngff-1"
    shutil.copytree(sample, sample_copy)
    with open(sample_copy / ".zattrs", "r") as fh:
        zattrs = json.load(fh)
    zattrs["multiscales"][0]["axes"][0]["unit"] = "foo"
    with open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    with pytest.warns(UserWarning, match="micrometer"):
        wsi = wsireader.NGFFWSIReader(sample_copy)
    assert wsi.info.mpp is None


def test_ngff_zattrs_missing_axes_mpp(tmp_path):
    """Test that mpp is None if axes are missing."""
    sample = _fetch_remote_sample("ngff-1")
    # Create a copy of the sample with no axes
    sample_copy = tmp_path / "ngff-1"
    shutil.copytree(sample, sample_copy)
    with open(sample_copy / ".zattrs", "r") as fh:
        zattrs = json.load(fh)
    zattrs["multiscales"][0]["axes"] = []
    with open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    wsi = wsireader.NGFFWSIReader(sample_copy)
    assert wsi.info.mpp is None


def test_ngff_empty_datasets_mpp(tmp_path):
    """Test that mpp is None if there are no datasets."""
    sample = _fetch_remote_sample("ngff-1")
    # Create a copy of the sample with no axes
    sample_copy = tmp_path / "ngff-1"
    shutil.copytree(sample, sample_copy)
    with open(sample_copy / ".zattrs", "r") as fh:
        zattrs = json.load(fh)
    zattrs["multiscales"][0]["datasets"] = []
    with open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    wsi = wsireader.NGFFWSIReader(sample_copy)
    assert wsi.info.mpp is None


def test_nff_no_scale_transforms_mpp(tmp_path):
    """Test that mpp is None if no scale transforms are present."""
    sample = _fetch_remote_sample("ngff-1")
    # Create a copy of the sample with no axes
    sample_copy = tmp_path / "ngff-1"
    shutil.copytree(sample, sample_copy)
    with open(sample_copy / ".zattrs", "r") as fh:
        zattrs = json.load(fh)
    for i, _ in enumerate(zattrs["multiscales"][0]["datasets"]):
        datasets = zattrs["multiscales"][0]["datasets"][i]
        datasets["coordinateTransformations"][0]["type"] = "identity"
    with open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    wsi = wsireader.NGFFWSIReader(sample_copy)
    assert wsi.info.mpp is None


class TestReader:
    scenarios = [
        (
            "AnnotationReaderOverlaid",
            {
                "reader_class": AnnotationStoreReader,
                "sample_key": "annotation_store_svs_1",
                "kwargs": {
                    "renderer": AnnotationRenderer(
                        "type",
                        COLOR_DICT,
                    ),
                    "base_wsi": WSIReader.open(_fetch_remote_sample("svs-1-small")),
                    "alpha": 0.5,
                },
            },
        ),
        (
            "AnnotationReaderMaskOnly",
            {
                "reader_class": AnnotationStoreReader,
                "sample_key": "annotation_store_svs_1",
                "kwargs": {
                    "renderer": AnnotationRenderer(
                        "type",
                        COLOR_DICT,
                        blur_radius=3,
                    ),
                },
            },
        ),
        (
            "TIFFReader",
            {
                "reader_class": TIFFWSIReader,
                "sample_key": "ome-brightfield-pyramid-1-small",
                "kwargs": {},
            },
        ),
        (
            "DICOMReader",
            {
                "reader_class": DICOMWSIReader,
                "sample_key": "dicom-1",
                "kwargs": {},
            },
        ),
        (
            "NGFFWSIReader",
            {
                "reader_class": NGFFWSIReader,
                "sample_key": "ngff-1",
                "kwargs": {},
            },
        ),
        (
            "OpenSlideWSIReader (Small SVS)",
            {
                "reader_class": OpenSlideWSIReader,
                "sample_key": "svs-1-small",
                "kwargs": {},
            },
        ),
        (
            "OmnyxJP2WSIReader",
            {
                "reader_class": OmnyxJP2WSIReader,
                "sample_key": "jp2-omnyx-1",
                "kwargs": {},
            },
        ),
    ]

    @staticmethod
    def test_base_open(sample_key, reader_class, kwargs):
        """Checks that WSIReader.open detects the type correctly."""
        sample = _fetch_remote_sample(sample_key)
        wsi = WSIReader.open(sample)
        assert isinstance(wsi, reader_class)

    @staticmethod
    def test_wsimeta_attrs(sample_key, reader_class, kwargs):
        """Check for expected attrs in .info / WSIMeta.

        Checks for existence of expected attrs but not their contents.

        """
        sample = _fetch_remote_sample(sample_key)
        wsi = reader_class(sample, **kwargs)
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
    def test_read_rect_level_consistency(sample_key, reader_class, kwargs):
        """Compare the same region at each stored resolution level.

        Read the same region at each stored resolution level and compare
        the resulting image using phase cross correlation to check that
        they are aligned.

        """
        sample = _fetch_remote_sample(sample_key)
        wsi = reader_class(sample, **kwargs)
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
                _, error, phase_diff = phase_cross_correlation(a, b, normalization=None)
                assert phase_diff < 0.125
                assert error < 0.125

    @staticmethod
    def test_read_bounds_level_consistency(sample_key, reader_class, kwargs):
        """Compare the same region at each stored resolution level.

        Read the same region at each stored resolution level and compare
        the resulting image using phase cross correlation to check that
        they are aligned.

        """
        sample = _fetch_remote_sample(sample_key)
        wsi = reader_class(sample, **kwargs)
        bounds = (0, 0, 1024, 1024)
        # This logic can be moved from the helper to here when other
        # reader classes have been parameterised into scenarios also.
        read_bounds_level_consistency(wsi, bounds)

    @staticmethod
    def test_fuzz_read_region_baseline_size(sample_key, reader_class, kwargs):
        """Fuzz test for `read_bounds` output size at level 0 (baseline).

        - Tests that the output image size matches the input bounds size.
        - 50 random seeded reads are performed.
        - All test bounds are within the the slide dimensions.
        - Bounds sizes are randomised between 1 and 512 in width and height.

        """
        random.seed(123)
        sample = _fetch_remote_sample(sample_key)
        wsi = reader_class(sample, **kwargs)
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
    def test_read_rect_coord_space_consistency(sample_key, reader_class, kwargs):
        """Test that read_rect coord_space modes are consistent.

        Using `read_rect` with `coord_space="baseline"` and
        `coord_space="resolution"` should produce the same output when
        the bounds are a multiple of the scale difference between the two
        modes. I.E. reading at baseline with a set of coordinates should
        yield the same region as reading at half the resolution and
        with coordinates which are half the size. Note that the output
        will not be of the same size, but the field of view will match.

        """
        sample = _fetch_remote_sample(sample_key)
        reader = reader_class(sample, **kwargs)
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

        # Check MSE
        mse = np.mean((roi1 - roi2) ** 2)
        assert mse < 100

        # Check PSNR
        psnr = peak_signal_noise_ratio(roi1, roi2)
        assert psnr > 25

        # Check SSIM (skip very small roi regions)
        if np.greater(roi1.shape[2], 16).all():
            ssim = structural_similarity(roi1, roi2, multichannel=True)
            assert ssim > 0.9

    @staticmethod
    def test_file_path_does_not_exist(sample_key, reader_class, kwargs):
        """Test that FileNotFoundError is raised when file does not exist."""
        with pytest.raises(FileNotFoundError):
            _ = reader_class("./foo.bar")

    @staticmethod
    def test_read_mpp(sample_key, reader_class, kwargs):
        """Test that the mpp is read correctly."""
        sample = _fetch_remote_sample(sample_key)
        wsi = reader_class(sample, **kwargs)
        assert wsi.info.mpp == pytest.approx(0.25, 1)
