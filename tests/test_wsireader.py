"""Test for reading whole-slide images."""

from __future__ import annotations

import copy
import json
import logging
import re
import shutil
from copy import deepcopy
from pathlib import Path

# When no longer supporting Python <3.9 this should be collections.abc.Iterable
from typing import TYPE_CHECKING, Callable

import cv2
import glymur
import numpy as np
import pytest
import zarr
from click.testing import CliRunner
from packaging.version import Version
from skimage.filters import threshold_otsu
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.morphology import binary_dilation, disk, remove_small_objects
from skimage.registration import phase_cross_correlation

from tiatoolbox import cli, utils
from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.utils import imread
from tiatoolbox.utils.exceptions import FileNotSupportedError
from tiatoolbox.utils.magic import is_sqlite3
from tiatoolbox.utils.transforms import imresize, locsize2bounds
from tiatoolbox.utils.visualization import AnnotationRenderer
from tiatoolbox.wsicore import WSIReader, wsireader
from tiatoolbox.wsicore.wsireader import (
    AnnotationStoreReader,
    ArrayView,
    DICOMWSIReader,
    JP2WSIReader,
    NGFFWSIReader,
    OpenSlideWSIReader,
    TIFFWSIReader,
    VirtualWSIReader,
    is_ngff,
    is_zarr,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    import requests
    from openslide import OpenSlide

    from tiatoolbox.typing import IntBounds, IntPair
    from tiatoolbox.wsicore.wsimeta import WSIMeta

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------

NDPI_TEST_TISSUE_BOUNDS = (30400, 11810, 30912, 12322)
NDPI_TEST_TISSUE_LOCATION = (30400, 11810)
NDPI_TEST_TISSUE_SIZE = (512, 512)

SVS_TEST_TISSUE_BOUNDS = (1000, 2000, 2000, 3000)
SVS_TEST_TISSUE_LOCATION = (1000, 2000)
SVS_TEST_TISSUE_SIZE = (1000, 1000)

JP2_TEST_TISSUE_BOUNDS = (0, 0, 1024, 1024)
JP2_TEST_TISSUE_LOCATION = (0, 0)
JP2_TEST_TISSUE_SIZE = (1024, 1024)

COLOR_DICT = {
    0: (200, 0, 0, 255),
    1: (0, 200, 0, 255),
    2: (0, 0, 200, 255),
    3: (155, 155, 0, 255),
    4: (155, 0, 155, 255),
    5: (0, 155, 155, 255),
}
RNG = np.random.default_rng()  # Numpy Random Generator


# -------------------------------------------------------------------------------------
# Utility Test Functions
# -------------------------------------------------------------------------------------


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


def read_rect_objective_power(wsi: WSIReader, location: IntPair, size: IntPair) -> None:
    """Read rect objective helper."""
    for objective_power in [20, 10, 5, 2.5, 1.25]:
        im_region = wsi.read_rect(
            location,
            size,
            resolution=objective_power,
            units="power",
        )

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert im_region.shape == (*size[::-1], 3)


def read_bounds_mpp(
    wsi: WSIReader,
    bounds: IntBounds,
    size: IntPair,
    *,
    jp2: bool,
) -> None:
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


def read_bounds_objective_power(
    wsi: WSIReader,
    slide_power: float,
    bounds: IntBounds,
    size: IntPair,
    *,
    jp2: bool,
) -> None:
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
                expected_output_shape[:2],
                abs=1,
            )
        else:
            assert im_region.shape[:2] == expected_output_shape
        assert im_region.shape[2] == 3


def read_bounds_level_consistency(wsi: WSIReader, bounds: IntBounds) -> None:
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
    as_float = [img.astype(np.float64) for img in blurred]

    # Pair-wise check resolutions for mean squared error
    for i, a in enumerate(as_float):
        for b in as_float[i + 1 :]:
            _, error, phase_diff = phase_cross_correlation(a, b, normalization=None)
            assert phase_diff < 0.125
            assert error < 0.125


# -------------------------------------------------------------------------------------
# Utility Test Classes & Functions
# -------------------------------------------------------------------------------------


class DummyMutableOpenSlideObject:
    """Dummy OpenSlide object with mutable properties."""

    def __init__(self: DummyMutableOpenSlideObject, openslide_obj: OpenSlide) -> None:
        """DummyMutableOpenSlideObject initialization."""
        self.openslide_obj = openslide_obj
        self._properties = dict(openslide_obj.properties)

    def __getattr__(self: DummyMutableOpenSlideObject, name: str) -> object:
        """Catch references to OpenSlide object attributes."""
        return getattr(self.openslide_obj, name)

    @property
    def properties(self: DummyMutableOpenSlideObject) -> object:
        """Return the fake properties."""
        return self._properties


def relative_level_scales_baseline(wsi: WSIReader) -> None:
    """Relative level scales for pixels per baseline pixel."""
    level_scales = wsi.info.relative_level_scales(0.125, "baseline")
    level_scales = np.array(level_scales)
    downsamples = np.array(wsi.info.level_downsamples)
    expected = downsamples * 0.125
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


# -------------------------------------------------------------------------------------
# Generic Tests
# -------------------------------------------------------------------------------------


def test_wsireader_slide_info(sample_svs: Path, tmp_path: Path) -> None:
    """Test for slide_info in WSIReader class as a python function."""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(Path(sample_svs).parent),
        file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    slide_param = wsi.info
    out_path = tmp_path / slide_param.file_path.with_suffix(".yaml").name
    utils.misc.save_yaml(slide_param.as_dict(), out_path)


def test_wsireader_slide_info_cache(sample_svs: Path) -> None:
    """Test for caching slide_info in WSIReader class as a python function."""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(Path(sample_svs).parent),
        file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    info = wsi.info
    cached_info = wsi.info
    assert info.as_dict() == cached_info.as_dict()


# -------------------------------------------------------------------------------------
# Class-Specific Tests
# -------------------------------------------------------------------------------------


def test_relative_level_scales_openslide_baseline(sample_ndpi: Path) -> None:
    """Test openslide relative level scales for pixels per baseline pixel."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    relative_level_scales_baseline(wsi)


def test_relative_level_scales_jp2_baseline(sample_jp2: Path) -> None:
    """Test jp2 relative level scales for pixels per baseline pixel."""
    wsi = wsireader.JP2WSIReader(sample_jp2)
    relative_level_scales_baseline(wsi)


def test_relative_level_scales_openslide_mpp(sample_ndpi: Path) -> None:
    """Test openslide calculation of relative level scales for mpp."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    level_scales = wsi.info.relative_level_scales(0.5, "mpp")
    level_scales = np.array(level_scales)
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert all(level_scales[0] == wsi.info.mpp / 0.5)


def test_relative_level_scales_jp2_mpp(sample_jp2: Path) -> None:
    """Test jp2 calculation of relative level scales for mpp."""
    wsi = wsireader.JP2WSIReader(sample_jp2)
    level_scales = wsi.info.relative_level_scales(0.5, "mpp")
    level_scales = np.array(level_scales)
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert all(level_scales[0] == wsi.info.mpp / 0.5)


def relative_level_scales_power(wsi: WSIReader) -> None:
    """Calculation of relative level scales for objective power."""
    level_scales = wsi.info.relative_level_scales(wsi.info.objective_power, "power")
    level_scales = np.array(level_scales)
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert np.array_equal(level_scales[0], [1, 1])
    downsamples = np.array(wsi.info.level_downsamples)
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], downsamples)


def test_relative_level_scales_openslide_power(sample_ndpi: Path) -> None:
    """Test openslide calculation of relative level scales for objective power."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    relative_level_scales_power(wsi)


def test_relative_level_scales_jp2_power(sample_jp2: Path) -> None:
    """Test jp2 calculation of relative level scales for objective power."""
    wsi = wsireader.JP2WSIReader(sample_jp2)
    relative_level_scales_power(wsi)


def relative_level_scales_level(wsi: WSIReader) -> None:
    """Calculation of relative level scales for level."""
    level_scales = wsi.info.relative_level_scales(3, "level")
    level_scales = np.array(level_scales)
    assert np.array_equal(level_scales[3], [1, 1])
    downsamples = np.array(wsi.info.level_downsamples)
    expected = downsamples / downsamples[3]
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test_relative_level_scales_openslide_level(sample_ndpi: Path) -> None:
    """Test openslide calculation of relative level scales for level."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    relative_level_scales_level(wsi)


def test_relative_level_scales_jp2_level(sample_jp2: Path) -> None:
    """Test jp2 calculation of relative level scales for level."""
    wsi = wsireader.JP2WSIReader(sample_jp2)
    relative_level_scales_level(wsi)


def relative_level_scales_float(wsi: WSIReader) -> None:
    """Calculation of relative level scales for fractional level."""
    level_scales = wsi.info.relative_level_scales(1.5, "level")
    level_scales = np.array(level_scales)
    assert level_scales[0] == pytest.approx([1 / 3, 1 / 3])
    downsamples = np.array(wsi.info.level_downsamples)
    expected = downsamples / downsamples[0] * (1 / 3)
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test_relative_level_scales_openslide_level_float(sample_ndpi: Path) -> None:
    """Test openslide calculation of relative level scales for fractional level."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    relative_level_scales_float(wsi)


def test_relative_level_scales_jp2_level_float(sample_jp2: Path) -> None:
    """Test jp2 calculation of relative level scales for fractional level."""
    wsi = wsireader.JP2WSIReader(sample_jp2)
    relative_level_scales_float(wsi)


def test_relative_level_scales_invalid_units(sample_svs: Path) -> None:
    """Test relative_level_scales with invalid units."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    with pytest.raises(ValueError, match="Invalid units"):
        wsi.info.relative_level_scales(1.0, "gibberish")


def test_relative_level_scales_no_mpp() -> None:
    """Test relative_level_scales objective when mpp is None."""

    class DummyWSI:
        """Mock WSIReader for testing."""

        @property
        def info(self: DummyWSI) -> WSIMeta:
            return wsireader.WSIMeta((100, 100), axes="YXS")

    wsi = DummyWSI()
    with pytest.raises(ValueError, match="MPP is None"):
        wsi.info.relative_level_scales(1.0, "mpp")


def test_relative_level_scales_no_objective_power() -> None:
    """Test relative_level_scales objective when objective power is None."""

    class DummyWSI:
        """Mock WSIReader for testing."""

        @property
        def info(self: DummyWSI) -> WSIMeta:
            return wsireader.WSIMeta((100, 100), axes="YXS")

    wsi = DummyWSI()
    with pytest.raises(ValueError, match="Objective power is None"):
        wsi.info.relative_level_scales(10, "power")


def test_relative_level_scales_level_too_high(sample_svs: Path) -> None:
    """Test relative_level_scales levels set too high."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    with pytest.raises(ValueError, match="levels"):
        wsi.info.relative_level_scales(100, "level")


def test_find_optimal_level_and_downsample_openslide_interpolation_warning(
    sample_ndpi: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test finding optimal level for mpp read with scale > 1.

    This tests the case where the scale is found to be > 1 and interpolation
    will be applied to the output. A UserWarning should be raised in this case.

    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    _, _ = wsi._find_optimal_level_and_downsample(0.1, "mpp")
    assert (
        "Read: Scale > 1.This means that the desired resolution is higher"
        in caplog.text
    )


def test_find_optimal_level_and_downsample_jp2_interpolation_warning(
    sample_jp2: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test finding optimal level for mpp read with scale > 1.

    This tests the case where the scale is found to be > 1 and interpolation
    will be applied to the output. A UserWarning should be raised in this case.

    """
    wsi = wsireader.JP2WSIReader(sample_jp2)
    _, _ = wsi._find_optimal_level_and_downsample(0.1, "mpp")
    assert (
        "Read: Scale > 1.This means that the desired resolution is higher"
        in caplog.text
    )


def test_find_optimal_level_and_downsample_mpp(sample_ndpi: Path) -> None:
    """Test finding optimal level for mpp read."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)

    mpps = [0.5, 10]
    expected_levels = [0, 4]
    expected_scales = [[0.91282519, 0.91012514], [0.73026016, 0.72810011]]

    for mpp, expected_level, expected_scale in zip(
        mpps,
        expected_levels,
        expected_scales,
    ):
        read_level, post_read_scale_factor = wsi._find_optimal_level_and_downsample(
            mpp,
            "mpp",
        )

        assert read_level == expected_level
        assert post_read_scale_factor == pytest.approx(expected_scale)


def test_find_optimal_level_and_downsample_power(sample_ndpi: Path) -> None:
    """Test finding optimal level for objective power read."""
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)

    objective_powers = [20, 10, 5, 2.5, 1.25]
    expected_levels = [0, 1, 2, 3, 4]
    for objective_power, expected_level in zip(objective_powers, expected_levels):
        read_level, post_read_scale_factor = wsi._find_optimal_level_and_downsample(
            objective_power,
            "power",
        )

        assert read_level == expected_level
        assert np.array_equal(post_read_scale_factor, [1.0, 1.0])


def test_find_optimal_level_and_downsample_level(sample_ndpi: Path) -> None:
    """Test finding optimal level for level read.

    For integer levels, the returned level should always be the same as
    the input level.

    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)

    for level in range(wsi.info.level_count):
        read_level, post_read_scale_factor = wsi._find_optimal_level_and_downsample(
            level,
            "level",
        )

        assert read_level == level
        assert np.array_equal(post_read_scale_factor, [1.0, 1.0])


def test_convert_resolution_units(
    sample_ndpi: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        gt_power / 2,
        input_unit="power",
        output_unit="mpp",
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
    _ = wsi.convert_resolution_units(0, input_unit="level", output_unit="mpp")
    assert "output_unit is returned as None." in caplog.text


def test_find_read_rect_params_power(sample_ndpi: Path) -> None:
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


def test_find_read_rect_params_mpp(sample_ndpi: Path) -> None:
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


def test_read_rect_openslide_baseline(sample_ndpi: Path) -> None:
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


def test_read_rect_jp2_baseline(sample_jp2: Path) -> None:
    """Test jp2 read rect at baseline.

    Location coordinate is in baseline (level 0) reference frame.

    """
    wsi = wsireader.JP2WSIReader(sample_jp2)
    location = JP2_TEST_TISSUE_LOCATION
    size = JP2_TEST_TISSUE_SIZE
    im_region = wsi.read_rect(location, size, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)


def test_read_rect_tiffreader_svs_baseline(sample_svs: Path) -> None:
    """Test TIFFWSIReader.read_rect with an SVS file at baseline."""
    wsi = wsireader.TIFFWSIReader(sample_svs)
    location = SVS_TEST_TISSUE_LOCATION
    size = SVS_TEST_TISSUE_SIZE
    im_region = wsi.read_rect(location, size, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)


def test_read_rect_tiffreader_ome_tiff_baseline(sample_ome_tiff: Path) -> None:
    """Test TIFFWSIReader.read_rect with an OME-TIFF file at baseline."""
    wsi = wsireader.TIFFWSIReader(sample_ome_tiff)
    location = SVS_TEST_TISSUE_LOCATION
    size = SVS_TEST_TISSUE_SIZE
    im_region = wsi.read_rect(location, size, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)


def test_is_tiled_tiff(source_image: Path) -> None:
    """Test if source_image is a tiled tiff."""
    source_image.replace(source_image.with_suffix(".tiff"))
    assert wsireader.is_tiled_tiff(source_image.with_suffix(".tiff")) is False
    source_image.with_suffix(".tiff").replace(source_image)


def test_read_rect_openslide_levels(sample_ndpi: Path) -> None:
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


def test_read_rect_jp2_levels(sample_jp2: Path) -> None:
    """Test jp2 read rect with resolution in levels.

    Location coordinate is in baseline (level 0) reference frame.

    """
    wsi = wsireader.JP2WSIReader(sample_jp2)
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


def read_rect_mpp(wsi: WSIReader, location: IntPair, size: IntPair) -> None:
    """Read rect with resolution in microns per pixel."""
    for factor in range(1, 10):
        mpp = wsi.info.mpp * factor
        im_region = wsi.read_rect(location, size, resolution=mpp, units="mpp")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert im_region.shape == (*size[::-1], 3)


def test_read_rect_openslide_mpp(sample_ndpi: Path) -> None:
    """Test openslide read rect with resolution in microns per pixel.

    Location coordinate is in baseline (level 0) reference frame.

    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    location = NDPI_TEST_TISSUE_LOCATION
    size = NDPI_TEST_TISSUE_SIZE
    read_rect_mpp(wsi, location, size)


def test_read_rect_jp2_mpp(sample_jp2: Path) -> None:
    """Test jp2 read rect with resolution in microns per pixel.

    Location coordinate is in baseline (level 0) reference frame.

    """
    wsi = wsireader.JP2WSIReader(sample_jp2)
    location = JP2_TEST_TISSUE_LOCATION
    size = JP2_TEST_TISSUE_SIZE
    read_rect_mpp(wsi, location, size)


def test_read_rect_openslide_objective_power(sample_ndpi: Path) -> None:
    """Test openslide read rect with resolution in objective power.

    Location coordinate is in baseline (level 0) reference frame.

    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    location = NDPI_TEST_TISSUE_LOCATION
    size = NDPI_TEST_TISSUE_SIZE

    read_rect_objective_power(wsi, location, size)


def test_read_rect_jp2_objective_power(sample_jp2: Path) -> None:
    """Test jp2 read rect with resolution in objective power.

    Location coordinate is in baseline (level 0) reference frame.

    """
    wsi = wsireader.JP2WSIReader(sample_jp2)
    location = JP2_TEST_TISSUE_LOCATION
    size = JP2_TEST_TISSUE_SIZE

    read_rect_objective_power(wsi, location, size)


def test_read_bounds_openslide_baseline(sample_ndpi: Path) -> None:
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


def test_read_bounds_jp2_baseline(sample_jp2: Path) -> None:
    """Test JP2 read bounds at baseline.

    Coordinates in baseline (level 0) reference frame.

    """
    wsi = wsireader.JP2WSIReader(sample_jp2)
    bounds = JP2_TEST_TISSUE_BOUNDS
    size = JP2_TEST_TISSUE_SIZE
    im_region = wsi.read_bounds(bounds, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)


def test_read_bounds_openslide_levels(sample_ndpi: Path) -> None:
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
            np.round([height / downsample, width / downsample, 3]).astype(int),
        )
        assert im_region.shape == expected_output_shape


def test_read_bounds_jp2_levels(sample_jp2: Path) -> None:
    """Test jp2 read bounds with resolution in levels.

    Coordinates in baseline (level 0) reference frame.

    """
    wsi = wsireader.JP2WSIReader(sample_jp2)
    bounds = JP2_TEST_TISSUE_BOUNDS
    width, height = JP2_TEST_TISSUE_SIZE
    for level, downsample in enumerate(wsi.info.level_downsamples):
        im_region = wsi.read_bounds(bounds, resolution=level, units="level")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        expected_output_shape = tuple(
            np.round([height / downsample, width / downsample]),
        )
        assert im_region.shape[:2] == pytest.approx(expected_output_shape, abs=1)
        assert im_region.shape[2] == 3


def test_read_bounds_openslide_mpp(sample_ndpi: Path) -> None:
    """Test openslide read bounds with resolution in microns per pixel.

    Coordinates in baseline (level 0) reference frame.

    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    bounds = NDPI_TEST_TISSUE_BOUNDS
    size = NDPI_TEST_TISSUE_SIZE

    read_bounds_mpp(wsi, bounds, size, jp2=False)


def test_read_bounds_jp2_mpp(sample_jp2: Path) -> None:
    """Test jp2 read bounds with resolution in microns per pixel.

    Coordinates in baseline (level 0) reference frame.

    """
    wsi = wsireader.JP2WSIReader(sample_jp2)
    bounds = JP2_TEST_TISSUE_BOUNDS
    size = JP2_TEST_TISSUE_SIZE

    read_bounds_mpp(wsi, bounds, size, jp2=True)


def test_read_bounds_openslide_objective_power(sample_ndpi: Path) -> None:
    """Test openslide read bounds with resolution in objective power.

    Coordinates in baseline (level 0) reference frame.

    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    bounds = NDPI_TEST_TISSUE_BOUNDS
    size = NDPI_TEST_TISSUE_SIZE
    slide_power = wsi.info.objective_power

    read_bounds_objective_power(wsi, slide_power, bounds, size, jp2=False)


def test_read_bounds_jp2_objective_power(sample_jp2: Path) -> None:
    """Test jp2 read bounds with resolution in objective power.

    Coordinates in baseline (level 0) reference frame.

    """
    wsi = wsireader.JP2WSIReader(sample_jp2)
    bounds = JP2_TEST_TISSUE_BOUNDS
    size = JP2_TEST_TISSUE_SIZE
    slide_power = wsi.info.objective_power

    read_bounds_objective_power(wsi, slide_power, bounds, size, jp2=True)


def test_read_bounds_interpolated(sample_svs: Path) -> None:
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

    assert wsi.info.mpp[0] > 0.1
    assert wsi.info.mpp[1] > 0.1
    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape[2] == 3
    assert all(np.array(im_region.shape[:2]) > size)


def test_read_bounds_level_consistency_openslide(sample_ndpi: Path) -> None:
    """Test read_bounds produces the same visual field across resolution levels.

    with OpenSlideWSIReader.

    """
    wsi = wsireader.OpenSlideWSIReader(sample_ndpi)
    bounds = NDPI_TEST_TISSUE_BOUNDS

    read_bounds_level_consistency(wsi, bounds)


def test_read_bounds_level_consistency_jp2(sample_jp2: Path) -> None:
    """Test read_bounds produces the same visual field across resolution levels.

    Using JP2WSIReader.

    """
    bounds = JP2_TEST_TISSUE_BOUNDS
    wsi = wsireader.JP2WSIReader(sample_jp2)

    read_bounds_level_consistency(wsi, bounds)


def test_wsireader_save_tiles(sample_svs: Path, tmp_path: Path) -> None:
    """Test for save_tiles in wsireader as a python function."""
    tmp_path = Path(tmp_path)
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(Path(sample_svs).parent),
        file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    wsi.save_tiles(
        output_dir=str(tmp_path / "test_wsireader_save_tiles"),
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


def test_incompatible_objective_value(sample_svs: Path, tmp_path: Path) -> None:
    """Test for incompatible objective value."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    with pytest.raises(ValueError, match="objective power"):
        wsi.save_tiles(
            output_dir=str(
                Path(tmp_path).joinpath("test_wsireader_save_tiles"),
            ),
            tile_objective_value=3,
            tile_read_size=(5000, 5000),
            verbose=True,
        )


def test_incompatible_level(
    sample_svs: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test for incompatible objective value."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    wsi.save_tiles(
        output_dir=str(Path(tmp_path).joinpath("test_wsireader_save_tiles2")),
        tile_objective_value=1,
        tile_read_size=(500, 500),
        verbose=True,
    )

    assert "Reading at tile_objective_value 1 not allowed" in caplog.text


def test_wsireader_jp2_save_tiles(sample_jp2: Path, tmp_path: Path) -> None:
    """Test for save_tiles in wsireader as a python function."""
    tmp_path = Path(tmp_path)
    wsi = wsireader.JP2WSIReader(sample_jp2)
    wsi.save_tiles(
        output_dir=str(tmp_path / "test_wsireader_jp2_save_tiles"),
        tile_objective_value=5,
        tile_read_size=(5000, 5000),
        verbose=True,
    )
    assert (
        tmp_path
        / "test_wsireader_jp2_save_tiles"
        / "CMU-1-Small-Region.omnyx.jp2"
        / "Output.csv"
    ).exists()
    assert (
        tmp_path
        / "test_wsireader_jp2_save_tiles"
        / "CMU-1-Small-Region.omnyx.jp2"
        / "slide_thumbnail.jpg"
    ).exists()
    assert (
        tmp_path
        / "test_wsireader_jp2_save_tiles"
        / "CMU-1-Small-Region.omnyx.jp2"
        / "Tile_5_0_0.jpg"
    ).exists()


def test_openslide_objective_power_from_mpp(
    sample_svs: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test OpenSlideWSIReader approximation of objective power from mpp."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    wsi.openslide_wsi = DummyMutableOpenSlideObject(wsi.openslide_wsi)
    props = wsi.openslide_wsi._properties

    del props["openslide.objective-power"]  # skipcq
    _ = wsi.info
    assert "Objective power inferred" in caplog.text

    del props["openslide.mpp-x"]  # skipcq
    del props["openslide.mpp-y"]  # skipcq
    _ = wsi._info()
    assert "Unable to determine objective power" in caplog.text


def test_openslide_mpp_from_tiff_resolution(
    sample_svs: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test OpenSlideWSIReader mpp from TIFF resolution tags."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    wsi.openslide_wsi = DummyMutableOpenSlideObject(wsi.openslide_wsi)
    props = wsi.openslide_wsi._properties

    del props["openslide.mpp-x"]  # skipcq
    del props["openslide.mpp-y"]  # skipcq
    props["tiff.ResolutionUnit"] = "centimeter"
    props["tiff.XResolution"] = 1e4  # Pixels per cm
    props["tiff.YResolution"] = 1e4  # Pixels per cm
    _ = wsi.info
    assert "Falling back to TIFF resolution" in caplog.text

    assert np.array_equal(wsi.info.mpp, [1, 1])


def test_virtual_wsi_reader(
    source_image: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test VirtualWSIReader."""
    wsi = wsireader.VirtualWSIReader(Path(source_image))
    _ = wsi._info()
    assert "Unknown scale" in caplog.text

    _ = wsi._info()
    assert "Raw data is None" in caplog.text

    assert wsi.img.shape == (256, 256, 3)

    img = wsi.read_rect(location=(0, 0), size=(100, 50))
    assert img.shape == (50, 100, 3)

    img = wsi.read_region(location=(0, 0), size=(100, 50), level=0)
    assert img.shape == (50, 100, 3)


def test_virtual_wsi_reader_invalid_mode(source_image: Path) -> None:
    """Test creating a VritualWSIReader with an invalid mode."""
    with pytest.raises(ValueError, match="Invalid mode"):
        wsireader.VirtualWSIReader(Path(source_image), mode="foo")


def test_virtual_wsi_reader_read_bounds(source_image: Path) -> None:
    """Test VirtualWSIReader read bounds."""
    wsi = wsireader.VirtualWSIReader(Path(source_image))
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


def test_virtual_wsi_reader_read_rect(source_image: Path) -> None:
    """Test VirtualWSIReader read rect."""
    wsi = wsireader.VirtualWSIReader(Path(source_image))
    info = wsi.info

    img = wsi.read_rect(location=(0, 0), size=(50, 100))
    assert img.shape == (100, 50, 3)

    img = wsi.read_rect(
        location=(0, 0),
        size=(50, 100),
        resolution=1.5,
        units="baseline",
    )
    assert img.shape == (100, 50, 3)

    img = wsi.read_rect(
        location=(0, 0),
        size=(50, 100),
        resolution=0.5,
        units="baseline",
    )
    assert img.shape == (100, 50, 3)

    with pytest.raises(IndexError):
        _ = wsi.read_rect(
            location=(0, 0),
            size=(50, 100),
            resolution=0.5,
            units="level",
        )

    with pytest.raises(ValueError, match="level"):
        _ = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=1, units="level")

    wsi = wsireader.VirtualWSIReader(Path(source_image), info=info)

    assert info.as_dict() == wsi.info.as_dict()


def test_virtual_wsi_reader_read_bounds_virtual_baseline(source_image: Path) -> None:
    """Test VirtualWSIReader read bounds with virtual baseline."""
    image_path = Path(source_image)
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
        img_array[:50, :25, :],
        target_size,
        interpolation=cv2.INTER_CUBIC,
    )

    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.1


def test_virtual_wsi_reader_read_rect_virtual_baseline(source_image: Path) -> None:
    """Test VirtualWSIReader read rect with virtual baseline.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image.

    """
    img_array = utils.misc.imread(Path(source_image))
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(slide_dimensions=double_size, axes="YXS")
    wsi = wsireader.VirtualWSIReader(Path(source_image), info=meta)
    region = wsi.read_rect(location=(0, 0), size=(50, 100), pad_mode="reflect")
    target = cv2.resize(
        img_array[:50, :25, :],
        (50, 100),
        interpolation=cv2.INTER_CUBIC,
    )
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2


def test_virtual_wsi_reader_read_rect_virtual_levels(source_image: Path) -> None:
    """Test VirtualWSIReader read rect with vritual levels.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image and the pyramid/resolution levels.
    Checks that the regions read at each level line up with expected values.

    """
    img_array = utils.misc.imread(Path(source_image))
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(
        slide_dimensions=double_size,
        level_downsamples=[1, 2, 4],
        axes="YXS",
    )
    wsi = wsireader.VirtualWSIReader(Path(source_image), info=meta)
    region = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=1, units="level")
    target = img_array[:100, :50, :]
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 1

    region = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=2, units="level")
    target = cv2.resize(img_array[:200, :100, :], (50, 100))
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 1


def test_virtual_wsi_reader_read_bounds_virtual_levels(source_image: Path) -> None:
    """Test VirtualWSIReader read bounds with vritual levels.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image and the pyramid/resolution levels.
    Checks that the regions read at each level line up with expected values.

    """
    img_array = utils.misc.imread(Path(source_image))
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(
        slide_dimensions=double_size,
        level_downsamples=[1, 2, 4],
        axes="YXS",
    )
    wsi = wsireader.VirtualWSIReader(Path(source_image), info=meta)
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
        img_array[:50, :25, :],
        target_size,
        interpolation=cv2.INTER_CUBIC,
    )
    offset, error, _ = phase_cross_correlation(target, region, normalization=None)
    assert all(offset == 0)
    assert error < 0.1
    psnr = peak_signal_noise_ratio(target, region)
    assert psnr < 50


def test_virtual_wsi_reader_read_rect_virtual_levels_mpp(source_image: Path) -> None:
    """Test VirtualWSIReader read rect with vritual levels and MPP.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image and the pyramid/resolution levels and
    a baseline MPP specified.
    Checks that the regions read with specified MPP for each level lines up
    with expected values.

    """
    img_array = utils.misc.imread(Path(source_image))
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(
        slide_dimensions=double_size,
        axes="YXS",
        level_downsamples=[1, 2, 4],
        mpp=(0.25, 0.25),
    )
    wsi = wsireader.VirtualWSIReader(Path(source_image), info=meta)
    region = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=0.5, units="mpp")
    target = img_array[:100, :50, :]
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 1

    region = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=1, units="mpp")
    target = cv2.resize(
        img_array[:200, :100, :],
        (50, 100),
        interpolation=cv2.INTER_CUBIC,
    )
    offset, error, _ = phase_cross_correlation(target, region, normalization=None)
    assert all(offset == 0)
    assert error < 0.1
    psnr = peak_signal_noise_ratio(target, region)
    assert psnr < 50


def test_virtual_wsi_reader_read_bounds_virtual_levels_mpp(source_image: Path) -> None:
    """Test VirtualWSIReader read bounds with vritual levels and MPP.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image and the pyramid/resolution levels.
    Checks that the regions read at each level line up with expected values.

    """
    img_array = utils.misc.imread(Path(source_image))
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(
        slide_dimensions=double_size,
        axes="YXS",
        level_downsamples=[1, 2, 4],
        mpp=(0.25, 0.25),
    )
    wsi = wsireader.VirtualWSIReader(Path(source_image), info=meta)
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
        img_array[:50, :25, :],
        target_size,
        interpolation=cv2.INTER_CUBIC,
    )
    offset, error, _ = phase_cross_correlation(target, region, normalization=None)
    assert all(offset == 0)
    assert error < 0.1
    psnr = peak_signal_noise_ratio(target, region)
    assert psnr < 50


def test_tissue_mask_otsu(sample_svs: Path) -> None:
    """Test wsi.tissue_mask with Otsu's method."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)

    tissue_thumb = wsi.slide_thumbnail()
    grey_thumb = cv2.cvtColor(tissue_thumb, cv2.COLOR_RGB2GRAY)

    otsu_threhold = threshold_otsu(grey_thumb)
    otsu_mask = grey_thumb < otsu_threhold

    mask = wsi.tissue_mask(method="otsu")
    mask_thumb = mask.slide_thumbnail()

    assert np.mean(np.logical_xor(mask_thumb, otsu_mask)) < 0.05


def test_tissue_mask_morphological(sample_svs: Path) -> None:
    """Test wsi.tissue_mask with morphological method."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    resolutions = [5, 10]
    units = ["power", "mpp"]
    scale_fns = [lambda x: x * 2, lambda x: 32 / x]
    for unit, scaler in zip(units, scale_fns):
        for resolution in resolutions:
            mask = wsi.tissue_mask(
                method="morphological",
                resolution=resolution,
                units=unit,
            )

            tissue_thumb = wsi.slide_thumbnail(resolution, unit)
            grey_thumb = tissue_thumb.mean(axis=-1)
            mask_thumb = mask.slide_thumbnail(resolution, unit)

            otsu_threhold = threshold_otsu(grey_thumb)
            otsu_mask = grey_thumb < otsu_threhold
            morpho_mask = binary_dilation(otsu_mask, disk(scaler(resolution)))
            morpho_mask = remove_small_objects(morpho_mask, 100 * scaler(resolution))

    assert np.mean(np.logical_xor(mask_thumb, morpho_mask)) < 0.1


def test_tissue_mask_morphological_levels(sample_svs: Path) -> None:
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


def test_tissue_mask_read_bounds_none_interpolation(sample_svs: Path) -> None:
    """Test reading a mask using read_bounds with no interpolation."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    mask = wsi.tissue_mask("otsu")
    mask_region = mask.read_bounds((0, 0, 512, 512), interpolation="none")
    assert mask_region.shape[0] == 32
    assert mask_region.shape[1] == 32


def test_tissue_mask_read_rect_none_interpolation(sample_svs: Path) -> None:
    """Test reading a mask using read_rect with no interpolation."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    mask = wsi.tissue_mask("otsu")
    mask_region = mask.read_rect((0, 0), (512, 512), interpolation="none")
    assert mask_region.shape[0] == 32
    assert mask_region.shape[1] == 32


def test_invalid_masker_method(sample_svs: Path) -> None:
    """Test that an invalid masking method string raises a ValueError."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    with pytest.raises(ValueError, match="masking method"):
        wsi.tissue_mask(method="foo")


def test_wsireader_open(
    sample_svs: Path,
    sample_ndpi: Path,
    sample_jp2: Path,
    sample_ome_tiff: Path,
    sample_ventana_tif: Path,
    sample_regular_tif: Path,
    sample_qptiff: Path,
    source_image: Path,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Test WSIReader.open() to return correct object."""
    with pytest.raises(FileNotSupportedError):
        _ = WSIReader.open("./sample.csv")

    with pytest.raises(TypeError):
        _ = WSIReader.open([1, 2])

    wsi = WSIReader.open(sample_svs)
    assert isinstance(wsi, wsireader.OpenSlideWSIReader)

    wsi = WSIReader.open(sample_ndpi)
    assert isinstance(wsi, wsireader.OpenSlideWSIReader)

    wsi = WSIReader.open(sample_jp2)
    assert isinstance(wsi, wsireader.JP2WSIReader)

    wsi = WSIReader.open(sample_ome_tiff)
    assert isinstance(wsi, wsireader.TIFFWSIReader)

    wsi = WSIReader.open(sample_ventana_tif)
    assert isinstance(wsi, wsireader.OpenSlideWSIReader)

    wsi = WSIReader.open(sample_regular_tif)
    assert isinstance(wsi, wsireader.VirtualWSIReader)

    wsi = WSIReader.open(Path(source_image))
    assert isinstance(wsi, wsireader.VirtualWSIReader)

    wsi = WSIReader.open(sample_qptiff)
    assert isinstance(wsi, wsireader.TIFFWSIReader)

    img = utils.misc.imread(str(Path(source_image)))
    wsi = WSIReader.open(input_img=img)
    assert isinstance(wsi, wsireader.VirtualWSIReader)

    # test if WSIReader.open can accept a wsireader instance
    wsi_type = type(wsi)
    wsi_out = WSIReader.open(input_img=wsi)
    assert isinstance(wsi_out, wsi_type)

    # test loading .npy
    temp_file = str(tmp_path / "sample.npy")
    np.save(temp_file, RNG.integers(1, 255, [5, 5, 5]))
    wsi_out = WSIReader.open(temp_file)
    assert isinstance(wsi_out, VirtualWSIReader)


def test_jp2_missing_cod(sample_jp2: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test for warning if JP2 is missing COD segment."""
    wsi = wsireader.JP2WSIReader(sample_jp2)
    wsi.glymur_jp2.codestream.segment = []
    _ = wsi.info
    assert "missing COD" in caplog.text


def test_read_rect_at_resolution(sample_wsi_dict: dict) -> None:
    """Test for read rect using location at requested."""
    mini_wsi2_svs = Path(sample_wsi_dict["wsi1_8k_8k_svs"])
    mini_wsi2_jpg = Path(sample_wsi_dict["wsi1_8k_8k_jpg"])
    mini_wsi2_jp2 = Path(sample_wsi_dict["wsi1_8k_8k_jp2"])

    # * check sync read between Virtual Reader and WSIReader (openslide) (reference)
    reader_list = [
        VirtualWSIReader(mini_wsi2_jpg),
        OpenSlideWSIReader(mini_wsi2_svs),
        JP2WSIReader(mini_wsi2_jp2),
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


def test_read_bounds_location_in_requested_resolution(  # noqa: PLR0915
    sample_wsi_dict: dict,
) -> None:
    """Actually a duel test for sync read and read at requested."""
    # """Test synchronize read for VirtualReader"""
    # convert to pathlib Path to prevent wsireader complaint
    mini_wsi1_msk = Path(sample_wsi_dict["wsi2_4k_4k_msk"])
    mini_wsi2_svs = Path(sample_wsi_dict["wsi1_8k_8k_svs"])
    mini_wsi2_jpg = Path(sample_wsi_dict["wsi1_8k_8k_jpg"])
    mini_wsi2_jp2 = Path(sample_wsi_dict["wsi1_8k_8k_jp2"])

    def compare_reader(
        reader1: WSIReader,
        reader2: WSIReader,
        read_coord: IntBounds,
        read_cfg: dict,
        *,
        check_content: bool,
    ) -> None:
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
        msk,
        (0, 0),
        fx=4.0,
        fy=4.0,
        interpolation=cv2.INTER_NEAREST,
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
        read_coord_ = requested_coords if read_coord is None else read_coord
        compare_reader(
            msk_reader,
            bigger_msk_reader,
            read_coord_,
            read_cfg,
            check_content=True,
        )

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
        read_coord_ = requested_coords if read_coord is None else read_coord
        compare_reader(
            wsi_reader,
            vrt_reader,
            read_coord_,
            read_cfg,
            check_content=True,
        )

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
    wsi_reader = JP2WSIReader(mini_wsi2_jp2)
    wsi_thumb = wsi_reader.slide_thumbnail(resolution=0.85, units="mpp")
    vrt_reader = VirtualWSIReader(wsi_thumb)
    vrt_reader.info = wsi_reader.info

    for _, (read_cfg, read_coord) in enumerate(read_cfg_list):
        read_coord_ = requested_coords if read_coord is None else read_coord
        compare_reader(
            wsi_reader,
            vrt_reader,
            read_coord_,
            read_cfg,
            check_content=True,
        )


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_read_bounds(sample_ndpi: Path, tmp_path: Path) -> None:
    """Test OpenSlide read_bounds CLI."""
    runner = CliRunner()
    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--img-input",
            str(Path(sample_ndpi)),
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
            str(Path(tmp_path).joinpath("im_region.jpg")),
        ],
    )

    assert read_bounds_result.exit_code == 0
    assert Path(tmp_path).joinpath("im_region.jpg").is_file()

    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--img-input",
            str(Path(sample_ndpi)),
            "--resolution",
            "0",
            "--units",
            "level",
            "--mode",
            "save",
            "--output-path",
            str(Path(tmp_path).joinpath("im_region2.jpg")),
        ],
    )

    assert read_bounds_result.exit_code == 0
    assert Path(tmp_path).joinpath("im_region2.jpg").is_file()


def test_command_line_jp2_read_bounds(sample_jp2: Path, tmp_path: Path) -> None:
    """Test JP2 read_bounds."""
    runner = CliRunner()
    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--img-input",
            str(Path(sample_jp2)),
            "--resolution",
            "0",
            "--units",
            "level",
            "--mode",
            "save",
        ],
    )

    assert read_bounds_result.exit_code == 0
    assert Path(tmp_path).joinpath("../im_region.jpg").is_file()


@pytest.mark.skipif(
    utils.env_detection.running_on_ci(),
    reason="No need to display image on travis.",
)
def test_command_line_jp2_read_bounds_show(sample_jp2: Path) -> None:
    """Test JP2 read_bounds with mode as 'show'."""
    runner = CliRunner()
    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--img-input",
            str(Path(sample_jp2)),
            "--resolution",
            "0",
            "--units",
            "level",
            "--mode",
            "show",
        ],
    )

    assert read_bounds_result.exit_code == 0


def test_command_line_unsupported_file_read_bounds(sample_svs: Path) -> None:
    """Test unsupported file read bounds."""
    runner = CliRunner()
    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--img-input",
            str(Path(sample_svs))[:-1],
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
    assert isinstance(read_bounds_result.exception, FileNotSupportedError)


def test_openslide_read_rect_edge_reflect_padding(sample_svs: Path) -> None:
    """Test openslide edge reflect padding for read_rect."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    region = wsi.read_rect((-64, -64), (128, 128), pad_mode="reflect")
    assert 0 not in region.min(axis=-1)


def test_openslide_read_bounds_edge_reflect_padding(sample_svs: Path) -> None:
    """Test openslide edge reflect padding for read_bounds."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    region = wsi.read_bounds((-64, -64, 64, 64), pad_mode="reflect")
    assert 0 not in region.min(axis=-1)


def test_tiffwsireader_invalid_tiff(remote_sample: Callable) -> None:
    """Test for TIFF which is not supported by TIFFWSIReader."""
    with pytest.raises(ValueError, match="Unsupported TIFF"):
        _ = wsireader.TIFFWSIReader(remote_sample("two-tiled-pages"))


def test_tiffwsireader_invalid_svs_metadata(
    sample_svs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test for invalid SVS key-value pairs in TIFF escription tag."""
    wsi = wsireader.TIFFWSIReader(sample_svs)
    monkeypatch.setattr(
        wsi.tiff.pages[0],
        "description",
        wsi.tiff.pages[0].description.replace("=", "=="),
    )
    with pytest.raises(ValueError, match="key=value"):
        _ = wsi._info()


def test_tiffwsireader_invalid_ome_metadata(
    sample_ome_tiff_level_0: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test exception raised for invalid OME-XML metadata instrument."""
    wsi = wsireader.TIFFWSIReader(sample_ome_tiff_level_0)
    monkeypatch.setattr(
        wsi.tiff.pages[0],
        "description",
        wsi.tiff.pages[0].description.replace(
            '<Objective ID="Objective:0:0" NominalMagnification="20.0"/>',
            "",
        ),
    )
    with pytest.raises(KeyError, match="No matching Instrument"):
        _ = wsi._info()


def test_tiffwsireader_ome_metadata_missing_one_mppy(
    sample_ome_tiff: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test no exception raised for missing x/y mpp but warning given."""
    for dim in "XY":
        wsi = wsireader.TIFFWSIReader(sample_ome_tiff)
        monkeypatch.setattr(
            wsi.tiff.pages[0],
            "description",
            re.sub(f'PhysicalSize{dim}="[^"]*"', "", wsi.tiff.pages[0].description),
        )
        _ = wsi._info()
        assert "Only one MPP" in caplog.text


def test_arrayview_unsupported_axes() -> None:
    """Test unsupported axes in ArrayView."""
    array = zarr.ones((128, 128, 3))
    array_view = ArrayView(array=array, axes="FOO")
    with pytest.raises(ValueError, match="Unsupported axes"):
        array_view[:64, :64, :]


def test_arrayview_unsupported_axes_shape(
    sample_ome_tiff: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test accessing an unspported axes in TIFFWSIReader._shape_channels_last."""
    wsi = wsireader.TIFFWSIReader(sample_ome_tiff)
    monkeypatch.setattr(wsi, "_axes", "FOO")
    with pytest.raises(ValueError, match="Unsupported axes"):
        _ = wsi._info()


def test_arrayview_incomplete_index() -> None:
    """Test reading from ArrayView without specifying all axes slices."""
    array = zarr.array(RNG.random((128, 128, 3)))
    array_view = ArrayView(array=array, axes="YXS")
    view_1 = array_view[:64, :64, :]
    view_2 = array_view[:64, :64]
    assert np.array_equal(view_1, view_2)


def test_arrayview_single_number_index() -> None:
    """Test reading a column from ArrayView.

    I'm not sure why you would want to do this but it is implemented for
    consistency with other __getitem__ objects.

    """
    array = zarr.array(RNG.random((128, 128, 3)))
    array_view = ArrayView(array=array, axes="YXS")
    view_1 = array_view[0]
    view_2 = array_view[0]
    assert np.array_equal(view_1, view_2)


def test_manual_mpp_tuple(sample_svs: Path) -> None:
    """Test setting a manual mpp for a WSI."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs, mpp=(0.123, 0.123))
    assert tuple(wsi.info.mpp) == (0.123, 0.123)


def test_manual_mpp_float(sample_svs: Path) -> None:
    """Test setting a manual mpp for a WSI."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs, mpp=0.123)
    assert tuple(wsi.info.mpp) == (0.123, 0.123)


def test_manual_mpp_invalid(sample_svs: Path) -> None:
    """Test setting a manual mpp for a WSI."""
    with pytest.raises(TypeError, match="mpp"):
        _ = wsireader.OpenSlideWSIReader(sample_svs, mpp=(0.5,))
    with pytest.raises(TypeError, match="mpp"):
        _ = wsireader.OpenSlideWSIReader(sample_svs, mpp="foo")


def test_manual_power_tuple(sample_svs: Path) -> None:
    """Test setting a manual power for a WSI."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs, power=42)
    assert wsi.info.objective_power == 42


def test_manual_power_invalid(sample_svs: Path) -> None:
    """Test setting a manual power for a WSI."""
    with pytest.raises(TypeError, match="power"):
        _ = wsireader.OpenSlideWSIReader(sample_svs, power=(42,))


def test_tiled_tiff_openslide(remote_sample: Callable) -> None:
    """Test reading a tiled TIFF file with OpenSlide."""
    sample_path = remote_sample("tiled-tiff-1-small-jpeg")
    # Test with top-level import
    wsi = WSIReader.open(sample_path)
    assert isinstance(wsi, wsireader.OpenSlideWSIReader)


def test_tiled_tiff_tifffile(remote_sample: Callable) -> None:
    """Test fallback to tifffile for files which openslide cannot read.

    E.G. tiled tiffs with JPEG XL compression.

    """
    sample_path = remote_sample("tiled-tiff-1-small-jp2k")
    wsi = wsireader.WSIReader.open(sample_path)
    assert isinstance(wsi, wsireader.TIFFWSIReader)


def test_is_zarr_empty_dir(tmp_path: Path) -> None:
    """Test is_zarr is false for an empty .zarr directory."""
    zarr_dir = tmp_path / "zarr.zarr"
    zarr_dir.mkdir()
    assert not is_zarr(zarr_dir)


def test_is_zarr_array(tmp_path: Path) -> None:
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
    with Path.open(_zarray_path, "w") as f:
        json.dump(minimal_zarray, f)
    assert is_zarr(zarr_dir)


def test_is_zarr_group(tmp_path: Path) -> None:
    """Test is_zarr is true for a .zarr directory with an group."""
    zarr_dir = tmp_path / "zarr.zarr"
    zarr_dir.mkdir()
    _zgroup_path = zarr_dir / ".zgroup"
    minimal_zgroup = {
        "zarr_format": 2,
    }
    with Path.open(_zgroup_path, "w") as f:
        json.dump(minimal_zgroup, f)
    assert is_zarr(zarr_dir)


def test_is_ngff_regular_zarr(tmp_path: Path) -> None:
    """Test is_ngff is false for a regular zarr."""
    zarr_path = tmp_path / "zarr.zarr"
    # Create zarr array on disk
    zarr.array(RNG.random((32, 32)), store=zarr.DirectoryStore(zarr_path))
    assert is_zarr(zarr_path)
    assert not is_ngff(zarr_path)

    # check we get the appropriate error message if we open it
    with pytest.raises(FileNotSupportedError, match="does not appear to be a v0.4"):
        WSIReader.open(zarr_path)


def test_is_ngff_sqlite3(tmp_path: Path, remote_sample: Callable) -> None:
    """Test is_ngff is false for a sqlite3 file.

    Copies the ngff-1 sample to a sqlite3 file and checks that it is
    identified as an ngff file.

    """
    ngff_path = remote_sample("ngff-1")
    source = zarr.DirectoryStore(ngff_path)
    dest = zarr.SQLiteStore(tmp_path / "ngff.sqlite3")
    # Copy the store to a sqlite3 file
    zarr.copy_store(source, dest)

    assert is_sqlite3(dest.path)


def test_store_reader_no_info(tmp_path: Path) -> None:
    """Test AnnotationStoreReader with no info."""
    SQLiteStore(tmp_path / "store.db")
    with pytest.raises(ValueError, match="No metadata found"):
        AnnotationStoreReader(tmp_path / "store.db")


def test_store_reader_explicit_info(remote_sample: Callable, tmp_path: Path) -> None:
    """Test AnnotationStoreReader with explicit info."""
    SQLiteStore(tmp_path / "store.db")
    wsi_reader = WSIReader.open(remote_sample("svs-1-small"))
    reader = AnnotationStoreReader(tmp_path / "store.db", wsi_reader.info)
    assert reader._info().as_dict() == wsi_reader.info.as_dict()


def test_store_reader_from_store(remote_sample: Callable) -> None:
    """Test AnnotationStoreReader from an AnnotationStore object."""
    store = SQLiteStore(remote_sample("annotation_store_svs_1"))
    reader = AnnotationStoreReader(store)
    assert isinstance(reader.store, SQLiteStore)


def test_store_reader_base_wsi_str(remote_sample: Callable) -> None:
    """Test AnnotationStoreReader with base_wsi as a string."""
    store = SQLiteStore(remote_sample("annotation_store_svs_1"))
    reader = AnnotationStoreReader(store, base_wsi=remote_sample("svs-1-small"))
    assert isinstance(reader.store, SQLiteStore)
    assert isinstance(reader.base_wsi, WSIReader)


def test_store_reader_alpha(remote_sample: Callable) -> None:
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
        np.abs(store_thumb - wsi_thumb),
    )
    # the tile with low alpha should be closer to wsi_tile
    assert np.mean(np.abs(store_tile_alpha - wsi_tile)) < np.mean(
        np.abs(store_tile - wsi_tile),
    )


def test_store_reader_no_types(tmp_path: Path, remote_sample: Callable) -> None:
    """Test AnnotationStoreReader with no types."""
    SQLiteStore(tmp_path / "store.db")
    wsi_reader = WSIReader.open(remote_sample("svs-1-small"))
    reader = AnnotationStoreReader(tmp_path / "store.db", wsi_reader.info)
    # shouldn't try to color by type if not present
    assert reader.renderer.score_prop is None


def test_store_reader_info_from_base(tmp_path: Path, remote_sample: Callable) -> None:
    """Test AnnotationStoreReader with no wsi metadata.

    Test that AnnotationStoreReader will correctly get metadata
    from a provided base_wsi if the store has no wsi metadata.

    """
    SQLiteStore(tmp_path / "store.db")
    wsi_reader = WSIReader.open(remote_sample("svs-1-small"))
    store_reader = AnnotationStoreReader(tmp_path / "store.db", base_wsi=wsi_reader)
    # the store reader should have the same metadata as the base wsi
    assert store_reader.info.mpp[0] == wsi_reader.info.mpp[0]


def test_ngff_sqlitestore(tmp_path: Path, remote_sample: Callable) -> None:
    """Test SQLiteStore with an NGFF file."""
    ngff_path = remote_sample("ngff-1")
    source = zarr.DirectoryStore(ngff_path)
    dest = zarr.SQLiteStore(tmp_path / "ngff.sqlite3")
    # Copy the store to a sqlite3 file
    zarr.copy_store(source, dest)
    wsireader.NGFFWSIReader(dest.path)


def test_ngff_zattrs_non_micrometer_scale_mpp(
    tmp_path: Path,
    remote_sample: Callable,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that mpp is None if scale is not in micrometers."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample with a non-micrometer scale
    sample_copy = tmp_path / "ngff-1"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    zattrs["multiscales"][0]["axes"][0]["unit"] = "foo"
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)

    wsi = wsireader.NGFFWSIReader(sample_copy)
    assert "micrometer" in caplog.text

    assert wsi.info.mpp is None


def test_ngff_zattrs_missing_axes_mpp(tmp_path: Path, remote_sample: Callable) -> None:
    """Test that mpp is None if axes are missing."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample with no axes
    sample_copy = tmp_path / "ngff-1"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    zattrs["multiscales"][0]["axes"] = []
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    wsi = wsireader.NGFFWSIReader(sample_copy)
    assert wsi.info.mpp is None


def test_ngff_empty_datasets_mpp(tmp_path: Path, remote_sample: Callable) -> None:
    """Test that mpp is None if there are no datasets."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample with no axes
    sample_copy = tmp_path / "ngff-1"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    zattrs["multiscales"][0]["datasets"] = []
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    wsi = wsireader.NGFFWSIReader(sample_copy)
    assert wsi.info.mpp is None


def test_ngff_no_scale_transforms_mpp(tmp_path: Path, remote_sample: Callable) -> None:
    """Test that mpp is None if no scale transforms are present."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample with no axes
    sample_copy = tmp_path / "ngff-1.zarr"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    for i, _ in enumerate(zattrs["multiscales"][0]["datasets"]):
        datasets = zattrs["multiscales"][0]["datasets"][i]
        datasets["coordinateTransformations"][0]["type"] = "identity"
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    wsi = wsireader.NGFFWSIReader(sample_copy)
    assert wsi.info.mpp is None


def test_ngff_missing_omero_version(tmp_path: Path, remote_sample: Callable) -> None:
    """Test that the reader can handle missing omero version."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = tmp_path / "ngff-1.zarr"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    # Remove the omero version
    del zattrs["omero"]["version"]
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    wsireader.WSIReader.open(sample_copy)


def test_ngff_missing_multiscales_returns_false(
    tmp_path: Path,
    remote_sample: Callable,
) -> None:
    """Test that missing multiscales key returns False for is_ngff."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = tmp_path / "ngff-1.zarr"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    # Remove the multiscales key
    del zattrs["multiscales"]
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    assert not wsireader.is_ngff(sample_copy)


def test_ngff_wrong_format_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    remote_sample: Callable,
) -> None:
    """Test that is_ngff is False and logs a warning if metadata is wrong."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = tmp_path / "ngff-1.zarr"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    # Change the format to something else
    zattrs["multiscales"] = "foo"
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    with caplog.at_level(logging.WARNING):
        assert not wsireader.is_ngff(sample_copy)
    assert "must be present and of the correct type" in caplog.text


def test_ngff_omero_below_min_version(tmp_path: Path, remote_sample: Callable) -> None:
    """Test for FileNotSupportedError when omero version is below minimum."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = tmp_path / "ngff-1.zarr"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    # Change the format to something else
    zattrs["omero"]["version"] = "0.0"
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    with pytest.raises(FileNotSupportedError):
        wsireader.WSIReader.open(sample_copy)


def test_ngff_omero_above_max_version(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    remote_sample: Callable,
) -> None:
    """Test for FileNotSupportedError when omero version is above maximum."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = tmp_path / "ngff-1.zarr"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    # Change the format to something else
    zattrs["omero"]["version"] = "10.0"
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    # Check that the warning is logged
    with caplog.at_level(logging.WARNING):
        wsireader.WSIReader.open(sample_copy)
    assert "maximum supported version" in caplog.text


def test_ngff_multiscales_below_min_version(
    tmp_path: Path,
    remote_sample: Callable,
) -> None:
    """Test for FileNotSupportedError when multiscales version is below minimum."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = tmp_path / "ngff-1.zarr"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    # Change the format to something else
    zattrs["multiscales"][0]["version"] = "0.0"
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    with pytest.raises(FileNotSupportedError):
        wsireader.WSIReader.open(sample_copy)


def test_ngff_multiscales_above_max_version(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    remote_sample: Callable,
) -> None:
    """Test for FileNotSupportedError when multiscales version is above maximum."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = tmp_path / "ngff-1.zarr"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    # Change the format to something else
    zattrs["multiscales"][0]["version"] = "10.0"
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    # Check that the warning is logged
    with caplog.at_level(logging.WARNING):
        wsireader.WSIReader.open(sample_copy)
    assert "maximum supported version" in caplog.text


def test_ngff_non_numeric_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote_sample: Callable,
) -> None:
    """Test that the reader can handle non-numeric omero versions."""
    # Patch the is_ngff function to change the min/max version
    if_ngff = wsireader.is_ngff  # noqa: F841
    min_version = Version("0.4")
    max_version = Version("0.5")

    def patched_is_ngff(
        path: Path,
        min_version: Version = min_version,
        max_version: Version = max_version,
    ) -> bool:
        """Patched is_ngff function with new min/max version."""
        return is_ngff(path, min_version, max_version)

    monkeypatch.setattr(wsireader, "is_ngff", patched_is_ngff)

    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = tmp_path / "ngff-1.zarr"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    # Set the omero version to a non-numeric string
    zattrs["omero"]["version"] = "0.5-dev"
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    wsireader.WSIReader.open(sample_copy)


def test_ngff_inconsistent_multiscales_versions(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    remote_sample: Callable,
) -> None:
    """Test that the reader logs a warning inconsistent multiscales versions."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = tmp_path / "ngff-1.zarr"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    # Set the versions to be inconsistent
    multiscales = zattrs["multiscales"]
    # Needs at least 2 multiscales to be inconsistent
    if len(multiscales) < 2:
        multiscales.append(copy.deepcopy(multiscales[0]))
    for i, _ in enumerate(multiscales):
        multiscales[i]["version"] = f"0.{i}-dev"
    zattrs["omero"]["multiscales"] = multiscales
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    # Capture logger output to check for warning
    with caplog.at_level(logging.WARNING), pytest.raises(FileNotSupportedError):
        wsireader.WSIReader.open(sample_copy)
    assert "multiple versions" in caplog.text


def test_jp2_no_mpp_appmag(tmp_path: Path) -> None:
    """Test WSIReader init for a JP2 with no resolution metadata."""
    path = tmp_path / "test.jp2"
    glymur.Jp2k(
        path,
        data=np.ones((128, 128, 3), np.uint8),
    )
    _ = WSIReader.open(path).info


def test_jp2_empty_xml(tmp_path: Path) -> None:
    """Test WSIReader init for a JP2 with empty Jp2 XML."""
    path = tmp_path / "test.jp2"
    jp2 = glymur.Jp2k(
        path,
        data=np.ones((128, 128, 3), np.uint8),
    )
    xml_path = tmp_path / "data.xml"
    xml_path.write_text("<Jp2></Jp2>")
    xmlbox = glymur.jp2box.XMLBox(filename=xml_path)
    jp2.append(xmlbox)
    _ = WSIReader.open(path).info


def test_jp2_empty_xml_empty_description(tmp_path: Path) -> None:
    """Test WSIReader init for a JP2 with an empty JP2 XML description tag."""
    path = tmp_path / "test.jp2"
    jp2 = glymur.Jp2k(
        path,
        data=np.ones((128, 128, 3), np.uint8),
    )
    xml_path = tmp_path / "data.xml"
    xml_path.write_text("<Jp2><description></description></Jp2>")
    xmlbox = glymur.jp2box.XMLBox(filename=xml_path)
    jp2.append(xmlbox)
    _ = WSIReader.open(path).info


def test_jp2_empty_xml_description_no_appmag_no_mpp(tmp_path: Path) -> None:
    """Test WSIReader init for a JP2 with not AppMap or MPP in description."""
    path = tmp_path / "test.jp2"
    jp2 = glymur.Jp2k(
        path,
        data=np.ones((128, 128, 3), np.uint8),
    )
    xml_path = tmp_path / "data.xml"
    xml_path.write_text("<Jp2><description>Description here</description></Jp2>")
    xmlbox = glymur.jp2box.XMLBox(filename=xml_path)
    jp2.append(xmlbox)
    _ = WSIReader.open(path).info


def test_jp2_no_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test WSIReader with an image crafted with a missing header.

    Note: This should not be possible for a real file.

    """
    path = tmp_path / "test.jp2"
    jp2 = glymur.Jp2k(
        path,
        data=np.ones((128, 128, 3), np.uint8),
    )
    xml_path = tmp_path / "data.xml"
    xml_path.write_text("<Jp2><description>Description here</description></Jp2>")
    xmlbox = glymur.jp2box.XMLBox(filename=xml_path)
    jp2.append(xmlbox)
    wsi = WSIReader.open(path)
    monkeypatch.setattr(
        wsi.glymur_jp2,
        "box",
        [box for box in wsi.glymur_jp2.box if box.box_id != "jp2h"],
    )
    # Ensure there is no memoized version of info
    monkeypatch.setattr(wsi, "_m_info", None)
    with pytest.raises(ValueError, match="image header missing"):
        _ = wsi.info


# -----------------------------------------------------------------------------
# Parameterized WSIReader Tests
# -----------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=[
        {
            "reader_class": AnnotationStoreReader,
            "sample_key": "annotation_store_svs_1",
            "kwargs": {
                "base_wsi_key": "svs-1-small",
                "renderer": AnnotationRenderer(
                    "type",
                    COLOR_DICT,
                ),
                "alpha": 0.5,
            },
        },
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
        {
            "reader_class": TIFFWSIReader,
            "sample_key": "ome-brightfield-small-pyramid",
            "kwargs": {},
        },
        {
            "reader_class": OpenSlideWSIReader,
            "sample_key": "ventana-tif",
            "kwargs": {},
        },
        {
            "reader_class": DICOMWSIReader,
            "sample_key": "dicom-1",
            "kwargs": {},
        },
        {
            "reader_class": NGFFWSIReader,
            "sample_key": "ngff-1",
            "kwargs": {},
        },
        {
            "reader_class": OpenSlideWSIReader,
            "sample_key": "svs-1-small",
            "kwargs": {},
        },
        {
            "reader_class": JP2WSIReader,
            "sample_key": "jp2-omnyx-small",
            "kwargs": {},
        },
        {
            "reader_class": TIFFWSIReader,
            "sample_key": "qptiff_sample",
            "kwargs": {},
        },
    ],
    ids=[
        "AnnotationReaderOverlaid",
        "AnnotationReaderMaskOnly",
        "TIFFReader",
        "OpenSlideReader (Ventana non-tiled tif)",
        "DICOMReader",
        "NGFFWSIReader",
        "OpenSlideWSIReader (Small SVS)",
        "OmnyxJP2WSIReader",
        "TIFFReader_Multichannel",
    ],
)
def wsi(request: requests.request, remote_sample: Callable) -> WSIReader:
    """WSIReader instance fixture.

    Reader type varies as fixture is parametrized.

    """
    reader_class = request.param.pop("reader_class")
    sample = remote_sample(request.param.pop("sample_key"))

    kwargs = request.param.pop("kwargs")
    new_kwargs = {}

    for key, value in kwargs.items():
        if key.endswith("_key") and isinstance(value, str):
            new_kwargs[key[:-4]] = remote_sample(value)
        else:
            new_kwargs[key] = value

    return reader_class(
        sample,
        **new_kwargs,
    )


def test_base_open(wsi: WSIReader) -> None:
    """Checks that WSIReader.open detects the type correctly."""
    new_wsi = WSIReader.open(wsi.input_path)
    assert type(new_wsi) is type(wsi)


def test_wsimeta_attrs(wsi: WSIReader) -> None:
    """Check for expected attrs in .info / WSIMeta.

    Checks for existence of expected attrs but not their contents.

    """
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


def test_read_rect_level_consistency(wsi: WSIReader) -> None:
    """Compare the same region at each stored resolution level.

    Read the same region at each stored resolution level and compare
    the resulting image using phase cross correlation to check that
    they are aligned.

    """
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
    as_float = [img.astype(np.float64) for img in blurred]

    # Pair-wise check resolutions for mean squared error
    for i, a in enumerate(as_float):
        for b in as_float[i + 1 :]:
            _, error, phase_diff = phase_cross_correlation(a, b, normalization=None)
            assert phase_diff < 0.125
            assert error < 0.125


def test_read_bounds_level_consistency(wsi: WSIReader) -> None:
    """Compare the same region at each stored resolution level.

    Read the same region at each stored resolution level and compare
    the resulting image using phase cross correlation to check that
    they are aligned.

    """
    bounds = (0, 0, 1024, 1024)
    # This logic can be moved from the helper to here when other
    # reader classes have been parameterised into scenarios also.
    read_bounds_level_consistency(wsi, bounds)


def test_fuzz_read_region_baseline_size(wsi: WSIReader) -> None:
    """Fuzz test for `read_bounds` output size at level 0 (baseline).

    - Tests that the output image size matches the input bounds size.
    - 50 random seeded reads are performed.
    - All test bounds are within the slide dimensions.
    - Bounds sizes are randomised between 1 and 512 in width and height.

    """
    rng = np.random.default_rng(123)
    width, height = wsi.info.slide_dimensions
    iterations = 50

    if wsi.input_path.stem == "test1":
        iterations = 5

    for _ in range(iterations):
        size = (rng.integers(1, 512), rng.integers(1, 512))
        location = (
            rng.integers(0, width - size[0]),
            rng.integers(0, height - size[1]),
        )
        bounds = locsize2bounds(location, size)
        region = wsi.read_bounds(bounds, resolution=0, units="level")
        assert region.shape[:2][::-1] == size


def test_read_rect_coord_space_consistency(wsi: WSIReader) -> None:
    """Test that read_rect coord_space modes are consistent.

    Using `read_rect` with `coord_space="baseline"` and
    `coord_space="resolution"` should produce the same output when
    the bounds are a multiple of the scale difference between the two
    modes. I.E. reading at baseline with a set of coordinates should
    yield the same region as reading at half the resolution and
    with coordinates which are half the size. Note that the output
    will not be of the same size, but the field of view will match.

    """
    roi1 = wsi.read_rect(
        np.array([500, 500]),
        np.array([2000, 2000]),
        coord_space="baseline",
        resolution=1.00,
        units="baseline",
    )
    roi2 = wsi.read_rect(
        np.array([250, 250]),
        np.array([1000, 1000]),
        coord_space="resolution",
        resolution=0.5,
        units="baseline",
    )
    # Make the regions the same size for comparison of content
    roi2 = imresize(roi2, output_size=(2000, 2000))

    # Check MSE
    mse = np.mean((roi1 - roi2) ** 2)
    assert mse < 100

    # Check SSIM
    ssim = structural_similarity(
        roi1,
        roi2,
        channel_axis=-1,
        winsize=min(*roi1.shape[:2], 7),
    )
    assert ssim > 0.8


def test_file_path_does_not_exist() -> None:
    """Test that FileNotFoundError is raised when file does not exist."""
    for reader_class in [
        AnnotationStoreReader,
        TIFFWSIReader,
        DICOMWSIReader,
        NGFFWSIReader,
        OpenSlideWSIReader,
        JP2WSIReader,
    ]:
        with pytest.raises(FileNotFoundError):
            _ = reader_class("./foo.bar")


def test_read_mpp(wsi: WSIReader) -> None:
    """Test that the mpp is read correctly."""
    assert wsi.info.mpp == pytest.approx(0.25, 1)


def test_read_multi_channel(source_image: Path) -> None:
    """Test reading image with more than three channels.

    Create a virtual WSI by concatenating the source_image.

    """
    img_array = utils.misc.imread(Path(source_image))
    new_img_array = np.concatenate((img_array, img_array), axis=-1)

    new_img_size = new_img_array.shape[:2][::-1]
    meta = wsireader.WSIMeta(slide_dimensions=new_img_size, axes="YXS", mpp=(0.5, 0.5))
    wsi = wsireader.VirtualWSIReader(new_img_array, info=meta)

    region = wsi.read_rect(
        location=(0, 0),
        size=(50, 100),
        pad_mode="reflect",
        units="mpp",
        resolution=0.25,
    )
    target = cv2.resize(
        new_img_array[:50, :25, :],
        (50, 100),
        interpolation=cv2.INTER_CUBIC,
    )

    assert region.shape == (100, 50, (new_img_array.shape[-1]))
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2


def test_visualise_multi_channel(sample_qptiff: Path) -> None:
    """Test visualising a multi-channel qptiff multiplex image."""
    wsi = wsireader.TIFFWSIReader(sample_qptiff, post_proc="auto")
    wsi2 = wsireader.TIFFWSIReader(sample_qptiff, post_proc=None)

    region = wsi.read_rect(location=(0, 0), size=(50, 100))
    region2 = wsi2.read_rect(location=(0, 0), size=(50, 100))

    assert region.shape == (100, 50, 3)
    assert region2.shape == (100, 50, 7)
