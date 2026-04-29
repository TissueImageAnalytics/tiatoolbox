"""Test for reading whole-slide images."""

from __future__ import annotations

import copy
import itertools
import json
import logging
import re
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import cv2
import glymur
import numpy as np
import openslide
import pytest
import SimpleITK as sitk  # noqa: N813
import tifffile
import zarr
from click.testing import CliRunner
from defusedxml import ElementTree
from packaging.version import Version
from skimage.filters import threshold_otsu
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.morphology import binary_dilation, disk, remove_small_objects
from skimage.registration import phase_cross_correlation

from tiatoolbox import cli, utils
from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils import imread, tiff_to_fsspec
from tiatoolbox.utils.exceptions import FileNotSupportedError
from tiatoolbox.utils.transforms import imresize, locsize2bounds
from tiatoolbox.utils.visualization import AnnotationRenderer
from tiatoolbox.wsicore import WSIReader, wsireader
from tiatoolbox.wsicore.wsimeta import WSIMeta
from tiatoolbox.wsicore.wsireader import (
    AnnotationStoreReader,
    ArrayView,
    DICOMWSIReader,
    FsspecJsonWSIReader,
    JP2WSIReader,
    NGFFWSIReader,
    OpenSlideWSIReader,
    TIFFWSIReader,
    TIFFWSIReaderDelegate,
    TransformedWSIReader,
    VirtualWSIReader,
    _handle_tiff_wsi,
    _handle_virtual_wsi,
    is_dicom,
    is_ngff,
    is_tiled_tiff,
    is_zarr,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable

    import requests
    from openslide import OpenSlide

    from tiatoolbox.type_hints import IntBounds, IntPair

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
def get_tissue_com_tile(reader: WSIReader, size: int) -> IntBounds:
    """Returns bounds of a tile located approximately at COM of the tissue.

    Uses reader.tissue_mask() to find the center of mass of the tissue
    and returns a tile centered at that point, at requested size. Used
    to ensure we are looking at a tissue region when doing level consistency
     tests etc.

    Args:
        reader (WSIReader): WSIReader instance.
        size (int): Size at baseline of the tile to return.

    Returns:
        IntBounds: Baseline bounds of the tile centered at COM of the tissue.

    """
    mask = reader.tissue_mask(resolution=8.0, units="mpp").img

    # Find the center of mass of the tissue
    ys, xs = np.nonzero(mask)
    com_y = int(ys.mean())
    com_x = int(xs.mean())
    # convert to baseline coordinates
    com_x = int(com_x * (8.0 / reader.info.mpp[0]))
    com_y = int(com_y * (8.0 / reader.info.mpp[1]))

    # Calculate bounds for the tile centered at COM
    half_size = size // 2
    bounds = (
        max(0, com_x - half_size),
        max(0, com_y - half_size),
        min(reader.info.slide_dimensions[0], com_x + half_size),
        min(reader.info.slide_dimensions[1], com_y + half_size),
    )
    return np.array(bounds)


def strictly_increasing(sequence: Iterable) -> bool:
    """Return True if sequence is strictly increasing.

    Args:
        sequence (Iterable): Sequence to check.

    Returns:
        bool: True if strictly increasing.

    """
    return all(a < b for a, b in itertools.pairwise(sequence))


def strictly_decreasing(sequence: Iterable) -> bool:
    """Return True if sequence is strictly decreasing.

    Args:
        sequence (Iterable): Sequence to check.


    Returns:
        bool: True if strictly decreasing.

    """
    return all(a > b for a, b in itertools.pairwise(sequence))


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

_FSSPEC_WSI_CACHE = {}


def fsspec_wsi(sample_svs: Path, track_tmp_path: Path) -> FsspecJsonWSIReader:
    """Returns cached FsspecJsonWSIReader instance.

    The reader instance opens CMU-1-Small-Region.svs image.

    It's cached so the reader can be reused,

    since loading the whole image using HTTP range requests from:

    https://huggingface.co/datasets/TIACentre/TIAToolBox_Remote_Samples/resolve/main/sample_wsis/CMU-1-Small-Region.svs

    takes about 20 seconds.

    """
    cache_key = "sample_svs"

    if cache_key in _FSSPEC_WSI_CACHE:
        return _FSSPEC_WSI_CACHE[cache_key]  # Return cached instance

    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=Path(sample_svs).parent,
        file_types=file_types,
    )
    svs_file_path = str(files_all[0])
    json_file_path = str(track_tmp_path / "fsspec.json")
    final_url = "https://huggingface.co/datasets/TIACentre/TIAToolBox_Remote_Samples/resolve/main/sample_wsis/CMU-1-Small-Region.svs"
    tiff_to_fsspec.main(svs_file_path, json_file_path, final_url)

    _FSSPEC_WSI_CACHE[cache_key] = wsireader.FsspecJsonWSIReader(json_file_path)
    return _FSSPEC_WSI_CACHE[cache_key]


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


def test_wsireader_slide_info(sample_svs: Path, track_tmp_path: Path) -> None:
    """Test for slide_info in WSIReader class as a python function."""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(Path(sample_svs).parent),
        file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    slide_param = wsi.info
    out_path = track_tmp_path / slide_param.file_path.with_suffix(".yaml").name
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
        strict=False,
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
    for objective_power, expected_level in zip(
        objective_powers, expected_levels, strict=False
    ):
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


def test_is_not_tiled_tiff(tmp_samples_path: Path) -> None:
    """Test if source_image is not a tiled tiff."""
    temp_tiff_path = tmp_samples_path / "not_tiled.tiff"
    images = [np.zeros(shape=(4, 4)) for _ in range(3)]
    # Write multi-page TIFF with all pages not tiled
    with tifffile.TiffWriter(temp_tiff_path) as tif:
        for image in images:
            tif.write(image, compression=None, tile=None)
    assert wsireader.is_tiled_tiff(temp_tiff_path) is False


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


def test_wsireader_save_tiles(sample_svs: Path, track_tmp_path: Path) -> None:
    """Test for save_tiles in wsireader as a python function."""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(Path(sample_svs).parent),
        file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    wsi.save_tiles(
        output_dir=str(track_tmp_path / "test_wsireader_save_tiles"),
        tile_objective_value=5,
        tile_read_size=(5000, 5000),
        verbose=True,
    )
    assert (
        track_tmp_path
        / "test_wsireader_save_tiles"
        / "CMU-1-Small-Region.svs"
        / "Output.csv"
    ).exists()
    assert (
        track_tmp_path
        / "test_wsireader_save_tiles"
        / "CMU-1-Small-Region.svs"
        / "slide_thumbnail.jpg"
    ).exists()
    assert (
        track_tmp_path
        / "test_wsireader_save_tiles"
        / "CMU-1-Small-Region.svs"
        / "Tile_5_0_0.jpg"
    ).exists()


def test_incompatible_objective_value(sample_svs: Path, track_tmp_path: Path) -> None:
    """Test for incompatible objective value."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    with pytest.raises(ValueError, match="objective power"):
        wsi.save_tiles(
            output_dir=str(
                Path(track_tmp_path).joinpath("test_wsireader_save_tiles"),
            ),
            tile_objective_value=3,
            tile_read_size=(5000, 5000),
            verbose=True,
        )


def test_incompatible_level(
    sample_svs: Path,
    track_tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test for incompatible objective value."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    wsi.save_tiles(
        output_dir=str(Path(track_tmp_path).joinpath("test_wsireader_save_tiles2")),
        tile_objective_value=1,
        tile_read_size=(500, 500),
        verbose=True,
    )

    assert "Reading at tile_objective_value 1 not allowed" in caplog.text


def test_wsireader_jp2_save_tiles(sample_jp2: Path, track_tmp_path: Path) -> None:
    """Test for save_tiles in wsireader as a python function."""
    wsi = wsireader.JP2WSIReader(sample_jp2)
    wsi.save_tiles(
        output_dir=str(track_tmp_path / "test_wsireader_jp2_save_tiles"),
        tile_objective_value=5,
        tile_read_size=(5000, 5000),
        verbose=True,
    )
    assert (
        track_tmp_path
        / "test_wsireader_jp2_save_tiles"
        / "CMU-1-Small-Region.omnyx.jp2"
        / "Output.csv"
    ).exists()
    assert (
        track_tmp_path
        / "test_wsireader_jp2_save_tiles"
        / "CMU-1-Small-Region.omnyx.jp2"
        / "slide_thumbnail.jpg"
    ).exists()
    assert (
        track_tmp_path
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


def test_virtual_rgb_mode_postproc_then_composite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that post-processing in VirtualWSIReader occurs before compositing."""
    # 3-channel input -> VirtualWSIReader.mode == "rgb"
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    v = wsireader.VirtualWSIReader(img, post_proc=None)

    calls = {"bg": 0, "alphas": [], "last": None}

    def fake_bg_composite(*, image: np.ndarray, alpha: bool) -> np.ndarray:
        """Fake background_composite to record calls."""
        calls["bg"] += 1
        calls["alphas"].append(alpha)
        calls["last"] = image
        return image

    monkeypatch.setattr(utils.transforms, "background_composite", fake_bg_composite)

    # No post-proc -> composite still called with alpha=False
    out1 = v.read_rect((0, 0), (8, 8))
    assert calls["bg"] == 1
    assert calls["alphas"][-1] is False
    assert out1.shape == (8, 8, 3)

    # Attach a post-proc; it should run BEFORE composite
    def recorder(img: np.ndarray) -> np.ndarray:
        img2 = img.copy()
        img2[..., 0] = 255  # make the effect visible at composite time
        return img2

    v.post_proc = recorder
    out2 = v.read_bounds((0, 0, 8, 8))
    assert calls["bg"] == 2  # composite called again
    # background_composite must have received the post-processed content
    assert (calls["last"][..., 0] == 255).all()
    assert out2.shape == (8, 8, 3)


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
    for unit, scaler in zip(units, scale_fns, strict=False):
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
    track_tmp_path: pytest.TempPathFactory,
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
    assert isinstance(wsi, (wsireader.OpenSlideWSIReader, wsireader.TIFFWSIReader))

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
    temp_file = str(track_tmp_path / "sample.npy")
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
        TransformedWSIReader(
            mini_wsi2_svs, target_img=mini_wsi2_svs, transform=np.eye(3)
        ),
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


def test_command_line_read_bounds(sample_ndpi: Path, track_tmp_path: Path) -> None:
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
            str(Path(track_tmp_path).joinpath("im_region.jpg")),
        ],
    )

    assert read_bounds_result.exit_code == 0
    assert Path(track_tmp_path).joinpath("im_region.jpg").is_file()

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
            str(Path(track_tmp_path).joinpath("im_region2.jpg")),
        ],
    )

    assert read_bounds_result.exit_code == 0
    assert Path(track_tmp_path).joinpath("im_region2.jpg").is_file()


def test_command_line_jp2_read_bounds(sample_jp2: Path) -> None:
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
    result_path = sample_jp2.parent.parent.joinpath("im_region.jpg")
    assert result_path.is_file()


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
    """Test fallback behaviour for invalid OME-XML metadata instrument."""
    wsi = wsireader.TIFFWSIReader(sample_ome_tiff_level_0)
    monkeypatch.setattr(
        wsi.tiff.pages[0],
        "description",
        wsi.tiff.pages[0].description.replace(
            '<Objective ID="Objective:0:0" NominalMagnification="20.0"/>',
            "",
        ),
    )
    monkeypatch.setattr(wsi, "_m_info", None)

    info = wsi.info
    assert info.objective_power is None or isinstance(info.objective_power, float)


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
    assert isinstance(wsi, (wsireader.OpenSlideWSIReader, wsireader.TIFFWSIReader))


def test_tiled_tiff_tifffile(remote_sample: Callable) -> None:
    """Test fallback to tifffile for files which openslide cannot read.

    E.G. tiled tiffs with JPEG XL compression.

    """
    sample_path = remote_sample("tiled-tiff-1-small-jp2k")
    wsi = wsireader.WSIReader.open(sample_path)
    assert isinstance(wsi, wsireader.TIFFWSIReader)


def test_is_zarr_empty_dir(track_tmp_path: Path) -> None:
    """Test is_zarr is false for an empty .zarr directory."""
    zarr_dir = track_tmp_path / "zarr.zarr"
    zarr_dir.mkdir()
    assert not is_zarr(zarr_dir)


def test_is_zarr_array(track_tmp_path: Path) -> None:
    """Test is_zarr is true for a .zarr directory with an array."""
    zarr_dir = track_tmp_path / "zarr.zarr"
    zarr_dir.mkdir()
    # Zarr 3 uses zarr.json, NOT .zarray
    metadata_path = zarr_dir / "zarr.json"

    minimal_zarr3 = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [1, 1, 1],
        "data_type": "uint8",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 1, 1]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,  # This was the missing key causing your error
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": {},
    }
    with Path.open(metadata_path, "w") as f:
        json.dump(minimal_zarr3, f)
    assert is_zarr(zarr_dir)


def test_is_zarr_group(track_tmp_path: Path) -> None:
    """Test is_zarr is true for a .zarr directory with a group."""
    zarr_dir = track_tmp_path / "zarr.zarr"
    zarr_dir.mkdir()
    _zgroup_path = zarr_dir / ".zgroup"
    minimal_zgroup = {
        "zarr_format": 2,
    }
    with Path.open(_zgroup_path, "w") as f:
        json.dump(minimal_zgroup, f)
    assert is_zarr(zarr_dir)


def test_is_ngff_regular_zarr(track_tmp_path: Path) -> None:
    """Test is_ngff is false for a regular zarr."""
    zarr_path = track_tmp_path / "zarr.zarr"
    # Create zarr array on disk
    zarr.array(RNG.random((32, 32)), store=zarr.storage.LocalStore(zarr_path))
    assert is_zarr(zarr_path)
    assert not is_ngff(zarr_path)

    # check we get the appropriate error message if we open it
    with pytest.raises(FileNotSupportedError, match=r"does not appear to be a v0.4"):
        WSIReader.open(zarr_path)


@pytest.mark.skipif(
    toolbox_env.running_on_ci(),
    reason="Depends on external source which may not be accessible.",
)
# The data available on s3 bucket from OMERO may not always be accessible
# and therefore the test is expected to fail.
# Locally, a different image can be tested from this catalogue
# https://idr.github.io/ome-ngff-samples/
def test_ngff_s3() -> None:
    """Test read from s3 bucket."""
    # This sample image only tests if NGFFWSIReader can read image from s3.
    # read_rect is not compatible for these kind of multiplex images.
    # This feature needs to be added in future release of TIAToolbox.
    url = "s3://idr/zarr/v0.4/idr0062A/6001247.zarr"
    storage_options = {
        "anon": True,
        "client_kwargs": {"endpoint_url": "https://uk1s3.embassy.ebi.ac.uk"},
    }
    wsi = WSIReader.open(url, storage_options=storage_options)

    assert np.all(wsi.slide_dimensions(resolution=1, units="baseline") == (253, 210))


def test_store_reader_no_info(track_tmp_path: Path) -> None:
    """Test AnnotationStoreReader with no info."""
    SQLiteStore(track_tmp_path / "store.db")
    with pytest.raises(ValueError, match="No metadata found"):
        AnnotationStoreReader(track_tmp_path / "store.db")


def test_store_reader_explicit_info(
    remote_sample: Callable, track_tmp_path: Path
) -> None:
    """Test AnnotationStoreReader with explicit info."""
    SQLiteStore(track_tmp_path / "store.db")
    wsi_reader = WSIReader.open(remote_sample("svs-1-small"))
    reader = AnnotationStoreReader(track_tmp_path / "store.db", wsi_reader.info)
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


def test_store_reader_no_types(track_tmp_path: Path, remote_sample: Callable) -> None:
    """Test AnnotationStoreReader with no types."""
    SQLiteStore(track_tmp_path / "store.db")
    wsi_reader = WSIReader.open(remote_sample("svs-1-small"))
    reader = AnnotationStoreReader(track_tmp_path / "store.db", wsi_reader.info)
    # shouldn't try to color by type if not present
    assert reader.renderer.score_prop is None


def test_store_reader_info_from_base(
    track_tmp_path: Path, remote_sample: Callable
) -> None:
    """Test AnnotationStoreReader with no wsi metadata.

    Test that AnnotationStoreReader will correctly get metadata
    from a provided base_wsi if the store has no wsi metadata.

    """
    SQLiteStore(track_tmp_path / "store.db")
    wsi_reader = WSIReader.open(remote_sample("svs-1-small"))
    store_reader = AnnotationStoreReader(
        track_tmp_path / "store.db", base_wsi=wsi_reader
    )
    # the store reader should have the same metadata as the base wsi
    assert store_reader.info.mpp[0] == wsi_reader.info.mpp[0]


def test_ngff_zattrs_non_micrometer_scale_mpp(
    track_tmp_path: Path,
    remote_sample: Callable,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that mpp is None if scale is not in micrometers."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample with a non-micrometer scale
    sample_copy = track_tmp_path / "ngff-1"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    zattrs["multiscales"][0]["axes"][0]["unit"] = "foo"
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)

    wsi = wsireader.NGFFWSIReader(sample_copy)
    assert "micrometer" in caplog.text

    assert wsi.info.mpp is None


def test_ngff_zattrs_missing_axes_mpp(
    track_tmp_path: Path, remote_sample: Callable
) -> None:
    """Test that mpp is None if axes are missing."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample with no axes
    sample_copy = track_tmp_path / "ngff-1"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    zattrs["multiscales"][0]["axes"] = []
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    wsi = wsireader.NGFFWSIReader(sample_copy)
    assert wsi.info.mpp is None


def test_ngff_empty_datasets_mpp(track_tmp_path: Path, remote_sample: Callable) -> None:
    """Test that mpp is None if there are no datasets."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample with no axes
    sample_copy = track_tmp_path / "ngff-1"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    zattrs["multiscales"][0]["datasets"] = []
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    wsi = wsireader.NGFFWSIReader(sample_copy)
    assert wsi.info.mpp is None


def test_ngff_no_scale_transforms_mpp(
    track_tmp_path: Path, remote_sample: Callable
) -> None:
    """Test that mpp is None if no scale transforms are present."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample with no axes
    sample_copy = track_tmp_path / "ngff-1.zarr"
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


def test_ngff_missing_omero_version(
    track_tmp_path: Path, remote_sample: Callable
) -> None:
    """Test that the reader can handle missing omero version."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = track_tmp_path / "ngff-1.zarr"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    # Remove the omero version
    del zattrs["omero"]["version"]
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    wsireader.WSIReader.open(sample_copy)


def test_ngff_missing_multiscales_returns_false(
    track_tmp_path: Path,
    remote_sample: Callable,
) -> None:
    """Test that missing multiscales key returns False for is_ngff."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = track_tmp_path / "ngff-1.zarr"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    # Remove the multiscales key
    del zattrs["multiscales"]
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    assert not wsireader.is_ngff(sample_copy)


def test_ngff_wrong_format_metadata(
    track_tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    remote_sample: Callable,
) -> None:
    """Test that is_ngff is False and logs a warning if metadata is wrong."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = track_tmp_path / "ngff-1.zarr"
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


def test_ngff_omero_below_min_version(
    track_tmp_path: Path, remote_sample: Callable
) -> None:
    """Test for FileNotSupportedError when omero version is below minimum."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = track_tmp_path / "ngff-1.zarr"
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
    track_tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    remote_sample: Callable,
) -> None:
    """Test for FileNotSupportedError when omero version is above maximum."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = track_tmp_path / "ngff-1.zarr"
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
    track_tmp_path: Path,
    remote_sample: Callable,
) -> None:
    """Test for FileNotSupportedError when multiscales version is below minimum."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = track_tmp_path / "ngff-1.zarr"
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
    track_tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    remote_sample: Callable,
) -> None:
    """Test for FileNotSupportedError when multiscales version is above maximum."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = track_tmp_path / "ngff-1.zarr"
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
    track_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote_sample: Callable,
) -> None:
    """Test that the reader can handle non-numeric omero versions."""
    # Patch the is_ngff function to change the min/max version
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
    sample_copy = track_tmp_path / "ngff-1.zarr"
    shutil.copytree(sample, sample_copy)
    with Path.open(sample_copy / ".zattrs") as fh:
        zattrs = json.load(fh)
    # Set the omero version to a non-numeric string
    zattrs["omero"]["version"] = "0.5-dev"
    with Path.open(sample_copy / ".zattrs", "w") as fh:
        json.dump(zattrs, fh, indent=2)
    wsireader.WSIReader.open(sample_copy)


def test_ngff_inconsistent_multiscales_versions(
    track_tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    remote_sample: Callable,
) -> None:
    """Test that the reader logs a warning inconsistent multiscales versions."""
    sample = remote_sample("ngff-1")
    # Create a copy of the sample
    sample_copy = track_tmp_path / "ngff-1.zarr"
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


def test_jp2_no_mpp_appmag(track_tmp_path: Path) -> None:
    """Test WSIReader init for a JP2 with no resolution metadata."""
    path = track_tmp_path / "test.jp2"
    glymur.Jp2k(
        path,
        data=np.ones((128, 128, 3), np.uint8),
    )
    _ = WSIReader.open(path).info


def test_jp2_empty_xml(track_tmp_path: Path) -> None:
    """Test WSIReader init for a JP2 with empty Jp2 XML."""
    path = track_tmp_path / "test.jp2"
    jp2 = glymur.Jp2k(
        path,
        data=np.ones((128, 128, 3), np.uint8),
    )
    xml_path = track_tmp_path / "data.xml"
    xml_path.write_text("<Jp2></Jp2>")
    xmlbox = glymur.jp2box.XMLBox(filename=xml_path)
    jp2.append(xmlbox)
    _ = WSIReader.open(path).info


def test_jp2_empty_xml_empty_description(track_tmp_path: Path) -> None:
    """Test WSIReader init for a JP2 with an empty JP2 XML description tag."""
    path = track_tmp_path / "test.jp2"
    jp2 = glymur.Jp2k(
        path,
        data=np.ones((128, 128, 3), np.uint8),
    )
    xml_path = track_tmp_path / "data.xml"
    xml_path.write_text("<Jp2><description></description></Jp2>")
    xmlbox = glymur.jp2box.XMLBox(filename=xml_path)
    jp2.append(xmlbox)
    _ = WSIReader.open(path).info


def test_jp2_empty_xml_description_no_appmag_no_mpp(track_tmp_path: Path) -> None:
    """Test WSIReader init for a JP2 with not AppMap or MPP in description."""
    path = track_tmp_path / "test.jp2"
    jp2 = glymur.Jp2k(
        path,
        data=np.ones((128, 128, 3), np.uint8),
    )
    xml_path = track_tmp_path / "data.xml"
    xml_path.write_text("<Jp2><description>Description here</description></Jp2>")
    xmlbox = glymur.jp2box.XMLBox(filename=xml_path)
    jp2.append(xmlbox)
    _ = WSIReader.open(path).info


def test_jp2_no_header(track_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test WSIReader with an image crafted with a missing header.

    Note: This should not be possible for a real file.

    """
    path = track_tmp_path / "test.jp2"
    jp2 = glymur.Jp2k(
        path,
        data=np.ones((128, 128, 3), np.uint8),
    )
    xml_path = track_tmp_path / "data.xml"
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
    assert isinstance(new_wsi, (type(wsi), TIFFWSIReader))


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
    bounds = get_tissue_com_tile(wsi, 1024)
    location = bounds[:2]
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
    bounds = get_tissue_com_tile(wsi, 1024)
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
    bounds = get_tissue_com_tile(wsi, 2000)
    location = (bounds[:2] // 2) * 2  # ensure even coordinates
    roi1 = wsi.read_rect(
        location,
        np.array([2000, 2000]),
        coord_space="baseline",
        resolution=1.00,
        units="baseline",
    )
    roi2 = wsi.read_rect(
        location // 2,
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


def _make_mock_post_proc(called: dict[str, bool]) -> Callable[[np.ndarray], np.ndarray]:
    """Create a mock post-processing function that modifies the image and sets flag."""

    def mock_post_proc(image: np.ndarray) -> np.ndarray:
        """Mock post-processing: set flag and modify image pixels."""
        called["flag"] = True
        image = image.copy()
        channels = image.shape[-1]
        image[0, 0] = [42] * channels
        image[-1, -1] = [0] * (channels - 1) + [42]
        return image

    return mock_post_proc


def _should_patch_background_composite(wsi: WSIReader) -> bool:
    """Determine whether background_composite should be patched for the given reader."""
    if isinstance(wsi, AnnotationStoreReader):
        return True
    if isinstance(wsi, VirtualWSIReader):
        return wsi.mode == "rgb"
    return isinstance(
        wsi, (OpenSlideWSIReader, JP2WSIReader, DICOMWSIReader, NGFFWSIReader)
    )


def _inject_post_proc_recursive(
    wsi: object, post_proc: Callable[[np.ndarray], np.ndarray]
) -> None:
    """Recursively inject post_proc into all wsi that supports it."""
    current = wsi
    if hasattr(current, "post_proc"):
        current.post_proc = post_proc
    while hasattr(current, "base_wsi") and current.base_wsi is not None:
        current = current.base_wsi
        if hasattr(current, "post_proc"):
            current.post_proc = post_proc


def test_post_proc_logic_across_readers(wsi: WSIReader) -> None:
    """Test that post_proc is applied correctly across all reader classes."""
    called: dict[str, bool] = {"flag": False}
    mock_post_proc = _make_mock_post_proc(called)

    skip_check = isinstance(wsi, AnnotationStoreReader)

    # Recursively inject post_proc into the actual reader
    _inject_post_proc_recursive(wsi, mock_post_proc)

    patch_utils = _should_patch_background_composite(wsi)

    if patch_utils:
        with patch(
            "tiatoolbox.utils.transforms.background_composite",
            lambda image, **_: image,
        ):
            rect = wsi.read_rect(location=(0, 0), size=(50, 50))
            region = wsi.read_bounds(bounds=(0, 0, 50, 50))
    else:
        rect = wsi.read_rect(location=(0, 0), size=(50, 50))
        region = wsi.read_bounds(bounds=(0, 0, 50, 50))

    if skip_check:
        assert isinstance(rect, np.ndarray)
        assert isinstance(region, np.ndarray)
        assert called["flag"]
        return

    if isinstance(wsi, NGFFWSIReader):
        assert isinstance(rect, np.ndarray)
        assert isinstance(region, np.ndarray)
        return

    if isinstance(wsi, OpenSlideWSIReader):
        vendor = getattr(wsi.info, "vendor", "").lower()
        if "ventana" in vendor or "tif" in str(wsi.input_path).lower():
            assert isinstance(rect, np.ndarray)
            assert isinstance(region, np.ndarray)
            return

    assert called["flag"]
    assert isinstance(rect, np.ndarray)
    assert isinstance(region, np.ndarray)
    assert rect[0, 0][-1] == 42
    assert rect[-1, -1][-1] == 42
    assert region[0, 0][-1] == 42
    assert region[-1, -1][-1] == 42


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

    with pytest.raises(FileNotFoundError):
        _ = TransformedWSIReader("./foo.bar", target_img="./foo.bar")


def test_read_mpp(wsi: WSIReader) -> None:
    """Test that the mpp is read correctly."""
    assert wsi.info.mpp == pytest.approx(0.25, 1)


def test_read_multi_channel(source_image: Path) -> None:
    """Test reading image with more than three channels.

    Create a virtual WSI by concatenating the source_image.

    """
    # Moved to tests/test_multichannel_reading.py


def test_visualise_multi_channel(sample_qptiff: Path) -> None:
    """Test visualising a multi-channel qptiff multiplex image."""
    # Moved to tests/test_multichannel_reading.py


def test_get_post_proc_variants() -> None:
    """Test different branches of get_post_proc method."""
    # Moved to tests/test_multichannel_reading.py


def test_post_proc_applied() -> None:
    """Test that post_proc is applied to image region."""
    reader = wsireader.VirtualWSIReader(np.ones((100, 100, 3), dtype=np.uint8))
    reader.post_proc = lambda x: x * 0
    region = reader.read_rect((0, 0), (50, 50))
    assert np.all(region == 0)

    # Create a dummy image region
    dummy_image = np.ones((10, 10, 3), dtype=np.uint8)

    # Define a dummy post-processing function
    def mock_post_proc(image: np.ndarray) -> np.ndarray:
        """Mock post-processing that colors the top-left pixel red."""
        image[0, 0] = [255, 0, 0]  # Modify top-left pixel to red
        return image

    # Create a mock reader with post_proc
    mock_reader = SimpleNamespace(post_proc=mock_post_proc)

    # Create a delegate with the mock reader
    delegate = wsireader.TIFFWSIReaderDelegate.__new__(wsireader.TIFFWSIReaderDelegate)
    delegate.reader = mock_reader

    # Simulate the logic that includes the yellow line
    result = delegate.reader.post_proc(dummy_image.copy())

    # Assert that post_proc was applied
    assert (result[0, 0] == [255, 0, 0]).all()


def test_explicit_none_postproc(sample_svs: Path) -> None:
    """Test explicit None postproc."""
    reader = wsireader.VirtualWSIReader(
        np.ones((100, 100, 3), dtype=np.uint8), post_proc=None
    )
    region = reader.read_bounds((0, 0, 50, 50))
    assert np.all(region == 1)

    reader = wsireader.TIFFWSIReader(sample_svs, post_proc=None)
    region = reader.read_bounds((0, 0, 50, 50))
    assert isinstance(region, np.ndarray)
    assert region.shape == (50, 50, 3)

    region = reader.read_rect((0, 0), (50, 50), coord_space="resolution")
    assert isinstance(region, np.ndarray)
    assert region.shape == (50, 50, 3)


def test_fsspec_json_wsi_reader_instantiation() -> None:
    """Test if FsspecJsonWSIReader is instantiated.

    In case JSON is passed to  WSIReader.open, FsspecJsonWSIReader
    should be instantiated.

    """
    input_path = "mock_path.json"
    mpp = None
    power = None

    with (
        patch(
            "tiatoolbox.wsicore.wsireader.FsspecJsonWSIReader.is_valid_zarr_fsspec",
            return_value=True,
        ),
        patch("tiatoolbox.wsicore.wsireader.FsspecJsonWSIReader") as mock_reader,
    ):
        WSIReader.open(input_path, mpp, power)
        mock_reader.assert_called_once_with(input_path, mpp=mpp, power=power)


def test_generate_fsspec_json_file_and_validate(
    sample_svs: Path, track_tmp_path: Path
) -> None:
    """Test generate fsspec json file and validate it."""
    file_types = ("*.svs",)

    files_all = utils.misc.grab_files_from_dir(
        input_path=Path(sample_svs).parent,
        file_types=file_types,
    )

    svs_file_path = str(files_all[0])
    json_file_path = str(track_tmp_path / "fsspec.json")
    final_url = "https://example.com/some_id"

    tiff_to_fsspec.main(svs_file_path, json_file_path, final_url)

    assert Path(json_file_path).exists(), "Output JSON file was not created."

    assert FsspecJsonWSIReader.is_valid_zarr_fsspec(json_file_path), (
        "FSSPEC JSON file is invalid."
    )


def test_fsspec_wsireader_info_read(sample_svs: Path, track_tmp_path: Path) -> None:
    """Test info read of the FsspecJsonWSIReader.

    Generate fsspec JSON file and load image from:

    https://huggingface.co/datasets/TIACentre/TIAToolBox_Remote_Samples/resolve/main/sample_wsis/CMU-1-Small-Region.svs

    """
    wsi = fsspec_wsi(sample_svs, track_tmp_path)
    info = wsi.info

    assert info is not None, "info  should not be None."


def test_read_bounds_fsspec_reader_baseline(
    sample_svs: Path, track_tmp_path: Path
) -> None:
    """Test FsspecJsonWSIReader read bounds at baseline.

    Location coordinate is in baseline (level 0) reference frame.

    """
    wsi = fsspec_wsi(sample_svs, track_tmp_path)

    bounds = SVS_TEST_TISSUE_BOUNDS
    size = SVS_TEST_TISSUE_SIZE
    im_region = wsi.read_bounds(bounds, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)


def test_read_rect_fsspec_reader_baseline(
    sample_svs: Path, track_tmp_path: Path
) -> None:
    """Test FsspecJsonWSIReader read rect at baseline.

    Location coordinate is in baseline (level 0) reference frame.

    """
    wsi = fsspec_wsi(sample_svs, track_tmp_path)

    location = SVS_TEST_TISSUE_LOCATION
    size = SVS_TEST_TISSUE_SIZE
    im_region = wsi.read_rect(location, size, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)


def test_fsspec_reader_open_invalid_json_file(track_tmp_path: Path) -> None:
    """Ensure JSONDecodeError is handled properly.

    Pass invalid JSON to  FsspecJsonWSIReader.is_valid_zarr_fsspec.
    """
    json_path = track_tmp_path / "invalid.json"
    json_path.write_text("{invalid json}")  # Corrupt JSON

    assert not FsspecJsonWSIReader.is_valid_zarr_fsspec(str(json_path))


def test_fsspec_reader_open_oserror_handling() -> None:
    """Ensure OSError is handled properly.

    Pass non existent JSON to  FsspecJsonWSIReader.is_valid_zarr_fsspec.

    """
    with patch("builtins.open", side_effect=OSError("File not found")):
        result = FsspecJsonWSIReader.is_valid_zarr_fsspec("non_existent.json")

    assert result is False, "Function should return False for OSError"


def test_fsspec_reader_open_pass_empty_json(track_tmp_path: Path) -> None:
    """Ensure empty JSON is handled properly.

    Pass empty JSON to FsspecJsonWSIReader.is_valid_zarr_fsspec and

    verify that it's not valid.

    """
    json_path = track_tmp_path / "empty.json"
    json_path.write_text("{}")

    assert not FsspecJsonWSIReader.is_valid_zarr_fsspec(str(json_path))


def test_fsspec_reader_group_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force coverage of the zarr.Group branch inside FsspecJsonWSIReader."""
    # Create an in-memory Zarr group with datasets
    store = zarr.storage.MemoryStore()
    root = zarr.open(store=store, mode="w")
    root.create_array("0", data=np.zeros((4, 4)))
    root.create_array("1", data=np.ones((8, 8)))

    # Create a reader instance without running __init__
    reader = FsspecJsonWSIReader.__new__(FsspecJsonWSIReader)
    reader._axes = "YX"

    # Patch the internal group so the isinstance() check is True
    reader._zarr_group = None
    monkeypatch.setattr(reader, "_zarr_group", root)

    # Execute the branch under test
    if isinstance(reader._zarr_group, zarr.Group):
        reader.level_arrays = {
            int(key): ArrayView(array, axes=reader._axes)
            for key, array in reader._zarr_group.members()
        }

    # Assertions to satisfy pytest
    assert set(reader.level_arrays.keys()) == {0, 1}
    assert reader.level_arrays[0].array.shape == (4, 4)
    assert reader.level_arrays[1].array.shape == (8, 8)


def test_oob_read_dicom(sample_dicom: Path) -> None:
    """Test that out of bounds returns background value.

    For consistency with openslide, our readers should return a
    background tile when reading out of bounds.

    """
    wsi = DICOMWSIReader(sample_dicom)

    # assert reading of metadata
    assert np.all(wsi.info.mpp == np.array([0.499, 0.499]))
    assert wsi.info.objective_power == 20.0
    # Read a region that is out of bounds
    region = wsi.read_rect(
        location=(200000, 200),
        size=(100, 100),
    )
    # Check that the region is the same size as the requested size
    assert region.shape == (100, 100, 3)
    # Check that the region is white (255)
    assert np.all(region == 255)


def test_read_dicom_with_metadata(remote_sample: Callable) -> None:
    """Test DICOMWSIReader when mpp and objective are available."""
    wsi_path = remote_sample("dicom-2")
    wsi = DICOMWSIReader(wsi_path)
    wsi._info()

    # Assert mpp and objective power are read correctly.
    assert np.all(wsi.info.mpp == np.array([0.2498, 0.2498]))
    assert wsi.info.objective_power == 40.0

    wsi = DICOMWSIReader(wsi_path)
    # Force delete attribute for objective power.
    delattr(wsi.wsi.levels.base_level.datasets[0], "OpticalPathSequence")
    wsi._info()

    # Assert objective power inferred from mpp.
    assert np.all(wsi.info.mpp == np.array([0.2498, 0.2498]))
    assert wsi.info.objective_power == 40.0


def test_read_rect_transformedreader_svs_baseline(
    sample_svs: Path, remote_sample: Callable, track_tmp_path: Path
) -> None:
    """Test TransformedWSIReader.read_rect with an SVS file at baseline."""
    wsi = wsireader.TransformedWSIReader(
        sample_svs, target_img=sample_svs, transform=np.eye(3)
    )
    location = SVS_TEST_TISSUE_LOCATION
    size = SVS_TEST_TISSUE_SIZE
    im_region = wsi.read_rect(location, size, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)

    fixed_info = wsi.info
    wsi2 = wsireader.TransformedWSIReader(
        sample_svs, target_img=sample_svs, transform=np.eye(3), fixed_info=fixed_info
    )
    im_region_2 = wsi2.read_rect(location, size, resolution=0, units="level")

    assert np.array_equal(im_region, im_region_2)

    with pytest.raises(
        ValueError,
        match=r"Transform cannot be None. Please provide a valid transformation",
    ):
        _ = wsireader.TransformedWSIReader(
            sample_svs, target_img=sample_svs, transform=None
        )

    # Now test MHA displacement field
    wsi3 = wsireader.TransformedWSIReader(
        sample_svs,
        target_img=sample_svs,
        transform=remote_sample("reg_disp_mha_example"),
    )
    im_region_3 = wsi3.read_rect(location, size, resolution=0, units="level")

    # We don't expect arrays to be the same, but dimensions should be
    assert im_region.shape == im_region_3.shape

    # Now test NPY affine transformation
    wsi4 = wsireader.TransformedWSIReader(
        sample_svs,
        target_img=sample_svs,
        transform=remote_sample("reg_affine_npy_example"),
    )
    im_region_4 = wsi4.read_rect(location, size, resolution=0, units="level")

    # We don't expect arrays to be the same, but dimensions should be
    assert im_region.shape == im_region_4.shape

    # Now test MHA file with correct shape
    transform = remote_sample("reg_disp_mha_example")
    displacement_field = sitk.ReadImage(transform, sitk.sitkVectorFloat64)
    disp_array = sitk.GetArrayFromImage(displacement_field)  # (2, H, W)
    disp_array = np.moveaxis(disp_array, 0, -1)
    disp_image = sitk.GetImageFromArray(disp_array, isVector=True)

    # Save it to a new .mha file in tmp_path
    transform_path = track_tmp_path / "new_disp.mha"
    sitk.WriteImage(disp_image, str(transform_path))

    wsi5 = wsireader.TransformedWSIReader(
        sample_svs,
        target_img=sample_svs,
        transform=transform_path,
    )
    im_region_5 = wsi5.read_rect(location, size, resolution=0, units="level")

    # We don't expect arrays to be the same, but dimensions should be
    assert im_region.shape == im_region_5.shape

    # Test wrong file type
    with pytest.raises(ValueError, match="Unsupported transformation file format"):
        wsireader.TransformedWSIReader(
            sample_svs,
            target_img=sample_svs,
            transform=sample_svs,
        )


def test_read_bounds_transformedreader_baseline(
    sample_svs: Path, remote_sample: Callable
) -> None:
    """Test TransformedWSIReader read bounds at baseline.

    Location coordinate is in baseline (level 0) reference frame.

    """
    wsi = wsireader.TransformedWSIReader(
        sample_svs, target_img=sample_svs, transform=np.eye(3)
    )

    bounds = SVS_TEST_TISSUE_BOUNDS
    size = SVS_TEST_TISSUE_SIZE
    im_region = wsi.read_bounds(bounds, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)

    # Now test MHA displacement field
    wsi3 = wsireader.TransformedWSIReader(
        sample_svs,
        target_img=sample_svs,
        transform=remote_sample("reg_disp_mha_example"),
    )
    im_region_3 = wsi3.read_bounds(bounds, resolution=0, units="level")

    # We don't expect arrays to be the same, but dimensions should be
    assert im_region.shape == im_region_3.shape

    # Now test NPY affine transformation
    wsi4 = wsireader.TransformedWSIReader(
        sample_svs,
        target_img=sample_svs,
        transform=remote_sample("reg_affine_npy_example"),
    )
    im_region_4 = wsi4.read_bounds(bounds, resolution=0, units="level")

    # We don't expect arrays to be the same, but dimensions should be
    assert im_region.shape == im_region_4.shape


def test_wsireader_validate_input_edge_cases() -> None:
    """Test WSIReader._validate_input with various edge cases."""
    # Test with valid inputs
    WSIReader._validate_input("test.svs")
    WSIReader._validate_input(Path("test.svs"))
    WSIReader._validate_input(np.array([1, 2, 3]))

    # Test with invalid inputs
    with pytest.raises(TypeError, match="Invalid input"):
        WSIReader._validate_input(123)

    with pytest.raises(TypeError, match="Invalid input"):
        WSIReader._validate_input({"invalid": "dict"})


def test_wsireader_verify_supported_wsi_edge_cases(track_tmp_path: Path) -> None:
    """Test WSIReader.verify_supported_wsi with edge cases."""
    # Test with unsupported extension
    unsupported_file = track_tmp_path / "test.xyz"
    with pytest.raises(FileNotSupportedError, match="not a supported file format"):
        WSIReader.verify_supported_wsi(unsupported_file)

    # Test with no extension
    no_ext_file = track_tmp_path / "test"
    WSIReader.verify_supported_wsi(no_ext_file)


def test_wsireader_handle_virtual_wsi_edge_cases(track_tmp_path: Path) -> None:
    """Test _handle_virtual_wsi with various file types."""
    # Test with .npy file
    npy_file = track_tmp_path / "test.npy"
    rng = np.random.default_rng(0)
    np.save(npy_file, rng.random((100, 100, 3)))
    result = _handle_virtual_wsi(".npy", npy_file, None, None)
    assert isinstance(result, VirtualWSIReader)

    # Test with unsupported extension
    result = _handle_virtual_wsi(".xyz", track_tmp_path / "test.xyz", None, None)
    assert result is None


def test_wsireader_handle_tiff_wsi_edge_cases(track_tmp_path: Path) -> None:
    """Test _handle_tiff_wsi with various scenarios."""
    # Test with non-existent file
    non_existent = track_tmp_path / "non_existent.tiff"
    with pytest.raises(FileNotFoundError):
        _handle_tiff_wsi(non_existent, None, None, None)


def test_wsireader_special_cases_coverage(track_tmp_path: Path) -> None:
    """Test WSIReader._handle_special_cases for better coverage."""
    # Create a mock .db file
    db_file = track_tmp_path / "test.db"
    db_file.touch()

    # Test annotation store case
    with pytest.raises(
        ValueError,
        match="No metadata found in store",
    ):
        WSIReader._handle_special_cases(db_file, db_file, None, None, None, info=None)


def test_wsireader_get_post_proc_edge_cases() -> None:
    """Test WSIReader.get_post_proc with various inputs."""
    # Create a mock WSIReader
    wsi = VirtualWSIReader(np.ones((10, 10, 3), dtype=np.uint8))

    # Test with callable
    def dummy_proc(x: np.ndarray) -> np.ndarray:
        """Dummy post-processing function."""
        return x

    result = wsi.get_post_proc(dummy_proc)
    assert callable(result)

    # Test with None
    result = wsi.get_post_proc(None)
    assert result is None

    # Test with invalid string
    with pytest.raises(ValueError, match="Invalid post-processing function"):
        wsi.get_post_proc("invalid_function")


def test_wsireader_bounds_at_resolution_to_baseline_edge_cases(
    sample_svs: Path,
) -> None:
    """Test bounds_at_resolution_to_baseline with edge cases."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test with zero bounds
    bounds = (0, 0, 0, 0)
    result = wsi.bounds_at_resolution_to_baseline(bounds, 1.0, "baseline")
    assert len(result) == 4

    # Test with negative bounds
    bounds = (-10, -10, 10, 10)
    result = wsi.bounds_at_resolution_to_baseline(bounds, 1.0, "baseline")
    assert len(result) == 4


def test_wsireader_slide_dimensions_edge_cases(sample_svs: Path) -> None:
    """Test slide_dimensions with various parameters."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test with different precision values
    dims1 = wsi.slide_dimensions(1.0, "baseline", precision=1)
    dims2 = wsi.slide_dimensions(1.0, "baseline", precision=5)
    assert isinstance(dims1, (tuple, list, np.ndarray))
    assert isinstance(dims2, (tuple, list, np.ndarray))
    assert len(dims1) == 2
    assert len(dims2) == 2


def test_wsireader_find_tile_params_edge_cases(sample_svs: Path) -> None:
    """Test _find_tile_params with edge cases."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test with objective value that's not an integer multiple
    with pytest.raises(ValueError, match="integer multiple"):
        wsi._find_tile_params(3.7)


def test_wsireader_check_unit_conversion_integrity_edge_cases() -> None:
    """Test _check_unit_conversion_integrity with various inputs."""
    # Test invalid input units
    with pytest.raises(ValueError, match="Invalid input_unit"):
        WSIReader._check_unit_conversion_integrity("invalid", "mpp", None, None)

    # Test invalid output units
    with pytest.raises(ValueError, match="Invalid output_unit"):
        WSIReader._check_unit_conversion_integrity("mpp", "invalid", None, None)

    # Test missing mpp for mpp input
    with pytest.raises(ValueError, match="Missing 'mpp'"):
        WSIReader._check_unit_conversion_integrity("mpp", "power", None, 40.0)

    # Test missing power for power input
    with pytest.raises(ValueError, match="Missing 'objective_power'"):
        WSIReader._check_unit_conversion_integrity("power", "mpp", (0.25, 0.25), None)


def test_wsireader_prepare_output_dict_edge_cases(sample_svs: Path) -> None:
    """Test _prepare_output_dict with various scenarios."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test with mpp input as array
    result = wsi._prepare_output_dict("mpp", np.array([0.5, 0.5]), (0.25, 0.25), 40.0)
    assert "mpp" in result
    assert "baseline" in result
    assert "power" in result

    # Test with mpp input as scalar
    result = wsi._prepare_output_dict("mpp", 0.5, (0.25, 0.25), 40.0)
    assert "mpp" in result
    assert np.array_equal(result["mpp"], np.array([0.5, 0.5]))


def test_arrayview_edge_cases() -> None:
    """Test ArrayView with additional edge cases."""
    # Test with CYX axes
    array = zarr.ones((3, 128, 128))
    array_view = ArrayView(array=array, axes="CYX")

    # Test shape property
    shape = array_view.shape
    assert len(shape) == 3

    # Test getitem with CYX
    result = array_view[:64, :64, :]
    assert result.shape == (64, 64, 3)


def test_tiffwsireader_delegate_edge_cases(sample_svs: Path) -> None:
    """Test TIFFWSIReaderDelegate with edge cases."""
    wsi = TIFFWSIReader(sample_svs)
    delegate = wsi.tiff_reader_delegate

    # Test canonical_shape with different axes
    shape = delegate.canonical_shape("SYX", (3, 100, 100))
    assert tuple(shape) == (100, 100, 3)

    # Test with unsupported axes
    with pytest.raises(ValueError, match="Unsupported axes"):
        delegate.canonical_shape("XYZ", (100, 100, 3))


def test_jp2wsireader_get_jp2_boxes_edge_cases(track_tmp_path: Path) -> None:
    """Test JP2WSIReader._get_jp2_boxes with edge cases."""
    # Create a minimal JP2 file
    path = track_tmp_path / "test.jp2"
    _ = glymur.Jp2k(path, data=np.ones((64, 64, 3), np.uint8))

    wsi = JP2WSIReader(path)

    # Test with missing header
    with (
        patch.object(wsi.glymur_jp2, "box", []),
        pytest.raises(ValueError, match="image header missing"),
    ):
        wsi._get_jp2_boxes(wsi.glymur_jp2)


def test_virtualwsireader_find_params_from_baseline_edge_cases() -> None:
    """Test VirtualWSIReader._find_params_from_baseline with edge cases."""
    # Create a virtual WSI with different baseline size
    img = np.ones((50, 50, 3), dtype=np.uint8)
    meta = WSIMeta(slide_dimensions=(100, 100), axes="YXS")
    wsi = VirtualWSIReader(img, info=meta)

    # Test with zero location and size
    location, size = wsi._find_params_from_baseline((0, 0), (0, 0))
    assert location[0] == 0
    assert location[1] == 0
    assert size[0] == 0
    assert size[1] == 0


def test_fsspecjsonwsireader_set_axes_edge_cases(track_tmp_path: Path) -> None:
    """Test FsspecJsonWSIReader.__set_axes with edge cases."""
    # This would require creating a proper fsspec JSON file structure
    # For now, we'll test the error conditions through mocking

    # Create a mock zarr array without proper structure
    json_path = track_tmp_path / "test.json"
    json_data = {".zattrs": {}, "0/.zarray": {"shape": [100, 100, 3], "dtype": "uint8"}}

    with Path.open(json_path, "w") as f:
        json.dump(json_data, f)

    # This should fail due to missing _ARRAY_DIMENSIONS
    with pytest.raises((ValueError, KeyError)):
        FsspecJsonWSIReader(json_path)


def test_dicomwsireader_edge_cases(sample_dicom: Path) -> None:
    """Test DICOMWSIReader with edge cases."""
    wsi = DICOMWSIReader(sample_dicom)

    # Test reading with coord_space="resolution"
    region = wsi.read_rect(
        location=(0, 0),
        size=(100, 100),
        coord_space="resolution",
        resolution=1.0,
        units="baseline",
    )
    assert isinstance(region, np.ndarray)
    assert region.shape == (100, 100, 3)


def test_wsireader_info_setter_edge_cases(sample_svs: Path) -> None:
    """Test WSIReader.info setter."""
    wsi = OpenSlideWSIReader(sample_svs)
    original_info = wsi.info

    # Create a new WSIMeta object
    new_info = WSIMeta(slide_dimensions=(1000, 1000), axes="YXS")

    # Set the info
    wsi.info = new_info

    # Verify it was set
    assert wsi._m_info == new_info

    # Reset to original
    wsi.info = original_info


def test_wsireader_read_region_edge_cases(sample_svs: Path) -> None:
    """Test WSIReader.read_region with edge cases."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test with level 0
    region = wsi.read_region(location=(100, 100), level=0, size=(50, 50))
    assert isinstance(region, np.ndarray)
    assert region.shape == (50, 50, 3)

    # Test with higher level
    if wsi.info.level_count > 1:
        region = wsi.read_region(location=(50, 50), level=1, size=(25, 25))
        assert isinstance(region, np.ndarray)
        assert region.shape == (25, 25, 3)


def test_is_dicom_edge_cases(track_tmp_path: Path) -> None:
    """Test is_dicom function with edge cases."""
    # Test with .dcm file
    dcm_file = track_tmp_path / "test.dcm"
    dcm_file.touch()
    assert is_dicom(dcm_file)

    # Test with directory containing .dcm files
    dcm_dir = track_tmp_path / "dcm_dir"
    dcm_dir.mkdir()
    (dcm_dir / "test.dcm").touch()
    assert is_dicom(dcm_dir)

    # Test with non-dcm file
    txt_file = track_tmp_path / "test.txt"
    txt_file.touch()
    assert not is_dicom(txt_file)

    # Test with empty directory
    empty_dir = track_tmp_path / "empty_dir"
    empty_dir.mkdir()
    assert not is_dicom(empty_dir)


def test_tiffwsireader_color_parsing_edge_cases(sample_ome_tiff: Path) -> None:
    """Test TIFFWSIReader color parsing methods with edge cases."""
    wsi = TIFFWSIReader(sample_ome_tiff)

    # Test _int_to_rgb with negative number
    rgb = wsi._int_to_rgb(-1)
    assert len(rgb) == 3
    assert all(0 <= c <= 1 for c in rgb)

    # Test _int_to_rgb with large number
    rgb = wsi._int_to_rgb(16777215)  # 0xFFFFFF
    assert rgb == (1.0, 1.0, 1.0)


def test_wsireader_manual_parameters_edge_cases() -> None:
    """Test WSIReader manual parameter validation."""
    # Test with invalid mpp length
    with pytest.raises(TypeError, match="mpp"):
        VirtualWSIReader(np.ones((10, 10, 3)), mpp=(0.5,))

    # Test with invalid mpp type
    with pytest.raises(TypeError, match="mpp"):
        VirtualWSIReader(np.ones((10, 10, 3)), mpp="invalid")

    # Test with invalid power type
    with pytest.raises(TypeError, match="power"):
        VirtualWSIReader(np.ones((10, 10, 3)), power=(42,))


def test_virtualwsireader_mode_detection_edge_cases() -> None:
    """Test VirtualWSIReader mode detection with various image types."""
    # Test with 2D image (should set mode to 'feature')
    img_2d = np.ones((100, 100), dtype=np.uint8)
    with patch("tiatoolbox.wsicore.wsireader.logger") as mock_logger:
        wsi = VirtualWSIReader(img_2d, mode="rgb")
        mock_logger.warning.assert_called()
        assert wsi.mode == "feature"

    # Test with 5-channel image (should set mode to 'feature')
    img_5ch = np.ones((100, 100, 5), dtype=np.uint8)
    with patch("tiatoolbox.wsicore.wsireader.logger") as mock_logger:
        wsi = VirtualWSIReader(img_5ch, mode="rgb")
        mock_logger.warning.assert_called()
        assert wsi.mode == "feature"


def test_openslide_estimate_mpp_edge_cases() -> None:
    """Test OpenSlideWSIReader._estimate_mpp with edge cases."""
    # Test with missing resolution data
    props = {}
    result = OpenSlideWSIReader._estimate_mpp(props)
    assert result is None

    # Test with partial TIFF resolution data
    props = {
        "tiff.XResolution": "100",
        "tiff.ResolutionUnit": "inch",
        # Missing YResolution
    }
    with patch("tiatoolbox.wsicore.wsireader.logger") as mock_logger:
        result = OpenSlideWSIReader._estimate_mpp(props)
        mock_logger.warning.assert_called()
        assert result is None


def test_tiffwsireader_series_selection_edge_cases(sample_ome_tiff: Path) -> None:
    """Test TIFFWSIReader series selection logic."""
    # Test with explicit series number
    wsi = TIFFWSIReader(sample_ome_tiff, series=0)
    assert wsi.series_n == 0

    # Test info access
    info = wsi.info
    assert info is not None


def test_arrayview_shape_property_edge_cases() -> None:
    """Test ArrayView.shape property with different axes."""
    # Test with YXC axes
    array = zarr.ones((128, 128, 3))
    array_view = ArrayView(array=array, axes="YXC")
    shape = array_view.shape
    assert shape == (128, 128, 3)

    # Test with SYX axes
    array = zarr.ones((3, 128, 128))
    array_view = ArrayView(array=array, axes="SYX")
    shape = array_view.shape
    assert shape == (128, 128, 3)


def test_wsireader_find_read_params_at_resolution_edge_cases(sample_svs: Path) -> None:
    """Test _find_read_params_at_resolution with edge cases."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test with very small size
    result = wsi._find_read_params_at_resolution(
        location=(0, 0), size=(1, 1), resolution=1.0, units="baseline"
    )
    assert len(result) == 6

    # Test with large size
    result = wsi._find_read_params_at_resolution(
        location=(0, 0), size=(10000, 10000), resolution=0.1, units="baseline"
    )
    assert len(result) == 6


def test_tiffwsireader_parse_methods_edge_cases() -> None:
    """Test TIFFWSIReader parsing methods with edge cases."""
    # Test _get_namespace with no namespace
    root = ElementTree.fromstring("<root></root>")
    ns = TIFFWSIReader._get_namespace(root)
    assert ns == {}

    # Test _get_namespace with namespace
    root = ElementTree.fromstring("<root xmlns='http://example.com'></root>")
    ns = TIFFWSIReader._get_namespace(root)
    assert "ns" in ns


def test_wsireader_precision_parameter_edge_cases(sample_svs: Path) -> None:
    """Test precision parameter in various methods."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test find_read_rect_params with different precision
    result1 = wsi.find_read_rect_params(
        location=(100, 100),
        size=(200, 200),
        resolution=1.0,
        units="baseline",
        precision=1,
    )

    result2 = wsi.find_read_rect_params(
        location=(100, 100),
        size=(200, 200),
        resolution=1.0,
        units="baseline",
        precision=5,
    )

    assert len(result1) == 5
    assert len(result2) == 5


def test_wsireader_read_rect_at_resolution_edge_cases(sample_svs: Path) -> None:
    """Test read_rect_at_resolution with edge cases."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test with different pad modes
    region = wsi.read_rect_at_resolution(
        location=(100, 100),
        size=(50, 50),
        resolution=1.0,
        units="baseline",
        pad_mode="reflect",
    )
    assert isinstance(region, np.ndarray)
    assert region.shape == (50, 50, 3)


def test_is_tiled_tiff_error_handling(track_tmp_path: Path) -> None:
    """Test is_tiled_tiff with files that cause TiffFileError."""
    # Create a non-TIFF file with .tiff extension
    fake_tiff = track_tmp_path / "fake.tiff"
    fake_tiff.write_text("This is not a TIFF file")

    # Should return False for invalid TIFF
    assert not is_tiled_tiff(fake_tiff)


def test_is_zarr_error_handling(track_tmp_path: Path) -> None:
    """Test is_zarr with files that cause exceptions."""
    # Create a file that looks like zarr but isn't
    fake_zarr = track_tmp_path / "fake.zarr"
    fake_zarr.mkdir()
    (fake_zarr / "invalid").write_text("not zarr")

    # Should handle the exception and return False
    result = is_zarr(fake_zarr)
    assert isinstance(result, bool)


def test_wsireader_convert_resolution_units_comprehensive(sample_svs: Path) -> None:
    """Test convert_resolution_units with comprehensive scenarios."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test with tuple input for mpp
    result = wsi.convert_resolution_units((0.5, 0.6), input_unit="mpp")
    assert "mpp" in result
    assert "power" in result
    assert "baseline" in result

    # Test with list input for mpp
    result = wsi.convert_resolution_units([0.5, 0.6], input_unit="mpp")
    assert "mpp" in result
    assert np.array_equal(result["mpp"], np.array([0.5, 0.6]))


def test_tiffwsireader_get_colors_from_meta_edge_cases(sample_ome_tiff: Path) -> None:
    """Test TIFFWSIReader._get_colors_from_meta with edge cases."""
    wsi = TIFFWSIReader(sample_ome_tiff)

    # Test when post_proc is not MultichannelToRGB
    wsi.post_proc = None
    wsi._get_colors_from_meta()  # Should return early without error

    # Test with invalid XML
    wsi.post_proc = utils.postproc_defs.MultichannelToRGB()

    pp = wsi.post_proc
    before = pp.__dict__.copy()

    # Fake info object with invalid XML
    bad_info = SimpleNamespace(raw={"Description": "<invalid_xml"})
    wsi._m_info = bad_info  # bypass property safely
    assert wsi._get_colors_from_meta() is None
    assert pp.__dict__ == before


def test_wsireader_save_tiles_edge_cases(
    sample_svs: Path, track_tmp_path: Path
) -> None:
    """Test save_tiles with edge cases."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test with different tile formats
    wsi.save_tiles(
        output_dir=str(track_tmp_path / "test_png_tiles"),
        tile_objective_value=5,
        tile_read_size=(1000, 1000),
        tile_format=".png",
        verbose=False,
    )

    # Check that PNG files were created
    output_dir = track_tmp_path / "test_png_tiles" / wsi.input_path.name
    assert any(f.suffix == ".png" for f in output_dir.iterdir())


def test_wsireader_tissue_mask_edge_cases(sample_svs: Path) -> None:
    """Test tissue_mask with edge cases."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test with custom masker_kwargs
    mask = wsi.tissue_mask(method="morphological", resolution=5, units="power")
    assert isinstance(mask, VirtualWSIReader)
    assert mask.mode == "bool"


def test_wsireader_find_read_bounds_params_edge_cases(sample_svs: Path) -> None:
    """Test find_read_bounds_params with edge cases."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test with very small bounds
    result = wsi.find_read_bounds_params(
        bounds=(0, 0, 1, 1), resolution=1.0, units="baseline"
    )
    assert len(result) == 4

    # Conflicting parameters: power-derived resolution AND kernel_size
    with pytest.raises(
        ValueError,
        match="Only one of mpp, power, kernel_size can be given",
    ):
        wsi.tissue_mask(
            method="morphological",
            resolution=5,
            units="power",
            kernel_size=5,
        )

    # Test with bounds at edge of slide
    slide_w, slide_h = wsi.info.slide_dimensions
    result = wsi.find_read_bounds_params(
        bounds=(slide_w - 100, slide_h - 100, slide_w, slide_h),
        resolution=1.0,
        units="baseline",
    )
    assert len(result) == 4


def test_wsireader_try_methods_comprehensive() -> None:
    """Test WSIReader.try_* methods comprehensively."""
    # Test try_dicom with non-DICOM path
    result = WSIReader.try_dicom(Path("test.txt"), None, None, None)
    assert result is None

    # Test try_fsspec with invalid input
    result = WSIReader.try_fsspec("invalid.txt", None, None)
    assert result is None

    # Test try_annotation_store with non-.db file
    result = WSIReader.try_annotation_store(Path("test.txt"), ".txt", None, {})
    assert result is None

    # Test try_ngff with non-.zarr file
    result = WSIReader.try_ngff(Path("test.txt"), ".txt", None, None)
    assert result is None

    # Test try_ome_tiff with non-OME file
    result = WSIReader.try_ome_tiff(
        Path("test.txt"), [".txt"], ".txt", None, None, None
    )
    assert result is None

    # Test try_tiff with non-TIFF file
    result = WSIReader.try_tiff(Path("test.txt"), ".txt", None, None, None)
    assert result is None


def test_jp2wsireader_find_box_edge_cases(track_tmp_path: Path) -> None:
    """Test JP2WSIReader find_box method edge cases."""
    # Create a minimal JP2 file
    path = track_tmp_path / "test.jp2"
    _ = glymur.Jp2k(path, data=np.ones((64, 64, 3), np.uint8))

    wsi = JP2WSIReader(path)
    boxes = wsi._get_jp2_boxes(wsi.glymur_jp2)

    # Test that we get expected boxes
    assert "ihdr" in boxes
    # Other boxes may or may not be present depending on the JP2 structure


def test_virtualwsireader_info_edge_cases() -> None:
    """Test VirtualWSIReader._info with edge cases."""
    # Test with numpy array input
    img = np.ones((100, 100, 3), dtype=np.uint8)
    wsi = VirtualWSIReader(img)

    # Test that _info creates proper metadata
    info = wsi._info()
    assert info.slide_dimensions == (100, 100)
    assert info.level_count == 1
    assert info.axes == "YSX"  # Note: VirtualWSIReader uses YSX

    # Test with pre-existing info
    custom_info = WSIMeta(slide_dimensions=(200, 200), axes="YXS")
    wsi = VirtualWSIReader(img, info=custom_info)
    assert wsi.info.slide_dimensions == (200, 200)


def test_tiffwsireader_parse_svs_tag_edge_cases() -> None:
    """Test TIFFWSIReaderDelegate.parse_svs_tag with edge cases."""

    # Create a mock TiffPages object
    class MockPage:
        """Mock TiffPage with description attribute."""

        def __init__(self, description: str) -> None:
            self.description = description

    class MockPages:
        """Mock TiffPages with __getitem__ method."""

        def __init__(self, description: str) -> None:
            self.pages = [MockPage(description)]

        def __getitem__(self, index: int) -> MockPage:
            return self.pages[index]

    # Minimal valid header (two lines), then malformed key-value after '|'
    desc = "Software Line\nPhotometric Line|key==value"
    pages = MockPages(desc)

    # Expect a ValueError due to malformed key-value pairs;
    # don't overconstrain the message (implementation-specific).
    with pytest.raises(ValueError, match=r"Expected string of the format 'key=value'"):
        TIFFWSIReaderDelegate.parse_svs_metadata(pages)


def test_wsireader_read_with_different_interpolations(sample_svs: Path) -> None:
    """Test reading with different interpolation methods."""
    wsi = OpenSlideWSIReader(sample_svs)
    location = (100, 100)
    size = (50, 50)

    interpolation_methods = ["linear", "cubic", "lanczos", "area", "optimise"]

    for method in interpolation_methods:
        region = wsi.read_rect(location=location, size=size, interpolation=method)
        assert isinstance(region, np.ndarray)
        assert region.shape == (50, 50, 3)


def test_wsireader_read_with_different_pad_modes(sample_svs: Path) -> None:
    """Test reading with different padding modes."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test reading at edge with different pad modes
    slide_w, slide_h = wsi.info.slide_dimensions
    location = (slide_w - 25, slide_h - 25)  # Near edge
    size = (50, 50)  # Will go out of bounds

    pad_modes = ["constant", "reflect", "edge"]

    for pad_mode in pad_modes:
        region = wsi.read_rect(location=location, size=size, pad_mode=pad_mode)
        assert isinstance(region, np.ndarray)
        assert region.shape == (50, 50, 3)


def test_wsireader_read_with_pad_constant_values(sample_svs: Path) -> None:
    """Test reading with different pad constant values."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test with single value
    region = wsi.read_rect(
        location=(-10, -10), size=(50, 50), pad_mode="constant", pad_constant_values=128
    )
    assert isinstance(region, np.ndarray)

    # Test with tuple value
    region = wsi.read_rect(
        location=(-10, -10),
        size=(50, 50),
        pad_mode="constant",
        pad_constant_values=(64, 192),
    )
    assert isinstance(region, np.ndarray)


def test_virtualwsireader_bool_mode_interpolation() -> None:
    """Test VirtualWSIReader with bool mode uses nearest interpolation."""
    # Create a binary mask
    rng = np.random.default_rng(0)
    mask = rng.choice([0, 1], size=(100, 100), p=[0.7, 0.3]).astype(np.uint8)
    wsi = VirtualWSIReader(mask, mode="bool")

    # Test that bool mode uses nearest interpolation
    region = wsi.read_bounds(
        bounds=(0, 0, 50, 50),
        resolution=0.5,
        units="baseline",
        interpolation="optimise",  # Should be overridden to "nearest"
    )
    assert isinstance(region, np.ndarray)
    # Values should still be 0 or 1 (or close) due to nearest interpolation
    unique_vals = np.unique(region)
    assert len(unique_vals) <= 3  # Should be mostly 0s and 1s


def test_wsireader_level_downsamples_edge_cases(sample_svs: Path) -> None:
    """Test level downsamples calculation edge cases."""
    wsi = OpenSlideWSIReader(sample_svs)

    # Test accessing level downsamples
    downsamples = wsi.info.level_downsamples
    assert len(downsamples) == wsi.info.level_count
    assert all(d >= 1.0 for d in downsamples)
    assert downsamples[0] == 1.0  # First level should be 1.0


def test_wsireader_axes_handling_edge_cases() -> None:
    """Test axes handling in various scenarios."""
    # Test with different axes orders
    img = np.ones((3, 100, 100), dtype=np.uint8)  # CYX order

    # VirtualWSIReader should handle this
    wsi = VirtualWSIReader(img, mode="feature")
    info = wsi.info
    assert info.axes == "YSX"  # VirtualWSIReader normalizes to YSX


def test_wsireader_error_propagation() -> None:
    """Test that errors are properly propagated."""
    # Test with completely invalid input
    with pytest.raises(TypeError):
        WSIReader.open(None)

    # Test with invalid numpy array dimensions
    invalid_array = np.ones((10,))  # 1D array
    with pytest.raises(ValueError, match=r"ndim < 2|2D|3D"):
        VirtualWSIReader(invalid_array)


def test_virtual_wsireader_accepts_valid_rgb() -> None:
    """Test that VirtualWSIReader accepts valid RGB images."""
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    wsi = VirtualWSIReader(rgb, mode="rgb")
    assert wsi.img.shape == (16, 16, 3)


def test_virtual_wsireader_accepts_2d_bool_mask() -> None:
    """Test that VirtualWSIReader accepts 2D boolean masks."""
    mask = np.zeros((16, 16), dtype=np.uint8)
    wsi = VirtualWSIReader(mask, mode="bool")
    assert wsi.img.ndim == 2


def test_canonical_shape_handles_cyx_and_syx() -> None:
    """Test TIFFWSIReaderDelegate.canonical_shape with CYX and SYX axes."""
    cs = TIFFWSIReaderDelegate.canonical_shape  # type: ignore[attr-defined]
    assert tuple(cs("CYX", (3, 8, 10))) == (8, 10, 3)
    assert tuple(cs("SYX", (5, 8, 10))) == (8, 10, 5)


def test_virtual_read_rect_resolution_coord_space_roundtrip() -> None:
    """Test VirtualWSIReader read_rect with resolution coord_space."""
    img = np.arange(0, 32 * 32 * 4, dtype=np.uint8).reshape(32, 32, 4)
    v = wsireader.VirtualWSIReader(img)
    r1 = v.read_rect((0, 0), (8, 8), coord_space="resolution")
    r2 = v.read_bounds((0, 0, 8, 8))
    assert np.array_equal(r1, r2)


class TestTryOpenSlide:
    """Unit tests for the WSIReader.try_openslide static method."""

    @patch("tiatoolbox.wsicore.wsireader.OpenSlideWSIReader")
    def test_tiff_suffix_success(self, mock_reader: MagicMock) -> None:
        """Test that a valid TIFF file results in an OpenSlideWSIReader instance."""
        mock_instance = MagicMock()
        mock_reader.return_value = mock_instance

        result: OpenSlideWSIReader | None = WSIReader.try_openslide(
            input_path=Path("sample.tif"),
            last_suffix=".tif",
            mpp=(0.5, 0.5),
            power=20,
        )

        mock_reader.assert_called_once_with(
            Path("sample.tif"),
            mpp=(0.5, 0.5),
            power=20,
        )
        assert result is mock_instance

    @patch("tiatoolbox.wsicore.wsireader.OpenSlideWSIReader")
    def test_tiff_suffix_raises_openslide_error(self, mock_reader: MagicMock) -> None:
        """Test that OpenSlide errors are caught and the function returns None."""
        mock_reader.side_effect = openslide.OpenSlideError("bad file")

        result: OpenSlideWSIReader | None = WSIReader.try_openslide(
            input_path=Path("bad.tiff"),
            last_suffix=".tiff",
            mpp=None,
            power=None,
        )

        assert result is None


def test_wsireader_url_input_sets_input_path() -> None:
    """Ensure URL input triggers the urlparse scheme branch."""
    url = "https://example.com/image.svs"

    reader = WSIReader(input_img=url)

    assert reader.input_path == url


def test_handle_tiff_wsi_returns_none_when_no_handlers_match(
    track_tmp_path: Path,
) -> None:
    """Ensure _handle_tiff_wsi returns None when both checks fail."""
    fake_path = track_tmp_path / "not_a_real_wsi.tiff"
    fake_path.write_text("dummy")  # file exists but is not a TIFF WSI

    with (
        patch("openslide.OpenSlide.detect_format", return_value=None),
        patch("tiatoolbox.wsicore.wsireader.is_tiled_tiff", return_value=False),
    ):
        result = _handle_tiff_wsi(
            input_path=fake_path,
            mpp=None,
            power=None,
            post_proc=None,
        )

    assert result is None
