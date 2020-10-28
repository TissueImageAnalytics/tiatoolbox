#!/usr/bin/env python

"""Pytests for `tiatoolbox` package."""
import pytest
from pytest import approx

from tiatoolbox.dataloader.slide_info import slide_info

from tiatoolbox.dataloader.save_tiles import save_tiles
from tiatoolbox.dataloader import wsireader, wsimeta
from tiatoolbox.tools.stainnorm import get_normaliser
from tiatoolbox import utils

from tiatoolbox.utils.exceptions import FileNotSupported, MethodNotSupported
from tiatoolbox.utils.misc import imread

from tiatoolbox import cli
from tiatoolbox import __version__

from click.testing import CliRunner
import requests
import os
import pathlib
import numpy as np
import shutil

# -------------------------------------------------------------------------------------
# Utility Test Functions
# -------------------------------------------------------------------------------------


def strictly_increasing(seq):
    """Return True if sequence is strictly increasing.

    Args:
        seq: Sequence to check.

    Returns:
        bool: True if strictly increasing.
    """
    return all(a < b for a, b in zip(seq, seq[1:]))


def strictly_decreasing(seq):
    """Return True if sequence is strictly decreasing.

    Args:
        seq: Sequence to check.


    Returns:
        bool: True if strictly decreasing.
    """
    return all(a > b for a, b in zip(seq, seq[1:]))


# -------------------------------------------------------------------------------------
# Pytest Fixtures
# -------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _response_ndpi(tmpdir_factory):
    """Sample pytest fixture for ndpi images. Download ndpi image for pytest."""
    ndpi_file_path = tmpdir_factory.mktemp("data").join("CMU-1.ndpi")
    if not os.path.isfile(ndpi_file_path):
        print("\nDownloading NDPI")
        r = requests.get(
            "http://openslide.cs.cmu.edu/download/openslide-testdata"
            "/Hamamatsu/CMU-1.ndpi"
        )
        with open(ndpi_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping NDPI")

    return ndpi_file_path


@pytest.fixture(scope="session")
def _response_svs(tmpdir_factory):
    """Sample pytest fixture for svs images. Download ndpi image for pytest."""
    svs_file_path = tmpdir_factory.mktemp("data").join("CMU-1-Small-Region.svs")
    if not os.path.isfile(svs_file_path):
        print("\nDownloading SVS")
        r = requests.get(
            "http://openslide.cs.cmu.edu/download/openslide-testdata"
            "/Aperio/CMU-1-Small-Region.svs"
        )
        with open(svs_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping SVS")

    return svs_file_path


@pytest.fixture(scope="session")
def _response_jp2(tmpdir_factory):
    """Sample pytest fixture for JP2 images. Download ndpi image for pytest."""
    jp2_file_path = tmpdir_factory.mktemp("data").join("test1.jp2")
    if not os.path.isfile(jp2_file_path):
        print("\nDownloading JP2")
        r = requests.get(
            "https://warwick.ac.uk/fac/sci/dcs/research/tia/tiatoolbox/files"
            "/test2.jp2"
        )
        with open(jp2_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping JP2")

    return jp2_file_path


@pytest.fixture(scope="session")
def _response_all_wsis(_response_ndpi, _response_svs, _response_jp2, tmpdir_factory):
    """pytest fixture for sample wsi(s) of all types supported by tiatoolbox."""
    dir_path = pathlib.Path(tmpdir_factory.mktemp("data"))

    try:
        dir_path.joinpath(_response_ndpi.basename).symlink_to(_response_ndpi)
        dir_path.joinpath(_response_svs.basename).symlink_to(_response_svs)
        dir_path.joinpath(_response_jp2.basename).symlink_to(_response_jp2)
    except OSError:
        shutil.copy(_response_ndpi, dir_path.joinpath(_response_ndpi.basename))
        shutil.copy(_response_svs, dir_path.joinpath(_response_svs.basename))
        shutil.copy(_response_jp2, dir_path.joinpath(_response_jp2.basename))

    return dir_path


# -------------------------------------------------------------------------------------
# Python API tests
# -------------------------------------------------------------------------------------


def test_slide_info(_response_all_wsis, tmp_path):
    """Test for slide_info as a python function."""
    file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
    files_all = utils.misc.grab_files_from_dir(
        input_path=_response_all_wsis, file_types=file_types,
    )

    for curr_file in files_all:
        slide_param = slide_info(input_path=curr_file, verbose=True)
        out_path = tmp_path / slide_param.file_path.with_suffix(".yaml").name
        utils.misc.save_yaml(slide_param.as_dict(), out_path)


def test_wsireader_slide_info(_response_svs, tmp_path):
    """Test for slide_info in WSIReader class as a python function."""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent), file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    slide_param = wsi.slide_info
    out_path = tmp_path / slide_param.file_path.with_suffix(".yaml").name
    utils.misc.save_yaml(slide_param.as_dict(), out_path)


def test_relative_level_scales_openslide_baseline(_response_ndpi):
    """Test openslide relative level scales for pixels per baseline pixel."""
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)
    level_scales = wsi.relative_level_scales(0.125, "baseline")
    level_scales = np.array(level_scales)
    downsamples = np.array(wsi.slide_info.level_downsamples)
    expected = downsamples * 0.125
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test_relative_level_scales_jp2_baseline(_response_jp2):
    """Test jp2 relative level scales for pixels per baseline pixel."""
    wsi = wsireader.OmnyxJP2WSIReader(_response_jp2)
    level_scales = wsi.relative_level_scales(0.125, "baseline")
    level_scales = np.array(level_scales)
    downsamples = np.array(wsi.slide_info.level_downsamples)
    expected = downsamples * 0.125
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test_relative_level_scales_openslide_mpp(_response_ndpi):
    """Test openslide calculation of relative level scales for mpp."""
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)
    level_scales = wsi.relative_level_scales(0.5, "mpp")
    level_scales = np.array(level_scales)
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert all(level_scales[0] == wsi.slide_info.mpp / 0.5)


def test_relative_level_scales_jp2_mpp(_response_jp2):
    """Test jp2 calculation of relative level scales for mpp."""
    wsi = wsireader.OmnyxJP2WSIReader(_response_jp2)
    level_scales = wsi.relative_level_scales(0.5, "mpp")
    level_scales = np.array(level_scales)
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert all(level_scales[0] == wsi.slide_info.mpp / 0.5)


def test_relative_level_scales_openslide_power(_response_ndpi):
    """Test openslide calculation of relative level scales for objective power."""
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)
    level_scales = wsi.relative_level_scales(wsi.slide_info.objective_power, "power")
    level_scales = np.array(level_scales)
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert np.array_equal(level_scales[0], [1, 1])
    downsamples = np.array(wsi.slide_info.level_downsamples)
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], downsamples)


def test_relative_level_scales_jp2_power(_response_jp2):
    """Test jp2 calculation of relative level scales for objective power."""
    wsi = wsireader.OmnyxJP2WSIReader(_response_jp2)
    level_scales = wsi.relative_level_scales(wsi.slide_info.objective_power, "power")
    level_scales = np.array(level_scales)
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert np.array_equal(level_scales[0], [1, 1])
    downsamples = np.array(wsi.slide_info.level_downsamples)
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], downsamples)


def test_relative_level_scales_openslide_level(_response_ndpi):
    """Test openslide calculation of relative level scales for level."""
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)
    level_scales = wsi.relative_level_scales(3, "level")
    level_scales = np.array(level_scales)
    assert np.array_equal(level_scales[3], [1, 1])
    downsamples = np.array(wsi.slide_info.level_downsamples)
    expected = downsamples / downsamples[3]
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test_relative_level_scales_jp2_level(_response_jp2):
    """Test jp2 calculation of relative level scales for level."""
    wsi = wsireader.OmnyxJP2WSIReader(_response_jp2)
    level_scales = wsi.relative_level_scales(3, "level")
    level_scales = np.array(level_scales)
    assert np.array_equal(level_scales[3], [1, 1])
    downsamples = np.array(wsi.slide_info.level_downsamples)
    expected = downsamples / downsamples[3]
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test_relative_level_scales_openslide_level_float(_response_ndpi):
    """Test openslide calculation of relative level scales for fracitonal level."""
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)
    level_scales = wsi.relative_level_scales(1.5, "level")
    level_scales = np.array(level_scales)
    assert level_scales[0] == approx([1 / 3, 1 / 3])
    downsamples = np.array(wsi.slide_info.level_downsamples)
    expected = downsamples / downsamples[0] * (1 / 3)
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test_relative_level_scales_jp2_level_float(_response_jp2):
    """Test jp2 calculation of relative level scales for fracitonal level."""
    wsi = wsireader.OmnyxJP2WSIReader(_response_jp2)
    level_scales = wsi.relative_level_scales(1.5, "level")
    level_scales = np.array(level_scales)
    assert level_scales[0] == approx([1 / 3, 1 / 3])
    downsamples = np.array(wsi.slide_info.level_downsamples)
    expected = downsamples / downsamples[0] * (1 / 3)
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test_find_optimal_level_and_downsample_interpolation_warning(_response_ndpi):
    """Test finding optimal level for mpp read with scale > 1.

    This tests the case where the scale is found to be > 1 and interpolation
    will be applied to the output. A UserWarning should be raised in this case.
    """
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)
    with pytest.warns(UserWarning):
        _, _ = wsi.find_optimal_level_and_downsample(0.1, "mpp")


def test_find_optimal_level_and_downsample_mpp(_response_ndpi):
    """Test finding optimal level for mpp read."""
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)

    mpps = [0.5, 10]
    expected_levels = [0, 4]
    expected_scales = [[0.91282519, 0.91012514], [0.73026016, 0.72810011]]

    for mpp, expected_level, expected_scale in zip(
        mpps, expected_levels, expected_scales
    ):
        read_level, post_read_scale_factor = wsi.find_optimal_level_and_downsample(
            mpp, "mpp"
        )

        assert read_level == expected_level
        assert post_read_scale_factor == approx(expected_scale)


def test_find_optimal_level_and_downsample_power(_response_ndpi):
    """Test finding optimal level for objective power read."""
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)

    objective_powers = [20, 10, 5, 2.5, 1.25]
    expected_levels = [0, 1, 2, 3, 4]
    for objective_power, expected_level in zip(objective_powers, expected_levels):
        read_level, post_read_scale_factor = wsi.find_optimal_level_and_downsample(
            objective_power, "power"
        )

        assert read_level == expected_level
        assert np.array_equal(post_read_scale_factor, [1.0, 1.0])


def test_find_optimal_level_and_downsample_level(_response_ndpi):
    """Test finding optimal level for level read."""
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)

    for level in range(wsi.slide_info.level_count):
        read_level, post_read_scale_factor = wsi.find_optimal_level_and_downsample(
            level, "level"
        )

        assert read_level == level
        assert np.array_equal(post_read_scale_factor, [1.0, 1.0])


def test_find_read_rect_params_power(_response_ndpi):
    """Test finding read rect parameters for objective power."""
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)

    location = (0, 0)
    size = (256, 256)
    # Test a range of objective powers
    for target_scale in [1.25, 2.5, 5, 10, 20]:
        (level, _, read_size, post_read_scale, _) = wsi.find_read_rect_params(
            location=location, size=size, resolution=target_scale, units="power",
        )
        assert level >= 0
        assert level < wsi.slide_info.level_count
        # Check that read_size * scale == size
        post_read_downscaled_size = np.round(read_size * post_read_scale).astype(int)
        assert np.array_equal(post_read_downscaled_size, np.array(size))


def test_find_read_rect_params_mpp(_response_ndpi):
    """Test finding read rect parameters for objective mpp."""
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)

    location = (0, 0)
    size = (256, 256)
    # Test a range of MPP
    for target_scale in range(1, 10):
        (level, _, read_size, post_read_scale, _) = wsi.find_read_rect_params(
            location=location, size=size, resolution=target_scale, units="mpp",
        )
        assert level >= 0
        assert level < wsi.slide_info.level_count
        # Check that read_size * scale == size
        post_read_downscaled_size = np.round(read_size * post_read_scale).astype(int)
        assert np.array_equal(post_read_downscaled_size, np.array(size))


def test_read_rect_openslide_baseline(_response_ndpi):
    """Test openslide read rect at baseline.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)
    location = (1000, 2000)
    size = (1000, 1000)
    im_region = wsi.read_rect(location, size, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (1000, 1000, 3)


def test_read_rect_jp2_baseline(_response_jp2):
    """Test jp2 read rect at baseline.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_response_jp2)
    location = (1000, 2000)
    size = (1000, 1000)
    im_region = wsi.read_rect(location, size, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (1000, 1000, 3)


def test_read_rect_openslide_levels(_response_ndpi):
    """Test openslide read rect with resolution in levels.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)
    location = (1000, 2000)
    size = (256, 256)
    for level in range(wsi.slide_info.level_count):
        im_region = wsi.read_rect(location, size, resolution=level, units="level")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert im_region.shape == (256, 256, 3)


def test_read_rect_jp2_levels(_response_jp2):
    """Test jp2 read rect with resolution in levels.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_response_jp2)
    location = (1000, 2000)
    size = (256, 256)
    for level in range(wsi.slide_info.level_count):
        level_width, level_height = wsi.slide_info.level_dimensions[level]
        im_region = wsi.read_rect(location, size, resolution=level, units="level")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert im_region.shape == (min(256, level_height), min(256, level_width), 3)


def test_read_rect_openslide_mpp(_response_ndpi):
    """Test openslide read rect with resolution in microns per pixel.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)
    location = (1000, 2000)
    size = (256, 256)
    for factor in range(1, 10):
        mpp = wsi.slide_info.mpp * factor
        im_region = wsi.read_rect(location, size, resolution=mpp, units="mpp")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert im_region.shape == (256, 256, 3)


def test_read_rect_jp2_mpp(_response_jp2):
    """Test jp2 read rect with resolution in microns per pixel.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_response_jp2)
    location = (1000, 2000)
    size = (256, 256)
    for factor in range(1, 10):
        mpp = wsi.slide_info.mpp * factor
        im_region = wsi.read_rect(location, size, resolution=mpp, units="mpp")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert im_region.shape == (256, 256, 3)


def test_read_rect_openslide_objective_power(_response_ndpi):
    """Test openslide read rect with resolution in objective power.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)
    location = (0, 0)
    size = (256, 256)
    for objective_power in [20, 10, 5, 2.5, 1.25]:
        im_region = wsi.read_rect(
            location, size, resolution=objective_power, units="power"
        )

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert im_region.shape == (256, 256, 3)


def test_read_rect_jp2_objective_power(_response_jp2):
    """Test jp2 read rect with resolution in objective power.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_response_jp2)
    location = (0, 0)
    size = (256, 256)
    for objective_power in [20, 10, 5, 2.5, 1.25]:
        im_region = wsi.read_rect(
            location, size, resolution=objective_power, units="mpp"
        )

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert im_region.shape == (256, 256, 3)


def test_read_bounds_openslide_baseline(_response_ndpi):
    """Test openslide read bounds at baseline.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)
    region = [1000, 2000, 2000, 3000]
    im_region = wsi.read_bounds(
        region[0], region[1], region[2], region[3], resolution=0, units="level"
    )

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (1000, 1000, 3)


def test_read_bounds_jp2_baseline(_response_jp2):
    """Test jp2 read bounds at baseline.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_response_jp2)
    region = [1000, 2000, 2000, 3000]
    im_region = wsi.read_bounds(
        region[0], region[1], region[2], region[3], resolution=0, units="level"
    )

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (1000, 1000, 3)


def test_read_bounds_openslide_levels(_response_ndpi):
    """Test openslide read bounds with resolution in levels.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)
    region = [1000, 2000, 2000, 3000]
    for level, downsample in enumerate(wsi.slide_info.level_downsamples):
        im_region = wsi.read_bounds(
            region[0], region[1], region[2], region[3], resolution=level, units="level"
        )

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        expected_output_shape = tuple(
            np.round([1000 / downsample, 1000 / downsample, 3]).astype(int)
        )
        assert im_region.shape == expected_output_shape


def test_read_bounds_jp2_levels(_response_jp2):
    """Test jp2 read bounds with resolution in levels.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_response_jp2)
    region = [1000, 2000, 2000, 3000]
    for level, downsample in enumerate(wsi.slide_info.level_downsamples):
        im_region = wsi.read_bounds(
            region[0], region[1], region[2], region[3], resolution=level, units="level"
        )

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        expected_output_shape = tuple(np.round([1000 / downsample, 1000 / downsample]))
        assert im_region.shape[:2] == approx(expected_output_shape, abs=1)
        assert im_region.shape[2] == 3


def test_read_bounds_openslide_mpp(_response_ndpi):
    """Test openslide read bounds with resolution in microns per pixel.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)
    region = [1000, 2000, 2000, 3000]
    slide_mpp = wsi.slide_info.mpp
    for factor in range(1, 10):
        mpp = slide_mpp * factor
        downsample = mpp / slide_mpp

        im_region = wsi.read_bounds(
            region[0], region[1], region[2], region[3], resolution=mpp, units="mpp"
        )

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        expected_output_shape = tuple(
            np.round((np.array([1000] * 2) / downsample)).astype(int)
        )
        assert im_region.shape[:2] == expected_output_shape
        assert im_region.shape[2] == 3


def test_read_bounds_jp2_mpp(_response_jp2):
    """Test jp2 read bounds with resolution in microns per pixel.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_response_jp2)
    region = [1000, 2000, 2000, 3000]
    slide_mpp = wsi.slide_info.mpp
    for factor in range(1, 10):
        mpp = slide_mpp * factor
        downsample = mpp / slide_mpp

        im_region = wsi.read_bounds(
            region[0], region[1], region[2], region[3], resolution=mpp, units="mpp"
        )

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        expected_output_shape = tuple(
            np.round((np.array([1000] * 2) / downsample)).astype(int)
        )
        assert im_region.shape[:2] == approx(expected_output_shape, abs=1)
        assert im_region.shape[2] == 3


def test_read_bounds_openslide_objective_power(_response_ndpi):
    """Test openslide read bounds with resolution in objective power.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_response_ndpi)
    region = [1000, 2000, 2000, 3000]
    slide_power = wsi.slide_info.objective_power
    for objective_power in [20, 10, 5, 2.5, 1.25]:
        downsample = slide_power / objective_power

        im_region = wsi.read_bounds(
            region[0],
            region[1],
            region[2],
            region[3],
            resolution=objective_power,
            units="power",
        )

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        expected_output_shape = tuple(
            np.round((np.array([1000] * 2) / downsample)).astype(int)
        )
        assert im_region.shape[:2] == expected_output_shape
        assert im_region.shape[2] == 3


def test_read_bounds_interpolated(_response_svs):
    """Test openslide read bounds with interpolated output.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_response_svs)
    region = [0, 0, 500, 500]
    im_region = wsi.read_bounds(
        region[0], region[1], region[2], region[3], resolution=0.1, units="mpp",
    )

    assert 0.1 < wsi.slide_info.mpp[0]
    assert 0.1 < wsi.slide_info.mpp[1]
    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape[2] == 3
    assert all(np.array(im_region.shape[:2]) > 500)


def test_read_bounds_jp2_objective_power(_response_jp2):
    """Test jp2 read bounds with resolution in objective power.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_response_jp2)
    region = [1000, 2000, 2000, 3000]
    slide_power = wsi.slide_info.objective_power
    for objective_power in [20, 10, 5, 2.5, 1.25]:
        downsample = slide_power / objective_power

        im_region = wsi.read_bounds(
            region[0],
            region[1],
            region[2],
            region[3],
            resolution=objective_power,
            units="power",
        )

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        expected_output_shape = tuple(
            np.round((np.array([1000] * 2) / downsample)).astype(int)
        )
        assert im_region.shape[:2] == approx(expected_output_shape[:2], abs=1)
        assert im_region.shape[2] == 3


def test_wsireader_slide_thumbnail(_response_svs):
    """Test for slide_thumbnail as a python function."""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent), file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    slide_thumbnail = wsi.slide_thumbnail()
    assert isinstance(slide_thumbnail, np.ndarray)
    assert slide_thumbnail.dtype == "uint8"


def test_wsireader_save_tiles(_response_svs, tmp_path):
    """Test for save_tiles in wsireader as a python function."""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent), file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(
        files_all[0],
        output_dir=str(pathlib.Path(tmp_path).joinpath("test_wsireader_save_tiles")),
        tile_objective_value=5,
    )
    wsi.save_tiles(verbose=True)
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


def test_save_tiles(_response_all_wsis, tmp_path):
    """Test for save_tiles as a python function."""
    file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_all_wsis)), file_types=file_types,
    )

    for curr_file in files_all:
        save_tiles(
            input_path=curr_file,
            tile_objective_value=5,
            output_dir=str(pathlib.Path(tmp_path).joinpath("tiles_save_tiles")),
            verbose=True,
        )

    assert (
        pathlib.Path(tmp_path)
        .joinpath("tiles_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("tiles_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("tiles_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("tiles_save_tiles")
        .joinpath("CMU-1.ndpi")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("tiles_save_tiles")
        .joinpath("CMU-1.ndpi")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("tiles_save_tiles")
        .joinpath("CMU-1.ndpi")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )


def test_wsireader_jp2_save_tiles(_response_jp2, tmp_path):
    """Test for save_tiles in wsireader as a python function."""
    wsi = wsireader.OmnyxJP2WSIReader(
        _response_jp2,
        output_dir=str(
            pathlib.Path(tmp_path).joinpath("test_wsireader_jp2_save_tiles")
        ),
        tile_objective_value=5,
    )
    wsi.save_tiles(verbose=True)
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


def test_exception_tests():
    """Test for Exceptions."""
    with pytest.raises(FileNotSupported):
        utils.misc.save_yaml(
            slide_info(input_path="/mnt/test/sample.txt", verbose=True).as_dict(),
            "test.yaml",
        )

    with pytest.raises(FileNotSupported):
        save_tiles(
            input_path="/mnt/test/sample.txt",
            tile_objective_value=5,
            output_dir=str(pathlib.Path(__file__).parent.joinpath("tiles_save_tiles")),
            verbose=True,
        )

    with pytest.raises(MethodNotSupported):
        get_normaliser(method_name="invalid_normaliser")


def test_imresize():
    """Test for imresize."""
    img = np.zeros((2000, 2000, 3))
    resized_img = utils.transforms.imresize(img, 0.5)
    assert resized_img.shape == (1000, 1000, 3)


def test_background_composite():
    """Test for background composite."""
    new_im = np.zeros((2000, 2000, 4)).astype("uint8")
    new_im[:1000, :, 3] = 255
    im = utils.transforms.background_composite(new_im)
    assert np.all(im[1000:, :, :] == 255)
    assert np.all(im[:1000, :, :] == 0)

    im = utils.transforms.background_composite(new_im, alpha=True)
    assert np.all(im[:, :, 3] == 255)


def test_reinhard_normalise():
    """Test for Reinhard colour normalisation."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(file_parent_dir.joinpath("data/source_image.png"))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    reinhard_img = imread(file_parent_dir.joinpath("data/norm_reinhard.png"))

    norm = get_normaliser("reinhard")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(reinhard_img / 255.0 - transform / 255.0)) < 1e-2


def test_custom_normalise():
    """Test for stain normalisation with user-defined stain matrix."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(file_parent_dir.joinpath("data/source_image.png"))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    custom_img = imread(file_parent_dir.joinpath("data/norm_ruifrok.png"))

    # init class with custom method - test with ruifrok stain matrix
    stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
    norm = get_normaliser("custom", stain_matrix=stain_matrix)
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(custom_img / 255.0 - transform / 255.0)) < 1e-2


def test_ruifrok_normalise():
    """Test for stain normalisation with stain matrix from Ruifrok and Johnston."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(file_parent_dir.joinpath("data/source_image.png"))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    ruifrok_img = imread(file_parent_dir.joinpath("data/norm_ruifrok.png"))

    # init class with Ruifrok & Johnston method
    norm = get_normaliser("ruifrok")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(ruifrok_img / 255.0 - transform / 255.0)) < 1e-2


def test_macenko_normalise():
    """Test for stain normalisation with stain matrix from Macenko et al."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(file_parent_dir.joinpath("data/source_image.png"))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    macenko_img = imread(file_parent_dir.joinpath("data/norm_macenko.png"))

    # init class with Macenko method
    norm = get_normaliser("macenko")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(macenko_img / 255.0 - transform / 255.0)) < 1e-2


def test_vahadane_normalise():
    """Test for stain normalisation with stain matrix from Vahadane et al."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(file_parent_dir.joinpath("data/source_image.png"))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    vahadane_img = imread(file_parent_dir.joinpath("data/norm_vahadane.png"))

    # init class with Vahadane method
    norm = get_normaliser("vahadane")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(vahadane_img / 255.0 - transform / 255.0)) < 1e-2


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_help_interface():
    """Test the CLI help."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert help_result.output == result.output


def test_command_line_version():
    """Test for version check."""
    runner = CliRunner()
    version_result = runner.invoke(cli.main, ["-V"])
    assert __version__ in version_result.output


def test_command_line_slide_info(_response_all_wsis):
    """Test the Slide information CLI."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--wsi_input",
            str(pathlib.Path(_response_all_wsis)),
            "--file_types",
            '"*.ndpi, *.svs"',
            "--mode",
            "save",
        ],
    )

    assert slide_info_result.exit_code == 0

    file_types = "*.svs"
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_all_wsis)), file_types=file_types,
    )
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--wsi_input",
            files_all[0],
            "--file_types",
            '"*.ndpi, *.svs"',
            "--mode",
            "show",
        ],
    )

    assert slide_info_result.exit_code == 0


def test_command_line_read_bounds(_response_ndpi, tmp_path):
    """Test OpenSlide read_bounds CLI."""
    runner = CliRunner()
    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--wsi_input",
            str(pathlib.Path(_response_ndpi)),
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
            "--output_path",
            str(pathlib.Path(tmp_path).joinpath("im_region.jpg")),
        ],
    )

    assert read_bounds_result.exit_code == 0
    assert os.path.isfile(str(pathlib.Path(tmp_path).joinpath("im_region.jpg")))

    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--wsi_input",
            str(pathlib.Path(_response_ndpi)),
            "--resolution",
            "0",
            "--units",
            "level",
            "--mode",
            "save",
            "--output_path",
            str(pathlib.Path(tmp_path).joinpath("im_region2.jpg")),
        ],
    )

    assert read_bounds_result.exit_code == 0
    assert os.path.isfile(str(pathlib.Path(tmp_path).joinpath("im_region2.jpg")))


def test_command_line_jp2_read_bounds(_response_jp2, tmp_path):
    """Test JP2 read_bounds."""
    runner = CliRunner()
    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--wsi_input",
            str(pathlib.Path(_response_jp2)),
            "--resolution",
            "0",
            "--units",
            "level",
            "--mode",
            "save",
            "--output_path",
            str(pathlib.Path(tmp_path).joinpath("im_region.jpg")),
        ],
    )

    assert read_bounds_result.exit_code == 0
    assert os.path.isfile(str(pathlib.Path(tmp_path).joinpath("im_region.jpg")))


def test_command_line_slide_thumbnail(_response_ndpi, tmp_path):
    """Test for the slide_thumbnail CLI."""
    runner = CliRunner()
    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--wsi_input",
            str(pathlib.Path(_response_ndpi)),
            "--mode",
            "save",
            "--output_path",
            str(pathlib.Path(tmp_path).joinpath("slide_thumb.jpg")),
        ],
    )

    assert slide_thumb_result.exit_code == 0
    assert pathlib.Path(tmp_path).joinpath("slide_thumb.jpg").is_file()


def test_command_line_jp2_slide_thumbnail(_response_jp2, tmp_path):
    """Test for the jp2 slide_thumbnail CLI."""
    runner = CliRunner()
    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--wsi_input",
            str(pathlib.Path(_response_jp2)),
            "--mode",
            "save",
            "--output_path",
            str(pathlib.Path(tmp_path).joinpath("slide_thumb.jpg")),
        ],
    )

    assert slide_thumb_result.exit_code == 0
    assert pathlib.Path(tmp_path).joinpath("slide_thumb.jpg").is_file()


def test_command_line_save_tiles(_response_all_wsis, tmp_path):
    """Test for save_tiles CLI."""
    runner = CliRunner()
    save_tiles_result = runner.invoke(
        cli.main,
        [
            "save-tiles",
            "--wsi_input",
            str(pathlib.Path(_response_all_wsis)),
            "--file_types",
            '"*.ndpi, *.svs"',
            "--tile_objective_value",
            "5",
            "--output_dir",
            os.path.join(tmp_path, "all_tiles"),
        ],
    )

    assert save_tiles_result.exit_code == 0

    file_types = "*.svs"
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_all_wsis)), file_types=file_types,
    )
    save_svs_tiles_result = runner.invoke(
        cli.main,
        [
            "save-tiles",
            "--wsi_input",
            files_all[0],
            "--file_types",
            '"*.ndpi, *.svs"',
            "--tile_objective_value",
            "5",
            "--output_dir",
            tmp_path,
        ],
    )

    assert save_svs_tiles_result.exit_code == 0


def test_wsimeta_init_fail():
    """Test incorrect init for WSIMeta raises TypeError."""
    with pytest.raises(TypeError):
        wsimeta.WSIMeta(slide_dimensions=None)


@pytest.mark.filterwarnings("ignore")
def test_wsimeta_validate_fail():
    """Test failure cases for WSIMeta validation."""
    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512), level_dimensions=[])
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(
        slide_dimensions=(512, 512),
        level_dimensions=[(512, 512), (256, 256)],
        level_count=3,
    )
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512), level_downsamples=[1, 2],)
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512), level_downsamples=[1, 2],)
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512))
    meta.level_dimensions = None
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512))
    meta.level_downsamples = None
    assert meta.validate() is False


def test_command_line_stainnorm():
    """Test for the stain normalisation CLI."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = file_parent_dir.joinpath("data/source_image.png")
    target_img = file_parent_dir.joinpath("../data/target_image.png")
    runner = CliRunner()
    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stainnorm",
            "--source_input",
            source_img,
            "--target_input",
            target_img,
            "--method",
            "reinhard",
        ],
    )

    assert stainnorm_result.exit_code == 0

    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stainnorm",
            "--source_input",
            source_img,
            "--target_input",
            target_img,
            "--method",
            "ruifrok",
        ],
    )

    assert stainnorm_result.exit_code == 0

    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stainnorm",
            "--source_input",
            source_img,
            "--target_input",
            target_img,
            "--method",
            "macenko",
        ],
    )

    assert stainnorm_result.exit_code == 0

    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stainnorm",
            "--source_input",
            source_img,
            "--target_input",
            target_img,
            "--method",
            "vahadane",
        ],
    )

    assert stainnorm_result.exit_code == 0


@pytest.mark.filterwarnings("ignore")
def test_wsimeta_validate_pass():
    """Test WSIMeta validation."""
    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512))
    assert meta.validate()

    meta = wsimeta.WSIMeta(
        slide_dimensions=(512, 512),
        level_dimensions=[(512, 512), (256, 256)],
        level_downsamples=[1, 2],
    )


def test_wsimeta_openslidewsireader_ndpi(_response_ndpi, tmp_path):
    """Test OpenSlide reader metadata for ndpi."""
    wsi_obj = wsireader.OpenSlideWSIReader(_response_ndpi)
    meta = wsi_obj.slide_info
    assert meta.validate()


def test_wsimeta_openslidewsireader_svs(_response_svs, tmp_path):
    """Test OpenSlide reader metadata for svs."""
    wsi_obj = wsireader.OpenSlideWSIReader(_response_svs)
    meta = wsi_obj.slide_info
    assert meta.validate()
