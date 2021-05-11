from tiatoolbox.wsicore import wsireader
from tiatoolbox import utils
from tiatoolbox import cli
from tiatoolbox.utils.exceptions import FileNotSupported

import pytest
from pytest import approx
import pathlib
import numpy as np
import cv2
from click.testing import CliRunner
from skimage.filters import threshold_otsu
from skimage.morphology import disk, binary_dilation, remove_small_objects

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
            assert im_region.shape[:2] == approx(expected_output_shape, abs=1)
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
            assert im_region.shape[:2] == approx(expected_output_shape[:2], abs=1)
        else:
            assert im_region.shape[:2] == expected_output_shape
        assert im_region.shape[2] == 3


def read_bounds_level_consistency(wsi, bounds):
    """Read bounds level consistency helper."""
    from skimage.registration import phase_cross_correlation

    imgs = [wsi.read_bounds(bounds, power, "power") for power in [60, 40, 20, 10, 5]]
    smallest_size = imgs[-1].shape[:2][::-1]
    resized = [cv2.resize(img, smallest_size) for img in imgs]
    blurred = [cv2.GaussianBlur(img, (5, 5), cv2.BORDER_REFLECT) for img in resized]
    as_float = [img.astype(np.float) for img in blurred]

    # Pair-wise check resolutions for mean squared error
    for i, a in enumerate(as_float):
        for b in as_float[i + 1 :]:
            _, error, phase_diff = phase_cross_correlation(a, b)
            assert phase_diff < 0.1
            assert error < 0.1


def command_line_slide_thumbnail(runner, sample, tmp_path, mode="save"):
    """Command line slide thumbnail helper."""
    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--wsi_input",
            str(pathlib.Path(sample)),
            "--mode",
            mode,
            "--output_path",
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


def test_wsireader_slide_info(_sample_svs, tmp_path):
    """Test for slide_info in WSIReader class as a python function."""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_sample_svs).parent),
        file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    slide_param = wsi.info
    out_path = tmp_path / slide_param.file_path.with_suffix(".yaml").name
    utils.misc.save_yaml(slide_param.as_dict(), out_path)


def test_wsireader_slide_info_cache(_sample_svs):
    """Test for caching slide_info in WSIReader class as a python function."""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_sample_svs).parent),
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


def test__relative_level_scales_openslide_baseline(_sample_ndpi):
    """Test openslide relative level scales for pixels per baseline pixel."""
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
    relative_level_scales_baseline(wsi)


def test__relative_level_scales_jp2_baseline(_sample_jp2):
    """Test jp2 relative level scales for pixels per baseline pixel."""
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
    relative_level_scales_baseline(wsi)


def test__relative_level_scales_openslide_mpp(_sample_ndpi):
    """Test openslide calculation of relative level scales for mpp."""
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
    level_scales = wsi._relative_level_scales(0.5, "mpp")
    level_scales = np.array(level_scales)
    assert strictly_increasing(level_scales[:, 0])
    assert strictly_increasing(level_scales[:, 1])
    assert all(level_scales[0] == wsi.info.mpp / 0.5)


def test__relative_level_scales_jp2_mpp(_sample_jp2):
    """Test jp2 calculation of relative level scales for mpp."""
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
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


def test__relative_level_scales_openslide_power(_sample_ndpi):
    """Test openslide calculation of relative level scales for objective power."""
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
    relative_level_scales_power(wsi)


def test__relative_level_scales_jp2_power(_sample_jp2):
    """Test jp2 calculation of relative level scales for objective power."""
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
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


def test__relative_level_scales_openslide_level(_sample_ndpi):
    """Test openslide calculation of relative level scales for level."""
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
    relative_level_scales_level(wsi)


def test__relative_level_scales_jp2_level(_sample_jp2):
    """Test jp2 calculation of relative level scales for level."""
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
    relative_level_scales_level(wsi)


def relative_level_scales_float(wsi):
    """Calculation of relative level scales for fractional level."""
    level_scales = wsi._relative_level_scales(1.5, "level")
    level_scales = np.array(level_scales)
    assert level_scales[0] == approx([1 / 3, 1 / 3])
    downsamples = np.array(wsi.info.level_downsamples)
    expected = downsamples / downsamples[0] * (1 / 3)
    assert np.array_equal(level_scales[:, 0], level_scales[:, 1])
    assert np.array_equal(level_scales[:, 0], expected)


def test__relative_level_scales_openslide_level_float(_sample_ndpi):
    """Test openslide calculation of relative level scales for fractional level."""
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
    relative_level_scales_float(wsi)


def test__relative_level_scales_jp2_level_float(_sample_jp2):
    """Test jp2 calculation of relative level scales for fractional level."""
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
    relative_level_scales_float(wsi)


def test__relative_level_scales_invalid_units(_sample_svs):
    """Test _relative_level_scales with invalid units."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    with pytest.raises(ValueError):
        wsi._relative_level_scales(1.0, "gibberish")


def test__relative_level_scales_no_mpp():
    """Test _relative_level_scales objective when mpp is None."""

    class DummyWSI:
        """Mock WSIReader for testing."""

        _relative_level_scales = wsireader.WSIReader._relative_level_scales

        @property
        def info(self):
            return wsireader.WSIMeta((100, 100))

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
            return wsireader.WSIMeta((100, 100))

    wsi = DummyWSI()
    with pytest.raises(ValueError):
        wsi._relative_level_scales(10, "power")


def test__relative_level_scales_level_too_high(_sample_svs):
    """Test _relative_level_scales levels set too high."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    with pytest.raises(ValueError):
        wsi._relative_level_scales(100, "level")


def test_find_optimal_level_and_downsample_openslide_interpolation_warning(
    _sample_ndpi,
):
    """Test finding optimal level for mpp read with scale > 1.

    This tests the case where the scale is found to be > 1 and interpolation
    will be applied to the output. A UserWarning should be raised in this case.
    """
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
    with pytest.warns(UserWarning):
        _, _ = wsi._find_optimal_level_and_downsample(0.1, "mpp")


def test_find_optimal_level_and_downsample_jp2_interpolation_warning(_sample_jp2):
    """Test finding optimal level for mpp read with scale > 1.

    This tests the case where the scale is found to be > 1 and interpolation
    will be applied to the output. A UserWarning should be raised in this case.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
    with pytest.warns(UserWarning):
        _, _ = wsi._find_optimal_level_and_downsample(0.1, "mpp")


def test_find_optimal_level_and_downsample_mpp(_sample_ndpi):
    """Test finding optimal level for mpp read."""
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)

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
        assert post_read_scale_factor == approx(expected_scale)


def test_find_optimal_level_and_downsample_power(_sample_ndpi):
    """Test finding optimal level for objective power read."""
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)

    objective_powers = [20, 10, 5, 2.5, 1.25]
    expected_levels = [0, 1, 2, 3, 4]
    for objective_power, expected_level in zip(objective_powers, expected_levels):
        read_level, post_read_scale_factor = wsi._find_optimal_level_and_downsample(
            objective_power, "power"
        )

        assert read_level == expected_level
        assert np.array_equal(post_read_scale_factor, [1.0, 1.0])


def test_find_optimal_level_and_downsample_level(_sample_ndpi):
    """Test finding optimal level for level read.

    For integer levels, the returned level should always be the same as
    the input level.
    """
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)

    for level in range(wsi.info.level_count):
        read_level, post_read_scale_factor = wsi._find_optimal_level_and_downsample(
            level, "level"
        )

        assert read_level == level
        assert np.array_equal(post_read_scale_factor, [1.0, 1.0])


def test_find_read_rect_params_power(_sample_ndpi):
    """Test finding read rect parameters for objective power."""
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)

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


def test_find_read_rect_params_mpp(_sample_ndpi):
    """Test finding read rect parameters for objective mpp."""
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)

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


def test_read_rect_openslide_baseline(_sample_ndpi):
    """Test openslide read rect at baseline.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
    location = NDPI_TEST_TISSUE_LOCATION
    size = NDPI_TEST_TISSUE_SIZE
    im_region = wsi.read_rect(location, size, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)


def test_read_rect_jp2_baseline(_sample_jp2):
    """Test jp2 read rect at baseline.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
    location = JP2_TEST_TISSUE_LOCATION
    size = JP2_TEST_TISSUE_SIZE
    im_region = wsi.read_rect(location, size, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)


def test_read_rect_openslide_levels(_sample_ndpi):
    """Test openslide read rect with resolution in levels.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
    location = NDPI_TEST_TISSUE_LOCATION
    size = NDPI_TEST_TISSUE_SIZE
    for level in range(wsi.info.level_count):
        im_region = wsi.read_rect(location, size, resolution=level, units="level")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert im_region.shape == (*size[::-1], 3)


def test_read_rect_jp2_levels(_sample_jp2):
    """Test jp2 read rect with resolution in levels.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
    location = (0, 0)
    size = JP2_TEST_TISSUE_SIZE
    width, height = size
    for level in range(wsi.info.level_count):
        level_width, level_height = wsi.info.level_dimensions[level]
        im_region = wsi.read_rect(location, size, resolution=level, units="level")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        assert approx(
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


def test_read_rect_openslide_mpp(_sample_ndpi):
    """Test openslide read rect with resolution in microns per pixel.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
    location = NDPI_TEST_TISSUE_LOCATION
    size = NDPI_TEST_TISSUE_SIZE
    read_rect_mpp(wsi, location, size)


def test_read_rect_jp2_mpp(_sample_jp2):
    """Test jp2 read rect with resolution in microns per pixel.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
    location = JP2_TEST_TISSUE_LOCATION
    size = JP2_TEST_TISSUE_SIZE
    read_rect_mpp(wsi, location, size)


def test_read_rect_openslide_objective_power(_sample_ndpi):
    """Test openslide read rect with resolution in objective power.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
    location = NDPI_TEST_TISSUE_LOCATION
    size = NDPI_TEST_TISSUE_SIZE

    read_rect_objective_power(wsi, location, size)


def test_read_rect_jp2_objective_power(_sample_jp2):
    """Test jp2 read rect with resolution in objective power.

    Location coordinate is in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
    location = JP2_TEST_TISSUE_LOCATION
    size = JP2_TEST_TISSUE_SIZE

    read_rect_objective_power(wsi, location, size)


def test_read_bounds_openslide_baseline(_sample_ndpi):
    """Test openslide read bounds at baseline.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
    bounds = NDPI_TEST_TISSUE_BOUNDS
    size = NDPI_TEST_TISSUE_SIZE
    im_region = wsi.read_bounds(bounds, resolution=0, units="level")

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (*size[::-1], 3)


def test_read_bounds_jp2_baseline(_sample_jp2):
    """Test jp2 read bounds at baseline.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
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


def test_read_bounds_openslide_levels(_sample_ndpi):
    """Test openslide read bounds with resolution in levels.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
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


def test_read_bounds_jp2_levels(_sample_jp2):
    """Test jp2 read bounds with resolution in levels.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
    bounds = JP2_TEST_TISSUE_BOUNDS
    width, height = JP2_TEST_TISSUE_SIZE
    for level, downsample in enumerate(wsi.info.level_downsamples):
        im_region = wsi.read_bounds(bounds, resolution=level, units="level")

        assert isinstance(im_region, np.ndarray)
        assert im_region.dtype == "uint8"
        expected_output_shape = tuple(
            np.round([height / downsample, width / downsample])
        )
        assert im_region.shape[:2] == approx(expected_output_shape, abs=1)
        assert im_region.shape[2] == 3


def test_read_bounds_openslide_mpp(_sample_ndpi):
    """Test openslide read bounds with resolution in microns per pixel.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
    bounds = NDPI_TEST_TISSUE_BOUNDS
    size = NDPI_TEST_TISSUE_SIZE

    read_bounds_mpp(wsi, bounds, size)


def test_read_bounds_jp2_mpp(_sample_jp2):
    """Test jp2 read bounds with resolution in microns per pixel.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
    bounds = JP2_TEST_TISSUE_BOUNDS
    size = JP2_TEST_TISSUE_SIZE

    read_bounds_mpp(wsi, bounds, size, jp2=True)


def test_read_bounds_openslide_objective_power(_sample_ndpi):
    """Test openslide read bounds with resolution in objective power.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
    bounds = NDPI_TEST_TISSUE_BOUNDS
    size = NDPI_TEST_TISSUE_SIZE
    slide_power = wsi.info.objective_power

    read_bounds_objective_power(wsi, slide_power, bounds, size)


def test_read_bounds_jp2_objective_power(_sample_jp2):
    """Test jp2 read bounds with resolution in objective power.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
    bounds = JP2_TEST_TISSUE_BOUNDS
    size = JP2_TEST_TISSUE_SIZE
    slide_power = wsi.info.objective_power

    read_bounds_objective_power(wsi, slide_power, bounds, size, jp2=True)


def test_read_bounds_interpolated(_sample_svs):
    """Test openslide read bounds with interpolated output.

    Coordinates in baseline (level 0) reference frame.
    """
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
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


def test_read_bounds_level_consistency_openslide(_sample_ndpi):
    """Test read_bounds produces the same visual field across resolution levels."""
    wsi = wsireader.OpenSlideWSIReader(_sample_ndpi)
    bounds = NDPI_TEST_TISSUE_BOUNDS

    read_bounds_level_consistency(wsi, bounds)


def test_read_bounds_level_consistency_jp2(_sample_jp2):
    """Test read_bounds produces the same visual field across resolution levels."""
    bounds = JP2_TEST_TISSUE_BOUNDS
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)

    read_bounds_level_consistency(wsi, bounds)


def test_wsireader_get_thumbnail_openslide(_sample_svs):
    """Test for get_thumbnail as a python function."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    slide_thumbnail = wsi.slide_thumbnail()
    assert isinstance(slide_thumbnail, np.ndarray)
    assert slide_thumbnail.dtype == "uint8"


def test_wsireader_get_thumbnail_jp2(_sample_jp2):
    """Test for get_thumbnail as a python function."""
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
    slide_thumbnail = wsi.slide_thumbnail()
    assert isinstance(slide_thumbnail, np.ndarray)
    assert slide_thumbnail.dtype == "uint8"


def test_wsireader_save_tiles(_sample_svs, tmp_path):
    """Test for save_tiles in wsireader as a python function."""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_sample_svs).parent),
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


def test_incompatible_objective_value(_sample_svs, tmp_path):
    """Test for incompatible objective value."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    with pytest.raises(ValueError):
        wsi.save_tiles(
            output_dir=str(
                pathlib.Path(tmp_path).joinpath("test_wsireader_save_tiles")
            ),
            tile_objective_value=3,
            tile_read_size=(5000, 5000),
            verbose=True,
        )


def test_incompatible_level(_sample_svs, tmp_path):
    """Test for incompatible objective value."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    with pytest.warns(UserWarning):
        wsi.save_tiles(
            output_dir=str(
                pathlib.Path(tmp_path).joinpath("test_wsireader_save_tiles2")
            ),
            tile_objective_value=1,
            tile_read_size=(500, 500),
            verbose=True,
        )


def test_wsireader_jp2_save_tiles(_sample_jp2, tmp_path):
    """Test for save_tiles in wsireader as a python function."""
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
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


def test_openslide_objective_power_from_mpp(_sample_svs):
    """Test OpenSlideWSIReader approximation of objective power from mpp."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    wsi.openslide_wsi = DummyMutableOpenSlideObject(wsi.openslide_wsi)
    props = wsi.openslide_wsi._properties

    del props["openslide.objective-power"]  # skipcq: PTC-W0043
    with pytest.warns(UserWarning, match=r"Objective power inferred"):
        _ = wsi.info

    del props["openslide.mpp-x"]  # skipcq: PTC-W0043
    del props["openslide.mpp-y"]  # skipcq: PTC-W0043
    with pytest.warns(UserWarning, match=r"Unable to determine objective power"):
        _ = wsi._info()


def test_openslide_mpp_from_tiff_resolution(_sample_svs):
    """Test OpenSlideWSIReader mpp from TIFF resolution tags."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
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


def test_VirtualWSIReader(_source_image):
    """Test VirtualWSIReader"""
    wsi = wsireader.VirtualWSIReader(pathlib.Path(_source_image))
    with pytest.warns(UserWarning, match=r"Unknown scale"):
        _ = wsi._info()
    with pytest.warns(UserWarning, match=r"Raw data is None"):
        _ = wsi._info()

    assert wsi.img.shape == (256, 256, 3)

    img = wsi.read_rect(location=(0, 0), size=(100, 50))
    assert img.shape == (50, 100, 3)

    img = wsi.read_region(location=(0, 0), size=(100, 50), level=0)
    assert img.shape == (50, 100, 3)


def test_VirtualWSIReader_invalid_mode(_source_image):
    """Test creating a VritualWSIReader with an invalid mode."""
    with pytest.raises(ValueError):
        wsireader.VirtualWSIReader(pathlib.Path(_source_image), mode="foo")


def test_VirtualWSIReader_read_bounds(_source_image):
    """Test VirtualWSIReader read bounds"""
    wsi = wsireader.VirtualWSIReader(pathlib.Path(_source_image))
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


def test_VirtualWSIReader_read_rect(_source_image):
    """Test VirtualWSIReader read rect."""
    wsi = wsireader.VirtualWSIReader(pathlib.Path(_source_image))
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

    wsi = wsireader.VirtualWSIReader(pathlib.Path(_source_image), info=info)

    assert info.as_dict() == wsi.info.as_dict()


def test_VirtualWSIReader_read_bounds_virtual_baseline(_source_image):
    """Test VirtualWSIReader read bounds with virtual baseline."""
    image_path = pathlib.Path(_source_image)
    img_array = utils.misc.imread(image_path)
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(slide_dimensions=double_size)
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


def test_VirtualWSIReader_read_rect_virtual_baseline():
    """Test VirtualWSIReader read rect with virtual baseline.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image.
    """
    file_parent_dir = pathlib.Path(__file__).parent
    image_path = file_parent_dir.joinpath("data/source_image.png")
    img_array = utils.misc.imread(image_path)
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(slide_dimensions=double_size)
    wsi = wsireader.VirtualWSIReader(image_path, info=meta)
    region = wsi.read_rect(location=(0, 0), size=(50, 100))
    target = cv2.resize(
        img_array[:50, :25, :], (50, 100), interpolation=cv2.INTER_CUBIC
    )
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2


def test_VirtualWSIReader_read_rect_virtual_levels():
    """Test VirtualWSIReader read rect with vritual levels.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image and the pyramid/resolution levels.

    Checks that the regions read at each level line up with expected values.
    """
    file_parent_dir = pathlib.Path(__file__).parent
    image_path = file_parent_dir.joinpath("data/source_image.png")
    img_array = utils.misc.imread(image_path)
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(slide_dimensions=double_size, level_downsamples=[1, 2, 4])
    wsi = wsireader.VirtualWSIReader(image_path, info=meta)
    region = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=1, units="level")
    target = img_array[:100, :50, :]
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2

    region = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=2, units="level")
    target = cv2.resize(img_array[:200, :100, :], (50, 100))
    assert np.abs(np.median(region.astype(int) - target.astype(int))) == 0
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2


def test_VirtualWSIReader_read_bounds_virtual_levels():
    """Test VirtualWSIReader read bounds with vritual levels.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image and the pyramid/resolution levels.

    Checks that the regions read at each level line up with expected values.
    """
    file_parent_dir = pathlib.Path(__file__).parent
    image_path = file_parent_dir.joinpath("data/source_image.png")
    img_array = utils.misc.imread(image_path)
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(slide_dimensions=double_size, level_downsamples=[1, 2, 4])
    wsi = wsireader.VirtualWSIReader(image_path, info=meta)
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


def test_VirtualWSIReader_read_rect_virtual_levels_mpp():
    """Test VirtualWSIReader read rect with vritual levels and MPP.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image and the pyramid/resolution levels and
    a baseline MPP specified.

    Checks that the regions read with specified MPP for each level lines up
    with expected values.
    """
    file_parent_dir = pathlib.Path(__file__).parent
    image_path = file_parent_dir.joinpath("data/source_image.png")
    img_array = utils.misc.imread(image_path)
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(
        slide_dimensions=double_size, level_downsamples=[1, 2, 4], mpp=(0.25, 0.25)
    )
    wsi = wsireader.VirtualWSIReader(image_path, info=meta)
    region = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=0.5, units="mpp")
    target = img_array[:100, :50, :]
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2

    region = wsi.read_rect(location=(0, 0), size=(50, 100), resolution=1, units="mpp")
    target = cv2.resize(
        img_array[:200, :100, :], (50, 100), interpolation=cv2.INTER_CUBIC
    )
    assert np.abs(np.mean(region.astype(int) - target.astype(int))) < 0.2


def test_VirtualWSIReader_read_bounds_virtual_levels_mpp():
    """Test VirtualWSIReader read bounds with vritual levels and MPP.

    Creates a virtual slide with a virtualbaseline size which is twice
    as large as the input image and the pyramid/resolution levels.

    Checks that the regions read at each level line up with expected values.
    """
    file_parent_dir = pathlib.Path(__file__).parent
    image_path = file_parent_dir.joinpath("data/source_image.png")
    img_array = utils.misc.imread(image_path)
    img_size = np.array(img_array.shape[:2][::-1])
    double_size = tuple((img_size * 2).astype(int))
    meta = wsireader.WSIMeta(
        slide_dimensions=double_size, level_downsamples=[1, 2, 4], mpp=(0.25, 0.25)
    )
    wsi = wsireader.VirtualWSIReader(image_path, info=meta)
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


def test_tissue_mask_otsu(_sample_svs):
    """Test wsi.tissue_mask with Otsu's method."""

    wsi = wsireader.OpenSlideWSIReader(_sample_svs)

    tissue_thumb = wsi.slide_thumbnail()
    grey_thumb = cv2.cvtColor(tissue_thumb, cv2.COLOR_RGB2GRAY)

    otsu_threhold = threshold_otsu(grey_thumb)
    otsu_mask = grey_thumb < otsu_threhold

    mask = wsi.tissue_mask(method="otsu")
    mask_thumb = mask.slide_thumbnail()

    assert np.mean(np.logical_xor(mask_thumb, otsu_mask)) < 0.05


def test_tissue_mask_morphological(_sample_svs):
    """Test wsi.tissue_mask with morphological method."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
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


def test_tissue_mask_morphological_levels(_sample_svs):
    """Test wsi.tissue_mask with morphological method and resolution in level."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    thumb = wsi.slide_thumbnail(0, "level")
    grey_thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    threshold = threshold_otsu(grey_thumb)
    reference = grey_thumb < threshold
    # Using kernel_size of 1
    mask = wsi.tissue_mask("morphological", 0, "level")
    mask_thumb = mask.slide_thumbnail(0, "level")
    assert np.mean(mask_thumb == reference) > 0.99
    # Custom kernel_size (should still be close to refernce)
    reference = binary_dilation(reference, disk(3))
    mask = wsi.tissue_mask("morphological", 0, "level", kernel_size=3)
    mask_thumb = mask.slide_thumbnail(0, "level")
    assert np.mean(mask_thumb == reference) > 0.95


def test_tissue_mask_read_bounds_none_interpolation(_sample_svs):
    """Test reading a mask using read_bounds with no interpolation."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    mask = wsi.tissue_mask("otsu")
    mask_region = mask.read_bounds((0, 0, 512, 512), interpolation="none")
    assert mask_region.shape[0] == 32
    assert mask_region.shape[1] == 33


def test_tissue_mask_read_rect_none_interpolation(_sample_svs):
    """Test reading a mask using read_rect with no interpolation."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    mask = wsi.tissue_mask("otsu")
    mask_region = mask.read_rect((0, 0), (512, 512), interpolation="none")
    assert mask_region.shape[0] == 32
    assert mask_region.shape[1] == 33


def test_invalid_masker_method(_sample_svs):
    """Test that an invalid masker method string raises a ValueError."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    with pytest.raises(ValueError):
        wsi.tissue_mask(method="foo")


def test_get_wsireader(_sample_svs, _sample_ndpi, _sample_jp2, _source_image):
    """Test get_wsireader to return correct object."""
    _sample_svs = str(_sample_svs)
    _sample_ndpi = str(_sample_ndpi)
    _sample_jp2 = str(_sample_jp2)

    with pytest.raises(FileNotSupported):
        _ = wsireader.get_wsireader("./sample.csv")

    with pytest.raises(TypeError):
        _ = wsireader.get_wsireader([1, 2])

    wsi = wsireader.get_wsireader(_sample_svs)
    assert isinstance(wsi, wsireader.OpenSlideWSIReader)

    wsi = wsireader.get_wsireader(_sample_ndpi)
    assert isinstance(wsi, wsireader.OpenSlideWSIReader)

    wsi = wsireader.get_wsireader(_sample_jp2)
    assert isinstance(wsi, wsireader.OmnyxJP2WSIReader)

    wsi = wsireader.get_wsireader(pathlib.Path(_source_image))
    assert isinstance(wsi, wsireader.VirtualWSIReader)

    img = utils.misc.imread(str(pathlib.Path(_source_image)))
    wsi = wsireader.get_wsireader(input_img=img)
    assert isinstance(wsi, wsireader.VirtualWSIReader)

    # test if get_wsireader can accept a wsireader instance
    wsi_type = type(wsi)
    wsi_out = wsireader.get_wsireader(input_img=wsi)
    assert isinstance(wsi_out, wsi_type)


def test_jp2_missing_cod(_sample_jp2):
    """Test for warning if JP2 is missing COD segment."""
    wsi = wsireader.OmnyxJP2WSIReader(_sample_jp2)
    wsi.glymur_wsi.codestream.segment = []
    with pytest.warns(UserWarning, match="missing COD"):
        _ = wsi.info


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_read_bounds(_sample_ndpi, tmp_path):
    """Test OpenSlide read_bounds CLI."""
    runner = CliRunner()
    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--wsi_input",
            str(pathlib.Path(_sample_ndpi)),
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
    assert pathlib.Path(tmp_path).joinpath("im_region.jpg").is_file()

    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--wsi_input",
            str(pathlib.Path(_sample_ndpi)),
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
    assert pathlib.Path(tmp_path).joinpath("im_region2.jpg").is_file()


def test_command_line_jp2_read_bounds(_sample_jp2, tmp_path):
    """Test JP2 read_bounds."""
    runner = CliRunner()
    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--wsi_input",
            str(pathlib.Path(_sample_jp2)),
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


def test_command_line_jp2_read_bounds_show(_sample_jp2, tmp_path):
    """Test JP2 read_bounds with mode as 'show'."""
    runner = CliRunner()
    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--wsi_input",
            str(pathlib.Path(_sample_jp2)),
            "--resolution",
            "0",
            "--units",
            "level",
            "--mode",
            "show",
        ],
    )

    assert read_bounds_result.exit_code == 0


def test_command_line_unsupported_file_read_bounds(_sample_svs, tmp_path):
    """Test unsupported file read bounds."""
    runner = CliRunner()
    read_bounds_result = runner.invoke(
        cli.main,
        [
            "read-bounds",
            "--wsi_input",
            str(pathlib.Path(_sample_svs))[:-1],
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


def test_command_line_slide_thumbnail(_sample_ndpi, tmp_path):
    """Test for the slide_thumbnail CLI."""
    runner = CliRunner()

    command_line_slide_thumbnail(runner, sample=_sample_ndpi, tmp_path=tmp_path)


def test_command_line_slide_thumbnail_output_none(_sample_svs, tmp_path):
    """Test cli slide thumbnail with output dir None."""
    runner = CliRunner()
    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--wsi_input",
            str(pathlib.Path(_sample_svs)),
            "--mode",
            "save",
        ],
    )

    assert slide_thumb_result.exit_code == 0
    assert pathlib.Path(tmp_path).joinpath("../slide_thumb.jpg").is_file()


def test_command_line_jp2_slide_thumbnail(_sample_jp2, tmp_path):
    """Test for the jp2 slide_thumbnail CLI."""
    runner = CliRunner()

    command_line_slide_thumbnail(runner, sample=_sample_jp2, tmp_path=tmp_path)


def test_command_line_jp2_slide_thumbnail_mode_show(_sample_jp2, tmp_path):
    """Test for the jp2 slide_thumbnail CLI mode='show'."""
    runner = CliRunner()

    command_line_slide_thumbnail(
        runner, sample=_sample_jp2, tmp_path=tmp_path, mode="show"
    )


def test_command_line_jp2_slide_thumbnail_file_not_supported(_sample_jp2, tmp_path):
    """Test for the jp2 slide_thumbnail CLI."""
    runner = CliRunner()

    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--wsi_input",
            str(pathlib.Path(_sample_jp2))[:-1],
            "--mode",
            "save",
            "--output_path",
            str(pathlib.Path(tmp_path).joinpath("slide_thumb.jpg")),
        ],
    )

    assert slide_thumb_result.output == ""
    assert slide_thumb_result.exit_code == 1
    assert isinstance(slide_thumb_result.exception, FileNotSupported)


def test_openslide_read_rect_edge_reflect_padding(_sample_svs):
    """Test openslide edge reflect padding for read_rect."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    region = wsi.read_rect((-64, -64), (128, 128), pad_mode="reflect")
    assert 0 not in region.min(axis=-1)


def test_openslide_read_bounds_edge_reflect_padding(_sample_svs):
    """Test openslide edge reflect padding for read_bounds."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    region = wsi.read_bounds((-64, -64, 64, 64), pad_mode="reflect")
    assert 0 not in region.min(axis=-1)
