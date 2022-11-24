"""Tests for obtaining whole-slide image metadata."""

import numpy as np
import pytest

from tiatoolbox.wsicore import wsimeta, wsireader


# noinspection PyTypeChecker
def test_wsimeta_init_fail():
    """Test incorrect init for WSIMeta raises TypeError."""
    with pytest.raises(TypeError):
        wsimeta.WSIMeta(slide_dimensions=(None, None), axes="YXS")


@pytest.mark.filterwarnings("ignore")
def test_wsimeta_validate_fail():
    """Test failure cases for WSIMeta validation."""
    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512), axes="YXS", level_dimensions=[])
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(
        slide_dimensions=(512, 512),
        axes="YXS",
        level_dimensions=[(512, 512), (256, 256)],
        level_count=3,
    )
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(
        slide_dimensions=(512, 512),
        axes="YXS",
        level_downsamples=[1, 2],
    )
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(
        slide_dimensions=(512, 512),
        axes="YXS",
        level_downsamples=[1, 2],
    )
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512), axes="YXS")
    meta.level_dimensions = None
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512), axes="YXS")
    meta.level_downsamples = None
    assert meta.validate() is False


@pytest.mark.filterwarnings("ignore")
def test_wsimeta_validate_invalid_axes():
    """Test failure cases for WSIMeta validation with invalid axes."""
    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512), axes="YXSF")
    assert meta.validate() is False


@pytest.mark.filterwarnings("ignore")
def test_wsimeta_validate_pass():
    """Test WSIMeta validation."""
    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512), axes="YXS")
    assert meta.validate()

    meta = wsimeta.WSIMeta(
        slide_dimensions=(512, 512),
        axes="YXS",
        level_dimensions=[(512, 512), (256, 256)],
        level_downsamples=[1, 2],
    )

    assert meta.validate()


def test_wsimeta_openslidewsireader_ndpi(sample_ndpi):
    """Test OpenSlide reader metadata for ndpi."""
    wsi_obj = wsireader.OpenSlideWSIReader(sample_ndpi)
    meta = wsi_obj.info
    assert meta.validate()


def test_wsimeta_openslidewsireader_svs(sample_svs):
    """Test OpenSlide reader metadata for svs."""
    wsi_obj = wsireader.OpenSlideWSIReader(sample_svs)
    meta = wsi_obj.info
    assert meta.validate()

    meta.mpp = None
    m = meta.as_dict()
    assert isinstance(m["mpp"], tuple)


def test_wsimeta_setter(sample_svs):
    """Test setter for metadata."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    meta = wsi.info
    assert not np.array_equal(meta.mpp, np.array([1, 1]))
    meta.mpp = np.array([1, 1])
    wsi.info = meta
    assert meta.validate()
    assert np.array_equal(wsi.info.mpp, np.array([1, 1]))
