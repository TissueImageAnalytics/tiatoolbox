import numpy as np
import pytest

from tiatoolbox.dataloader import wsimeta, wsireader


# noinspection PyTypeChecker
def test_wsimeta_init_fail():
    """Test incorrect init for WSIMeta raises TypeError."""
    with pytest.raises(TypeError):
        wsimeta.WSIMeta(slide_dimensions=(None, None))


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

    meta = wsimeta.WSIMeta(
        slide_dimensions=(512, 512),
        level_downsamples=[1, 2],
    )
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(
        slide_dimensions=(512, 512),
        level_downsamples=[1, 2],
    )
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512))
    meta.level_dimensions = None
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512))
    meta.level_downsamples = None
    assert meta.validate() is False


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

    assert meta.validate()


def test_wsimeta_openslidewsireader_ndpi(_sample_ndpi, tmp_path):
    """Test OpenSlide reader metadata for ndpi."""
    wsi_obj = wsireader.OpenSlideWSIReader(_sample_ndpi)
    meta = wsi_obj.info
    assert meta.validate()


def test_wsimeta_openslidewsireader_svs(_sample_svs, tmp_path):
    """Test OpenSlide reader metadata for svs."""
    wsi_obj = wsireader.OpenSlideWSIReader(_sample_svs)
    meta = wsi_obj.info
    assert meta.validate()


def test_wsimeta_setter(_sample_svs):
    """Test setter for metadata."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    meta = wsi.info
    assert not np.array_equal(meta.mpp, np.array([1, 1]))
    meta.mpp = np.array([1, 1])
    wsi.info = meta
    assert meta.validate()
    assert np.array_equal(wsi.info.mpp, np.array([1, 1]))
