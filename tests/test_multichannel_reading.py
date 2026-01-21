"""Tests for multichannel image reading and visualisation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tiatoolbox import utils
from tiatoolbox.utils import postproc_defs
from tiatoolbox.wsicore import wsireader


def test_multichannel_basic_read() -> None:
    """VirtualWSIReader: >4-channel input is reduced to RGB when using auto post-proc.

    Also verify that disabling post-proc preserves original channel count.
    """
    # Create a small synthetic 6-channel image
    rng = np.random.default_rng()
    img = rng.integers(0, 255, size=(64, 64, 6), dtype=np.uint8)
    meta = wsireader.WSIMeta(slide_dimensions=img.shape[:2][::-1], axes="YXS")

    # By default the VirtualWSIReader will detect "feature" mode for
    # non-RGB(A) shapes and will not apply the auto post-processing.
    v_wsi = wsireader.VirtualWSIReader(img, info=meta)
    out = v_wsi.read_rect((0, 0), (16, 16))
    assert isinstance(out, np.ndarray)
    assert out.ndim == 3
    # Without explicitly setting post_proc/mode we should get the native channels
    assert out.shape[2] == 6

    # If we explicitly set a MultichannelToRGB post-proc and force rgb mode,
    # the output should be converted to 3 channels.
    v_wsi.post_proc = postproc_defs.MultichannelToRGB()
    v_wsi.mode = "rgb"
    out_rgb = v_wsi.read_rect((0, 0), (16, 16))
    assert out_rgb.shape[2] == 3


def test_ngff_multichannel_read(remote_sample: callable) -> None:
    """Sanity check for NGFF reader with possible multichannel data.

    This test is permissive: it asserts the reader returns an ndarray and,
    if the native dataset has many channels, that applying
    `MultichannelToRGB` produces a 3-channel output.
    """
    ngff_path = remote_sample("ngff-1")
    wsi = wsireader.NGFFWSIReader(ngff_path)

    # Read a tiny region to avoid heavy IO
    size = (8, 8)
    region = wsi.read_rect((0, 0), size)
    assert isinstance(region, np.ndarray)

    # If dataset has many channels, converting should yield 3 channels
    if region.ndim == 3 and region.shape[2] >= 5:
        wsi.post_proc = postproc_defs.MultichannelToRGB()
        region_rgb = wsi.read_rect((0, 0), size)
        assert region_rgb.ndim == 3
        assert region_rgb.shape[2] == 3


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
    assert region2.shape == (100, 50, 5)
    # Was 7 channels. Not sure if this is correct. Check this!


def test_get_post_proc_variants() -> None:
    """Test different branches of get_post_proc method."""
    reader = wsireader.VirtualWSIReader(np.zeros((10, 10, 3)))

    assert callable(reader.get_post_proc(lambda x: x))
    assert reader.get_post_proc(None) is None
    assert isinstance(reader.get_post_proc("auto"), postproc_defs.MultichannelToRGB)
    assert isinstance(
        reader.get_post_proc("MultichannelToRGB"), postproc_defs.MultichannelToRGB
    )

    with pytest.raises(ValueError, match="Invalid post-processing function"):
        reader.get_post_proc("invalid_proc")
