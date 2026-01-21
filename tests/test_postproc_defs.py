"""Tests for post-processing utilities in tiatoolbox."""

import numpy as np

from tiatoolbox.utils.postproc_defs import MultichannelToRGB


def test_multichannel_to_rgb_returns_same_for_rgb_images() -> None:
    """Verify that RGB images are returned unchanged by MultichannelToRGB."""
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0] = [10, 20, 30]

    conv = MultichannelToRGB()
    out = conv(img)
    assert out.dtype == img.dtype
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_multichannel_to_rgb_uint16_conversion_and_enhance() -> None:
    """Check uint16 multichannel conversion, enhancement, and clipping behavior."""
    # create uint16 image with 6 channels, first channel non-zero
    img = np.zeros((1, 1, 6), dtype=np.uint16)
    img[0, 0, 0] = 65535

    conv = MultichannelToRGB()
    # assign a simple color dict so colors map to full red for channel 0
    conv.color_dict = {
        f"channel_{i}": (1.0, 0.0, 0.0) if i == 0 else (0.0, 0.0, 0.0) for i in range(6)
    }
    conv.enhance = 2.0

    out = conv(img)
    assert out.shape == (1, 1, 3)
    # red channel should be clipped to 255 after enhancement
    assert out.dtype == np.uint8
    assert out[0, 0, 0] == 255


def test_generate_colors_and_color_dict_assignment() -> None:
    """Ensure color generation and color_dict assignment update internal state."""
    conv = MultichannelToRGB()
    conv.generate_colors(4)
    assert conv.color_dict is not None
    assert len(conv.color_dict) == 4

    # assign a color_dict and ensure internal colors array updates
    cd = {"a": (1.0, 0.0, 0.0), "b": (0.0, 1.0, 0.0)}
    conv.color_dict = cd
    assert conv.colors is not None
    assert conv.colors.shape[0] == 2
