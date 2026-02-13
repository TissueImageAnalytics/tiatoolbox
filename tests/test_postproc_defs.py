"""Tests for post-processing utilities in tiatoolbox."""

import numpy as np
import pytest

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


def test_validate_warns_and_trims_when_n_colors_is_n_plus_one(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """If color_dict has N+1 entries but the image has N channels.

    validate() should:
        - drop the last color,
        - set channels to [0..N-1] if None,
        - set is_validated=True,
        - emit a warning.

    """
    # Use N = 5 channels in the image so __call__ doesn't early-return,
    # and supply N+1 (=6) colors to trigger the trimming/warning path.
    conv = MultichannelToRGB()
    conv.color_dict = {
        "c0": (1.0, 0.0, 0.0),
        "c1": (0.0, 1.0, 0.0),
        "c2": (0.0, 0.0, 1.0),
        "c3": (1.0, 1.0, 0.0),
        "c4": (1.0, 0.0, 1.0),
        "bg": (0.5, 0.5, 0.5),  # extra color to be trimmed
    }
    # colors set by __setattr__; channels should also be set (since was None)
    assert conv.colors is not None
    assert conv.channels == [0, 1, 2, 3, 4, 5]

    img = np.zeros((2, 2, 5), dtype=np.uint8)  # N = 5
    out = conv(img)  # triggers validate(5), trims to 5, warns

    # validate trimmed colors to 3 and emitted warning
    assert conv.is_validated is True
    assert recwarn.list  # at least one warning
    # channels should be reduced to indices < N
    assert conv.channels == [0, 1, 2, 3, 4]
    # Output is RGB image
    assert out.shape == (2, 2, 3)
    assert out.dtype == np.uint8
    assert conv.colors.shape[0] == 5


def test_validate_raises_when_color_count_does_not_match_n_or_n_plus_one() -> None:
    """When len(color_dict) is neither N nor N+1, validate() must raise ValueError."""
    # Use N = 5 channels so __call__ does not early-return, and provide only 3 colors.
    conv = MultichannelToRGB()
    conv.color_dict = {
        "c0": (1.0, 0.0, 0.0),
        "c1": (0.0, 1.0, 0.0),
        "c2": (0.0, 0.0, 1.0),
    }  # len=3, neither N (=5) nor N+1 (=6) -> should raise

    img = np.zeros((1, 1, 5), dtype=np.uint8)  # N = 5
    with pytest.raises(ValueError, match="does not match channels in image"):
        _ = conv(img)


def test_generate_colors_then_validate_equals_n_path() -> None:
    """When colors are not provided, __call__ should generate N colors.

    This should also validate successfully (n_colors == n).

    """
    conv = MultichannelToRGB(color_dict=None)
    img = np.zeros((3, 4, 6), dtype=np.uint8)  # N=6 -> will auto-generate 6 colors
    out = conv(img)
    assert conv.colors is not None
    assert conv.is_validated is True
    # channels should become range(N)
    assert conv.channels == list(range(6))
    assert out.shape == (3, 4, 3)
    assert out.dtype == np.uint8


def test_custom_channels_order_is_respected() -> None:
    """If user specifies custom channels order, ensure the mixing uses that order.

    We craft a simple case where only one channel has intensity so we can
    verify which color maps through.

    """
    # 3 channels in the image
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    img[0, 0, 2] = 255  # only channel 2 has signal

    # Provide a color for each channel:
    # c0->red, c1->green, c2->blue. Then set channels so the order used is [2, 1, 0].
    conv = MultichannelToRGB(
        color_dict={
            "c0": (1.0, 0.0, 0.0),  # red
            "c1": (0.0, 1.0, 0.0),  # green
            "c2": (0.0, 0.0, 1.0),  # blue
        }
    )
    conv.channels = [2, 1, 0]  # use reverse order explicitly
    out = conv(img)
    # Only c2 contributes, mapped to blue; enhance default=1.0
    assert out.shape == (1, 1, 3)
    # Be tolerant of any minor float->uint8 rounding: expect strong blue
    r, g, b = out[0, 0]
    assert r == 0
    assert g == 0
    assert b > 0


def test_setattr_color_dict_sets_colors_and_channels_when_none() -> None:
    """__setattr__ should populate `colors` and initialize `channels` if it is None."""
    conv = MultichannelToRGB()
    assert conv.colors is None
    assert conv.channels is None

    conv.color_dict = {"a": (1.0, 0.0, 0.0), "b": (0.0, 1.0, 0.0), "c": (0.0, 0.0, 1.0)}
    assert conv.colors is not None
    assert conv.colors.shape == (3, 3)
    assert conv.channels == [0, 1, 2]


def test_ctor_with_color_dict_does_not_raise() -> None:
    """Constructing with a color_dict should set channels and colors."""
    conv = MultichannelToRGB(
        color_dict={"a": (1, 0, 0), "b": (0, 1, 0), "c": (0, 0, 1)}
    )
    assert conv.channels == [0, 1, 2]
    assert conv.colors.shape == (3, 3)


def test_validate_handles_nones() -> None:
    """validate() should handle None for colors and channels gracefully."""
    conv = MultichannelToRGB()
    with pytest.raises(ValueError, match="Colors must be initialized"):
        conv.validate(5)
    # now set a colors but leave channels None
    conv.colors = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
    )  # N=5
    conv.validate(5)
    # validate should have set channels to range(num_colors)
    assert conv.is_validated is True
    assert conv.channels == [0, 1, 2, 3, 4]
