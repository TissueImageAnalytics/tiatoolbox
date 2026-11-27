"""Test for rust functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tiatoolbox import utils

if TYPE_CHECKING:
    import pytest


def test_contrast_enhancer() -> None:
    """Test contrast enhancement functionality."""
    input_array = np.array(
        [
            [
                [37, 244, 193, 106, 235, 128, 71, 140, 47],
                [103, 184, 72, 20, 188, 238, 126, 7, 0],
                [137, 195, 204, 32, 203, 170, 101, 77, 133],
            ]
        ],
        dtype=np.uint8,
    )

    # expected output of the contrast_enhancer
    result_array = np.array(
        [
            [
                [35, 255, 203, 110, 248, 133, 72, 146, 46],
                [106, 193, 73, 17, 198, 251, 131, 3, 0],
                [143, 205, 215, 30, 214, 178, 104, 78, 139],
            ]
        ],
        dtype=np.uint8,
    )

    # Calculating the contrast enhanced version of input_array
    output_array = utils.misc.contrast_enhancer(input_array, low_p=2, high_p=98)
    # The out_put array should be equal to expected result_array
    assert np.all(result_array == output_array)
    input_array = np.array(
        [
            [
                [0],
                [0],
                [0],
            ]
        ],
        dtype=np.uint8,
    )

    # expected output of the contrast_enhancer
    result_array = np.array(
        [
            [
                [0],
                [0],
                [0],
            ]
        ],
        dtype=np.uint8,
    )

    # Calculating the contrast enhanced version of input_array
    output_array = utils.misc.contrast_enhancer(input_array, low_p=2, high_p=98)
    # The out_put array should be equal to expected result_array
    assert np.all(result_array == output_array)


def test_rust_extension_built() -> None:
    """Test the Rust extension is compiled and used, not the NumPy fallback."""
    assert utils.misc.rmisc is not None
    assert hasattr(utils.misc.rmisc, "contrast_enhancer")


def test_contrast_enhancer_numpy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the NumPy fallback matches the Rust implementation."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 48, 3), dtype=np.uint8)
    rust_output = utils.misc.contrast_enhancer(img, low_p=2, high_p=98)

    monkeypatch.setattr(utils.misc, "rmisc", None)
    numpy_output = utils.misc.contrast_enhancer(img, low_p=2, high_p=98)

    assert np.array_equal(rust_output, numpy_output)
