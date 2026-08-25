"""Test for rust functionality."""

import numpy as np

from tiatoolbox import utils


def test_contrast_enhancer() -> None:
    """Test contrast enhancement functionality."""
    input_array = np.array(
        [
            [37, 244, 193, 106, 235, 128, 71, 140, 47],
            [103, 184, 72, 20, 188, 238, 126, 7, 0],
            [137, 195, 204, 32, 203, 170, 101, 77, 133],
        ],
        dtype=np.uint8,
    )

    # expected output of the contrast_enhancer
    result_array = np.array(
        [
            [35, 255, 203, 110, 248, 133, 72, 146, 46],
            [106, 193, 73, 17, 198, 251, 131, 3, 0],
            [143, 205, 215, 30, 214, 178, 104, 78, 139],
        ],
        dtype=np.uint8,
    )

    # Calculating the contrast enhanced version of input_array
    output_array = utils.misc.contrast_enhancer(input_array, low_p=2, high_p=98)
    # The out_put array should be equal to expected result_array
    assert np.all(result_array == output_array)
