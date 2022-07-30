import numpy as np
import pytest

from tiatoolbox.tools.registration.wsi_registration import match_histograms


def test_match_histograms():
    """Test for preprocessing/normalization of an image pair."""
    fixed_img = np.random.randint(256, size=(256, 256))
    moving_img = np.random.randint(256, size=(256, 256))
    _, _ = match_histograms(fixed_img, moving_img)
    _, _ = match_histograms(fixed_img, moving_img, 3)

    fixed_img = np.random.randint(256, size=(256, 256, 3))
    moving_img = np.random.randint(256, size=(256, 256, 3))
    with pytest.raises(
        ValueError, match=r".*The input images should be grayscale images.*"
    ):
        _, _ = match_histograms(fixed_img, moving_img)

    fixed_img = np.random.randint(256, size=(256, 256, 1))
    moving_img = np.random.randint(256, size=(256, 256, 1))
    _, _ = match_histograms(fixed_img, moving_img)
