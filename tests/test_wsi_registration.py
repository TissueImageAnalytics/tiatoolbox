import urllib

import PIL.Image as Image
import numpy as np
import pytest

from tiatoolbox.tools.registration.wsi_registration import DFBRegistrtation


def test_extract_features():
    """Test for CNN based feature extraction function."""
    main_url = "https://tiatoolbox.dcs.warwick.ac.uk/testdata/registration/"
    urllib.request.urlretrieve(main_url + "HE_1_level8_gray.png", "sample.png")
    fixed_img = np.asarray(Image.open("sample.png"))
    urllib.request.urlretrieve(main_url + "HE_2_level8_gray.png", "sample.png")
    moving_img = np.asarray(Image.open("sample.png"))

    df = DFBRegistrtation()
    with pytest.raises(
        ValueError,
        match=r".*The required shape for fixed and moving images is n x m x 3.*",
    ):
        _ = df.extract_features(fixed_img, moving_img)

    fixed_img = np.repeat(np.expand_dims(fixed_img, axis=2), 1, axis=2)
    moving_img = np.repeat(np.expand_dims(moving_img, axis=2), 1, axis=2)
    with pytest.raises(
        ValueError, match=r".*The input images are expected to have 3 channels.*"
    ):
        _ = df.extract_features(fixed_img, moving_img)

    fixed_img = np.repeat(fixed_img, 3, axis=2)
    moving_img = np.repeat(moving_img, 3, axis=2)
    _ = df.extract_features(fixed_img, moving_img)


def test_feature_mapping():
    main_url = "https://tiatoolbox.dcs.warwick.ac.uk/testdata/registration/"
    urllib.request.urlretrieve(main_url + "HE_1_level8_gray.png", "sample.png")
    fixed_img = np.asarray(Image.open("sample.png"))
    urllib.request.urlretrieve(main_url + "HE_2_level8_gray.png", "sample.png")
    moving_img = np.asarray(Image.open("sample.png"))

    df = DFBRegistrtation()
    fixed_img = np.repeat(np.expand_dims(fixed_img, axis=2), 3, axis=2)
    moving_img = np.repeat(np.expand_dims(moving_img, axis=2), 3, axis=2)
    features = df.extract_features(fixed_img, moving_img)
    df.feature_mapping(features)
