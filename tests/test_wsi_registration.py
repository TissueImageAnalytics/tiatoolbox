import pathlib

import numpy as np
import pytest

from tiatoolbox.tools.registration.wsi_registration import (
    match_histograms,
    prealignment,
)
from tiatoolbox.utils.misc import imread


def test_prealignment(fixed_image, moving_image, fixed_mask, moving_mask):
    """Test for prealignment of an image pair"""
    fixed_img = imread(fixed_image)
    moving_img = imread(moving_image)
    fixed_mask = imread(fixed_mask)
    moving_mask = imread(moving_mask)

    with pytest.raises(
        ValueError, match=r".*The input images should be grayscale images.*"
    ):
        _ = prealignment(fixed_img, moving_img, fixed_mask, moving_mask)

    expected = np.array([[1, 0, 337.8], [0, 0, 767.7], [0, 0, 1]])
    fixed_img, moving_img = fixed_img[:, :, 0], moving_img[:, :, 0]
    output = prealignment(
        fixed_img,
        moving_img,
        fixed_mask,
        moving_mask,
        dice_overlap=0.5,
        rotation_step=10,
    )
    assert output.shape == (3, 3)
    assert np.mean((expected[:2, :2] - output[:2, :2])) <= 0.75
    assert np.mean(output[:2, 2] - expected[:2, 2]) < 1.0

    no_fixed_mask = np.zeros(shape=fixed_img.shape, dtype=int)
    no_moving_mask = np.zeros(shape=moving_img.shape, dtype=int)
    with pytest.raises(ValueError, match=r".*The foreground is missing in the mask.*"):
        _ = prealignment(fixed_img, moving_img, no_fixed_mask, no_moving_mask)

    with pytest.raises(
        ValueError,
        match=r".*Mismatch of shape between image and its corresponding mask.*",
    ):
        _ = prealignment(fixed_img, moving_img, moving_mask, fixed_mask)

    with pytest.raises(
        ValueError, match=r".*Please select the rotation step in between 10 and 20.*"
    ):
        _ = prealignment(
            fixed_img, moving_img, fixed_mask, moving_mask, rotation_step=9
        )

    with pytest.raises(
        ValueError, match=r".*Please select the rotation step in between 10 and 20.*"
    ):
        _ = prealignment(
            fixed_img, moving_img, fixed_mask, moving_mask, rotation_step=21
        )


def test_dice_overlap_range():
    """Test if the value of rotation step is within the range"""
    fixed_img = np.random.randint(20, size=(256, 256))
    moving_img = np.random.randint(20, size=(256, 256))
    fixed_mask = np.random.randint(2, size=(256, 256))
    moving_mask = np.random.randint(2, size=(256, 256))

    with pytest.raises(
        ValueError, match=r".*The dice_overlap should be in between 0 and 1.0.*"
    ):
        _ = prealignment(fixed_img, moving_img, fixed_mask, moving_mask, dice_overlap=2)

    with pytest.raises(
        ValueError, match=r".*The dice_overlap should be in between 0 and 1.0.*"
    ):
        _ = prealignment(
            fixed_img, moving_img, fixed_mask, moving_mask, dice_overlap=-1
        )


def test_warning(
    fixed_image,
    moving_image,
    fixed_mask,
    moving_mask,
):
    fixed_img = imread(pathlib.Path(fixed_image))
    moving_img = imread(pathlib.Path(moving_image))
    fixed_mask = imread(pathlib.Path(fixed_mask))
    moving_mask = imread(pathlib.Path(moving_mask))
    fixed_img, moving_img = fixed_img[:, :, 0], moving_img[:, :, 0]
    with pytest.warns(UserWarning):
        _ = prealignment(
            fixed_img, moving_img, fixed_mask, moving_mask, dice_overlap=0.9
        )


def test_match_histograms():
    """Test for preprocessing/normalization of an image pair."""
    image_a = np.random.randint(256, size=(256, 256, 3))
    image_b = np.random.randint(256, size=(256, 256, 3))
    with pytest.raises(
        ValueError, match=r".*The input images should be grayscale images.*"
    ):
        _, _ = match_histograms(image_a, image_b)

    image_a = np.random.randint(256, size=(256, 256))
    image_b = np.random.randint(256, size=(256, 256))
    _, _ = match_histograms(image_a, image_b, 3)

    image_a = np.random.randint(256, size=(256, 256, 1))
    image_b = np.random.randint(256, size=(256, 256, 1))
    _, _ = match_histograms(image_a, image_b)

    image_a = np.array(
        [
            [129, 134, 195, 241, 168],
            [231, 91, 145, 91, 0],
            [64, 87, 194, 112, 99],
            [138, 111, 99, 124, 86],
            [164, 127, 167, 222, 100],
        ],
        dtype=np.uint8,
    )
    image_b = np.array(
        [
            [25, 91, 177, 212, 114],
            [62, 86, 83, 31, 17],
            [13, 16, 191, 19, 149],
            [58, 127, 22, 111, 255],
            [164, 7, 110, 76, 222],
        ],
        dtype=np.uint8,
    )
    expected_output = np.array(
        [
            [91, 110, 191, 255, 164],
            [222, 22, 114, 22, 7],
            [13, 17, 177, 76, 31],
            [111, 62, 31, 83, 16],
            [127, 86, 149, 212, 58],
        ]
    )
    norm_image_a, norm_image_b = match_histograms(image_a, image_b)
    assert np.all(norm_image_a == expected_output)
    assert np.all(norm_image_b == image_b)
