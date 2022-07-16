"""Tests for code related to Image Registration."""

import numpy as np
import pytest

from tiatoolbox.tools.registration.wsi_registration import (
    RegistrationConfig,
    preprocess,
)


def test_registeration_config():
    """Test for creating an instance of RegistrationConfig class"""
    _ = RegistrationConfig()

    _ = RegistrationConfig(
        resolution=0.03125,
        units="power",
        number_of_rotations=10,
        input_image_size=(224, 224),
    )

    with pytest.raises(ValueError, match=r".*Invalid resolution units.*"):
        _ = RegistrationConfig(
            resolution=0.03125,
            units="alpha",
            number_of_rotations=10,
            input_image_size=(224, 224),
        )


def test_preprocess():
    """Test for preprocessing/normalization of an image pair"""
    fixed_img = np.random.randint(256, size=(256, 256))
    moving_img = np.random.randint(256, size=(256, 256))
    _, _ = preprocess(fixed_img, moving_img)

    fixed_img = np.random.randint(256, size=(256, 256, 3))
    moving_img = np.random.randint(256, size=(256, 256, 3))
    with pytest.raises(
        ValueError, match=r".*The input images should be grayscale images.*"
    ):
        _, _ = preprocess(fixed_img, moving_img)
