import pytest

from tiatoolbox.tools.registration.wsi_registration import RegistrationConfig


def test_registeration_config():
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
