import pathlib

import numpy as np
import pytest

from tiatoolbox.tools.registration.wsi_registration import prealignment
from tiatoolbox.utils.misc import imread


@pytest.fixture(scope="session")
def fixed_image(remote_sample) -> pathlib.Path:
    """Sample pytest fixture for fixed image.
    Download fixed image for pytest.
    """
    return remote_sample("fixed_image")

@pytest.fixture(scope="session")
def moving_image(remote_sample) -> pathlib.Path:
    """Sample pytest fixture for moving image.
    Download moving image for pytest.
    """
    return remote_sample("moving_image")

@pytest.fixture(scope="session")
def fixed_mask(remote_sample) -> pathlib.Path:
    """Sample pytest fixture for fixed mask.
    Download fixed mask for pytest.
    """
    return remote_sample("fixed_mask")

@pytest.fixture(scope="session")
def moving_mask(remote_sample) -> pathlib.Path:
    """Sample pytest fixture for moving mask.
    Download moving mask for pytest.
    """
    return remote_sample("moving_mask")

def test_prealignment(fixed_image, moving_image, fixed_mask, moving_mask):
	"""Test for prealignment of an image pair"""
	# main_url = "https://tiatoolbox.dcs.warwick.ac.uk/testdata/registration/"
	# urllib.request.urlretrieve(main_url + "HE_1_level8_gray.png", "sample.png")
	# fixed_img = np.asarray(Image.open("sample.png"))
	# urllib.request.urlretrieve(main_url + "HE_2_level8_gray.png", "sample.png")
	# moving_img = np.asarray(Image.open("sample.png"))
	# urllib.request.urlretrieve(main_url + "HE_1_level8_mask.png", "sample.png")
	# fixed_mask = np.asarray(Image.open("sample.png"))
	# urllib.request.urlretrieve(main_url + "HE_2_level8_mask.png", "sample.png")
	# moving_mask = np.asarray(Image.open("sample.png"))

	fixed_img = imread(pathlib.Path(fixed_image))
	moving_img = imread(pathlib.Path(moving_image))

	transform = prealignment(fixed_img, moving_img, fixed_mask, moving_mask)
	assert transform.shape == (3, 3)

	no_fixed_mask = np.zeros(shape=fixed_img.shape, dtype=int)
	no_moving_mask = np.zeros(shape=moving_img.shape, dtype=int)
	with pytest.raises(ValueError, match=r".*The foreground is missing in the mask.*"):
		_ = prealignment(fixed_img, moving_img, no_fixed_mask, no_moving_mask)

	with pytest.raises(
		ValueError,
		match=r".*Mismatch of shape between image and its corresponding mask.*",
	):
		_ = prealignment(fixed_img, moving_img, moving_mask, fixed_mask)

	fixed_img_3d = np.repeat(np.expand_dims(fixed_img, axis=2), 3, axis=2)
	moving_img_3d = np.repeat(np.expand_dims(moving_img, axis=2), 3, axis=2)
	with pytest.raises(
		ValueError, match=r".*The input images should be grayscale images.*"
	):
		_ = prealignment(fixed_img_3d, moving_img_3d, fixed_mask, moving_mask)

	fixed_mask = np.repeat(np.expand_dims(fixed_mask, axis=2), 3, axis=2)
	moving_mask = np.repeat(np.expand_dims(moving_mask, axis=2), 3, axis=2)
	_ = prealignment(fixed_img, moving_img, fixed_mask, moving_mask)


def test_rotation_step_range():
	"""Test if the value of rotation step is within the range"""
	fixed_img = np.random.randint(20, size=(256, 256))
	moving_img = np.random.randint(20, size=(256, 256))
	fixed_mask = np.random.randint(2, size=(256, 256))
	moving_mask = np.random.randint(2, size=(256, 256))

	_ = prealignment(fixed_img, moving_img, fixed_mask, moving_mask, rotation_step=15)

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

	_ = prealignment(fixed_img, moving_img, fixed_mask, moving_mask, dice_overlap=0.5)

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
