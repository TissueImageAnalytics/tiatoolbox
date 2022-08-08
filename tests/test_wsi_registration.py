import pathlib

import numpy as np
import pytest

from tiatoolbox.tools.registration.wsi_registration import DFBRegistrtation
from tiatoolbox.utils.misc import imread


def test_extract_features(fixed_image, moving_image):
	"""Test for CNN based feature extraction function."""

	fixed_img = imread(pathlib.Path(fixed_image))
	moving_img = imread(pathlib.Path(moving_image))

	df = DFBRegistrtation()
	with pytest.raises(
		ValueError,
		match=r".*The required shape for fixed and moving images is n x m x 3.*",
	):
		_ = df.extract_features(fixed_img[:,:,0], moving_img[:,:,0])

	fixed_img = np.expand_dims(fixed_img[:,:,0], axis=2)
	moving_img = np.expand_dims(moving_img[:,:,0], axis=2)
	with pytest.raises(
		ValueError, match=r".*The input images are expected to have 3 channels.*"
	):
		_ = df.extract_features(fixed_img, moving_img)

	fixed_img = np.repeat(
		np.expand_dims(
			np.repeat(
				np.expand_dims(np.arange(0, 64, 1, dtype=np.uint8), axis=1), 64, axis=1
			),
			axis=2,
		),
		3,
		axis=2,
	)
	output = df.extract_features(fixed_img, fixed_img)
	pool3_feat = output["block3_pool"][0, :].detach().numpy()
	pool4_feat = output["block4_pool"][0, :].detach().numpy()
	pool5_feat = output["block5_pool"][0, :].detach().numpy()

	_pool3_feat, _pool4_feat, _pool5_feat = np.load("features.npy", allow_pickle=True)
	assert np.mean(np.abs(pool3_feat - _pool3_feat)) < 1.0e-4
	assert np.mean(np.abs(pool4_feat - _pool4_feat)) < 1.0e-4
	assert np.mean(np.abs(pool5_feat - _pool5_feat)) < 1.0e-4
