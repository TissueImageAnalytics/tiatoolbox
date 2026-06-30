"""Test for stain augmentation code."""

from pathlib import Path

import numpy as np
import pytest

from tiatoolbox.data import stain_norm_target
from tiatoolbox.tools.stainaugment import StainAugmentor
from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.utils import imread


def test_stainaugment(source_image: Path, norm_vahadane: Path) -> None:
    """Test functionality of the StainAugmentor class."""
    source_img = imread(Path(source_image))
    target_img = stain_norm_target()
    vahadane_img = imread(Path(norm_vahadane))

    # Test invalid method in the input
    with pytest.raises(ValueError, match=r".*Unsupported stain extractor method.*"):
        _ = StainAugmentor(method="invalid")

    # 1. Testing without stain matrix.
    # Test with macenko stain extractor
    augmentor = StainAugmentor(
        method="macenko",
        sigma1=3.0,
        sigma2=3.0,
        augment_background=True,
    )
    augmentor.fit(source_img)
    source_img_aug = augmentor.augment()
    assert source_img_aug.dtype == source_img.dtype
    assert np.shape(source_img_aug) == np.shape(source_img)
    assert np.mean(np.abs(source_img_aug / 255.0 - source_img / 255.0)) > 1e-2

    # 2. Testing with predefined stain matrix
    # We first extract the stain matrix of the target image and try to augment the
    # source image with respect to that image.
    norm = get_normalizer("vahadane")
    norm.fit(target_img)
    target_stain_matrix = norm.stain_matrix_target

    # Now we augment the source image with sigma1=0, sigma2=0 to force the augmentor
    # to act like a normalizer
    augmentor = StainAugmentor(
        method="vahadane",
        stain_matrix=target_stain_matrix,
        sigma1=0.0,
        sigma2=0.0,
        augment_background=False,
    )
    augmentor.fit(source_img, threshold=0.8)
    source_img_aug = augmentor.augment()

    # Should match vahadane normalized image
    assert np.mean(np.abs(vahadane_img / 255.0 - source_img_aug / 255.0)) < 1e-1
