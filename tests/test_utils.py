import random
from pathlib import Path

from tiatoolbox import utils

import pytest
import numpy as np


def test_imresize():
    """Test for imresize."""
    img = np.zeros((2000, 1000, 3))
    resized_img = utils.transforms.imresize(img, scale_factor=0.5)
    assert resized_img.shape == (1000, 500, 3)

    resized_img = utils.transforms.imresize(resized_img, scale_factor=2.0)
    assert resized_img.shape == (2000, 1000, 3)


def test_background_composite():
    """Test for background composite."""
    new_im = np.zeros((2000, 2000, 4)).astype("uint8")
    new_im[:1000, :, 3] = 255
    im = utils.transforms.background_composite(new_im)
    assert np.all(im[1000:, :, :] == 255)
    assert np.all(im[:1000, :, :] == 0)

    im = utils.transforms.background_composite(new_im, alpha=True)
    assert np.all(im[:, :, 3] == 255)


def test_mpp2objective_power(_sample_svs):
    """Test approximate conversion of from mpp to objective power."""
    mapping = [
        (0.05, 100),
        (0.07, 100),
        (0.10, 60),
        (0.12, 60),
        (0.15, 40),
        (0.29, 40),
        (0.30, 20),
        (0.49, 20),
        (0.60, 10),
        (1.00, 10),
        (1.20, 5),
        (2.00, 5),
        (2.40, 2.5),
        (3.00, 2.5),
        (4.80, 1.25),
        (9.00, 1.25),
    ]
    for mpp, result in mapping:
        assert utils.misc.mpp2objective_power(mpp) == result
        assert utils.misc.mpp2objective_power([mpp] * 2) == result

    with pytest.raises(ValueError):
        utils.misc.mpp2objective_power(mpp=10)


def test_sub_pixel_read_bounds():
    """Test sub-pixel numpy image reads with known tricky parameters."""
    image_path = Path(__file__).parent / "data" / "source_image.png"
    assert image_path.exists()
    test_image = utils.misc.imread(image_path)

    x = 6
    y = -4
    w = 21.805648705868652
    h = 0.9280264518437986
    bounds = (x, y, x + w, y + h)
    ow = 88
    oh = 98
    output = utils.image.sub_pixel_read_bounds(test_image, bounds, (ow, oh))
    assert (ow, oh) == tuple(output.shape[:2][::-1])

    x = 13
    y = 15
    w = 29.46
    h = 6.92
    bounds = (x, y, x + w, y + h)
    ow = 93
    oh = 34
    output = utils.image.sub_pixel_read_bounds(test_image, bounds, (ow, oh))
    assert (ow, oh) == tuple(output.shape[:2][::-1])


def test_fuzz_sub_pixel_read_bounds():
    """Fuzz test for numpy sub-pixel image reads."""
    random.seed(0)

    image_path = Path(__file__).parent / "data" / "source_image.png"
    assert image_path.exists()
    test_image = utils.misc.imread(image_path)

    for _ in range(10000):
        x = random.randint(-5, 32 - 5)
        y = random.randint(-5, 32 - 5)
        w = random.random() * random.randint(1, 32)
        h = random.random() * random.randint(1, 32)
        bounds = (x, y, x + w, y + h)
        ow = random.randint(4, 128)
        oh = random.randint(4, 128)
        output = utils.image.sub_pixel_read_bounds(
            test_image, bounds, (ow, oh), interpolation="linear"
        )
        assert (ow, oh) == tuple(output.shape[:2][::-1])


def test_fuzz_bounds2size():
    """"""
    random.seed(0)
    for _ in range(1000):
        size = (random.randint(-1000, 1000), random.randint(-1000, 1000))
        location = (random.randint(-1000, 1000), random.randint(-1000, 1000))
        bounds = (*location, *(sum(x) for x in zip(size, location)))
        assert np.array_equal(utils.transforms.bounds2size(bounds), size)
