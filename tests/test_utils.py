from tiatoolbox import utils
from tiatoolbox.utils.exceptions import FileNotSupported

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


def test_contrast_enhancer():
    """"Test contrast enhancement funcitionality."""
    input_array = np.array(
        [
            [[37, 244, 193], [106, 235, 128], [71, 140, 47]],
            [[103, 184, 72], [20, 188, 238], [126, 7, 0]],
            [[137, 195, 204], [32, 203, 170], [101, 77, 133]],
        ],
        dtype=np.uint8,
    )

    result_array = np.array(
        [
            [[35, 255, 203], [110, 248, 133], [72, 146, 46]],
            [[106, 193, 73], [17, 198, 251], [131, 3, 0]],
            [[143, 205, 215], [30, 214, 178], [104, 78, 139]],
        ],
        dtype=np.uint8,
    )

    output_array = utils.misc.contrast_enhancer(input_array, low_p=2, high_p=98)
    assert np.all(result_array == output_array)


def test_load_stain_matrix():
    with pytest.raises(FileNotSupported):
        utils.misc.load_stain_matrix("/samplefile.xlsx")

    with pytest.raises(TypeError):
        # load_stain_matrix requires numpy array as input providing list here
        utils.misc.load_stain_matrix([1, 2, 3])


def test_get_luminosity_tissue_mask():
    with pytest.raises(ValueError):
        utils.misc.get_luminosity_tissue_mask(img=np.zeros((100, 100, 3)), threshold=0)
