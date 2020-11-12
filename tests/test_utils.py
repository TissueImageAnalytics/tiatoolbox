from tiatoolbox import utils

import pytest
import numpy as np

import tiatoolbox


def test_imresize():
    """Test for imresize."""
    img = np.zeros((2000, 2000, 3))
    resized_img = utils.transforms.imresize(img, 0.5)
    assert resized_img.shape == (1000, 1000, 3)


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
    """Test approximation of objective power from mpp."""
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

    wsi = tiatoolbox.dataloader.wsireader.OpenSlideWSIReader(_sample_svs)
    openslide_obj = wsi.openslide_wsi
    dictionary = dict(wsi.openslide_wsi.properties)

    class DummyOpenSlideObject(object):
        def __getattr__(self, name: str):
            if name != "properties":
                return getattr(openslide_obj, name)

        @property
        def properties(self):
            return dictionary

    wsi.openslide_wsi = DummyOpenSlideObject()

    del dictionary["openslide.objective-power"]
    with pytest.warns(UserWarning, match=r"Objective power inferred"):
        wsi.info

    dictionary["openslide.mpp-x"] = 10
    dictionary["openslide.mpp-y"] = 10
    with pytest.warns(UserWarning, match=r"MPP outside of sensible range"):
        wsi.info

    del dictionary["openslide.mpp-x"]
    del dictionary["openslide.mpp-y"]
    with pytest.warns(UserWarning, match=r"Unable to determine objective power"):
        wsi.info
