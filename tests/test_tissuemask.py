import numpy as np
import pytest
import cv2

from tiatoolbox.wsicore import wsireader
from tiatoolbox.tools import tissuemask


# -------------------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------------------


def test_otsu_masker(_sample_svs):
    """Test Otsu's thresholding method."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    mpp = 32
    thumb = wsi.slide_thumbnail(mpp, "mpp")
    masker = tissuemask.OtsuTissueMasker()
    mask_a = masker.fit_transform([thumb])[0]

    masker.fit([thumb])
    mask_b = masker.transform([thumb])[0]

    assert np.array_equal(mask_a, mask_b)
    assert masker.threshold is not None
    assert len(np.unique(mask_a)) == 2
    assert mask_a.shape == thumb.shape[:2]


def test_otsu_grescale_masker(_sample_svs):
    """Test Otsu's thresholding method with greyscale inputs."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    mpp = 32
    thumb = wsi.slide_thumbnail(mpp, "mpp")
    thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    inputs = thumb[np.newaxis, ..., np.newaxis]
    masker = tissuemask.OtsuTissueMasker()
    mask_a = masker.fit_transform(inputs)[0]

    masker.fit(inputs)
    mask_b = masker.transform(inputs)[0]

    assert np.array_equal(mask_a, mask_b)
    assert masker.threshold is not None
    assert len(np.unique(mask_a)) == 2
    assert mask_a.shape == thumb.shape[:2]


def test_morphological_masker(_sample_svs):
    """Test simple morphological thresholding."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    thumb = wsi.slide_thumbnail()
    masker = tissuemask.MorphologicalMasker()
    mask_a = masker.fit_transform([thumb])[0]

    masker.fit([thumb])
    mask_b = masker.transform([thumb])[0]

    assert np.array_equal(mask_a, mask_b)
    assert masker.threshold is not None
    assert len(np.unique(mask_a)) == 2
    assert mask_a.shape == thumb.shape[:2]


def test_morphological_grescale_masker(_sample_svs):
    """Test morphological masker with greyscale inputs."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    mpp = 32
    thumb = wsi.slide_thumbnail(mpp, "mpp")
    thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    inputs = thumb[np.newaxis, ..., np.newaxis]
    masker = tissuemask.MorphologicalMasker()
    mask_a = masker.fit_transform(inputs)[0]

    masker.fit(inputs)
    mask_b = masker.transform(inputs)[0]

    assert np.array_equal(mask_a, mask_b)
    assert masker.threshold is not None
    assert len(np.unique(mask_a)) == 2
    assert mask_a.shape == thumb.shape[:2]


def test_morphological_masker_int_kernel_size(_sample_svs):
    """Test simple morphological thresholding with mpp with int kernel_size."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    mpp = 32
    thumb = wsi.slide_thumbnail(mpp, "mpp")
    masker = tissuemask.MorphologicalMasker(kernel_size=5)
    mask_a = masker.fit_transform([thumb])[0]

    masker.fit([thumb])
    mask_b = masker.transform([thumb])[0]

    assert np.array_equal(mask_a, mask_b)
    assert masker.threshold is not None
    assert len(np.unique(mask_a)) == 2
    assert mask_a.shape == thumb.shape[:2]


def test_morphological_masker_mpp(_sample_svs):
    """Test simple morphological thresholding with mpp."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    mpp = 32
    thumb = wsi.slide_thumbnail(mpp, "mpp")
    masker = tissuemask.MorphologicalMasker(mpp=mpp)
    mask_a = masker.fit_transform([thumb])[0]

    masker.fit([thumb])
    mask_b = masker.transform([thumb])[0]

    assert np.array_equal(mask_a, mask_b)
    assert masker.threshold is not None
    assert len(np.unique(mask_a)) == 2
    assert mask_a.shape == thumb.shape[:2]


def test_morphological_masker_power(_sample_svs):
    """Test simple morphological thresholding with objective power."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    power = 1.25
    thumb = wsi.slide_thumbnail(power, "power")
    masker = tissuemask.MorphologicalMasker(power=power)
    mask_a = masker.fit_transform([thumb])[0]

    masker.fit([thumb])
    mask_b = masker.transform([thumb])[0]

    assert np.array_equal(mask_a, mask_b)
    assert masker.threshold is not None
    assert len(np.unique(mask_a)) == 2
    assert mask_a.shape == thumb.shape[:2]


def test_transform_before_fit_otsu():
    """Test otsu masker error on transform before fit."""
    image = np.ones((1, 10, 10))
    masker = tissuemask.OtsuTissueMasker()
    with pytest.raises(Exception):
        masker.transform([image])[0]


def test_transform_before_fit_morphological():
    """Test morphological masker error on transform before fit."""
    image = np.ones((1, 10, 10))
    masker = tissuemask.MorphologicalMasker()
    with pytest.raises(Exception):
        masker.transform([image])[0]


def test_transform_fit_otsu_wrong_shape():
    """Test giving the incorrect input shape to otsu masker."""
    image = np.ones((10, 10))
    masker = tissuemask.OtsuTissueMasker()
    with pytest.raises(ValueError):
        masker.fit([image])


def test_transform_morphological_conflicting_args():
    """Test giving conflicting arguments to morphological masker."""
    with pytest.raises(ValueError):
        tissuemask.MorphologicalMasker(mpp=32, power=1.25)


def test_kernel_size_none():
    """Test giveing a None kernel size for morphological masker."""
    tissuemask.MorphologicalMasker(kernel_size=None)
