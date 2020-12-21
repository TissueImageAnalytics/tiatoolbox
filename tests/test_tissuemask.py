from tiatoolbox.dataloader import wsireader

import numpy as np

from tiatoolbox.tools import tissuemask


# -------------------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------------------


def test_otsu_masker(_sample_svs):
    """Test Otsu's thresholding method."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    mpp = 32
    thumb = wsi.get_thumbnail(mpp, "mpp")
    masker = tissuemask.OtsuTissueMasker()
    mask_a = masker.fit_transform(thumb)

    masker.fit(thumb)
    mask_b = masker.transform(thumb)

    assert np.array_equal(mask_a, mask_b)
    assert masker.threshold is not None
    assert len(np.unique(mask_a)) == 2
    assert mask_a.shape == thumb.shape[:2]


def test_morphological_masker(_sample_svs):
    """Test simple morphological thresholding."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    thumb = wsi.get_thumbnail()
    masker = tissuemask.MorphologicalMasker()
    mask_a = masker.fit_transform(thumb)

    masker.fit(thumb)
    mask_b = masker.transform(thumb)

    assert np.array_equal(mask_a, mask_b)
    assert masker.threshold is not None
    assert len(np.unique(mask_a)) == 2
    assert mask_a.shape == thumb.shape[:2]


def test_morphological_masker_int_kernel_size(_sample_svs):
    """Test simple morphological thresholding with mpp with int kernel_size."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    mpp = 32
    thumb = wsi.get_thumbnail(mpp, "mpp")
    masker = tissuemask.MorphologicalMasker(kernel_size=5)
    mask_a = masker.fit_transform(thumb)

    masker.fit(thumb)
    mask_b = masker.transform(thumb)

    assert np.array_equal(mask_a, mask_b)
    assert masker.threshold is not None
    assert len(np.unique(mask_a)) == 2
    assert mask_a.shape == thumb.shape[:2]


def test_morphological_masker_mpp(_sample_svs):
    """Test simple morphological thresholding with mpp."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    mpp = 32
    thumb = wsi.get_thumbnail(mpp, "mpp")
    masker = tissuemask.MorphologicalMasker(mpp=mpp)
    mask_a = masker.fit_transform(thumb)

    masker.fit(thumb)
    mask_b = masker.transform(thumb)

    assert np.array_equal(mask_a, mask_b)
    assert masker.threshold is not None
    assert len(np.unique(mask_a)) == 2
    assert mask_a.shape == thumb.shape[:2]


def test_morphological_masker_power(_sample_svs):
    """Test simple morphological thresholding with objective power."""
    wsi = wsireader.OpenSlideWSIReader(_sample_svs)
    power = 1.25
    thumb = wsi.get_thumbnail(power, "power")
    masker = tissuemask.MorphologicalMasker(power=power)
    mask_a = masker.fit_transform(thumb)

    masker.fit(thumb)
    mask_b = masker.transform(thumb)

    assert np.array_equal(mask_a, mask_b)
    assert masker.threshold is not None
    assert len(np.unique(mask_a)) == 2
    assert mask_a.shape == thumb.shape[:2]
