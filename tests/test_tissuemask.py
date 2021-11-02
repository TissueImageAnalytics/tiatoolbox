"""Tests for code related to tissue mask generation."""

import pathlib

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from tiatoolbox import cli
from tiatoolbox.tools import tissuemask
from tiatoolbox.utils.exceptions import MethodNotSupported
from tiatoolbox.wsicore import wsireader

# -------------------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------------------


def test_otsu_masker(sample_svs):
    """Test Otsu's thresholding method."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
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


def test_otsu_greyscale_masker(sample_svs):
    """Test Otsu's thresholding method with greyscale inputs."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
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


def test_morphological_masker(sample_svs):
    """Test simple morphological thresholding."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    thumb = wsi.slide_thumbnail()
    masker = tissuemask.MorphologicalMasker()
    mask_a = masker.fit_transform([thumb])[0]

    masker.fit([thumb])
    mask_b = masker.transform([thumb])[0]

    assert np.array_equal(mask_a, mask_b)
    assert masker.threshold is not None
    assert len(np.unique(mask_a)) == 2
    assert mask_a.shape == thumb.shape[:2]


def test_morphological_greyscale_masker(sample_svs):
    """Test morphological masker with greyscale inputs."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
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


def test_morphological_masker_int_kernel_size(sample_svs):
    """Test simple morphological thresholding with mpp with int kernel_size."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
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


def test_morphological_masker_mpp(sample_svs):
    """Test simple morphological thresholding with mpp."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    mpp = 32
    thumb = wsi.slide_thumbnail(mpp, "mpp")
    kwarg_sets = [
        dict(mpp=mpp),
        dict(mpp=[mpp, mpp]),
    ]
    for kwargs in kwarg_sets:
        masker = tissuemask.MorphologicalMasker(**kwargs)
        mask_a = masker.fit_transform([thumb])[0]

        masker.fit([thumb])
        mask_b = masker.transform([thumb])[0]

        assert np.array_equal(mask_a, mask_b)
        assert masker.threshold is not None
        assert len(np.unique(mask_a)) == 2
        assert mask_a.shape == thumb.shape[:2]


def test_morphological_masker_power(sample_svs):
    """Test simple morphological thresholding with objective power."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
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


def test_morphological_kernel_size_none():
    """Test giveing a None kernel size for morphological masker."""
    tissuemask.MorphologicalMasker(kernel_size=None)


def test_morphological_min_region_size():
    """Test morphological masker with min_region_size set.

    Creates a test image (0=foreground, 1=background) and applies the
    morphological masker with min_region_size=6. This should output
    only the largest square region as foreground in the mask
    (0=background, 1=foreground).
    """
    # Create a blank image of 1s
    img = np.ones((10, 10))
    # Create a large square region of 9 zeros
    img[1:4, 1:4] = 0
    # Create a row of 5 zeros
    img[1, 5:10] = 0
    # Create a single 0
    img[8, 8] = 0

    masker = tissuemask.MorphologicalMasker(kernel_size=1, min_region_size=6)
    output = masker.fit_transform([img[..., np.newaxis]])
    assert np.sum(output[0]) == 9

    # Create the expected output with jsut the large square region
    # but as ones against zeros (the mask is the inverse of the input).
    expected = np.zeros((10, 10))
    expected[1:4, 1:4] = 1

    assert np.all(output[0] == expected)


def test_cli_tissue_mask_Otsu(sample_svs):
    """Test Otsu tissue masking with default input CLI."""
    source_img = pathlib.Path(sample_svs)
    runner = CliRunner()
    tissue_mask_result = runner.invoke(
        cli.main,
        [
            "tissue-mask",
            "--img-input",
            str(source_img),
            "--method",
            "Otsu",
        ],
    )

    assert tissue_mask_result.exit_code == 0

    output_path = str(pathlib.Path(sample_svs.parent, "tissue_mask"))
    tissue_mask_result = runner.invoke(
        cli.main,
        [
            "tissue-mask",
            "--img-input",
            str(source_img),
            "--method",
            "Otsu",
            "--mode",
            "save",
            "--output-path",
            output_path,
        ],
    )

    assert tissue_mask_result.exit_code == 0
    assert pathlib.Path(output_path, source_img.stem + ".png").is_file()


def test_cli_tissue_mask_Otsu_dir(sample_all_wsis):
    """Test Otsu tissue masking for multiple files with default input CLI."""
    source_img = pathlib.Path(sample_all_wsis)
    runner = CliRunner()
    output_path = str(pathlib.Path(source_img, "tissue_mask"))
    tissue_mask_result = runner.invoke(
        cli.main,
        [
            "tissue-mask",
            "--img-input",
            str(source_img),
            "--method",
            "Otsu",
            "--mode",
            "save",
            "--output-path",
            output_path,
        ],
    )

    assert tissue_mask_result.exit_code == 0
    assert pathlib.Path(output_path, "test1.png").is_file()


def test_cli_tissue_mask_Morphological(sample_svs):
    """Test Morphological tissue masking with default input CLI."""
    source_img = pathlib.Path(sample_svs)
    runner = CliRunner()
    tissue_mask_result = runner.invoke(
        cli.main,
        [
            "tissue-mask",
            "--img-input",
            str(source_img),
            "--method",
            "Morphological",
        ],
    )

    assert tissue_mask_result.exit_code == 0

    output_path = str(pathlib.Path(sample_svs.parent, "tissue_mask"))
    tissue_mask_result = runner.invoke(
        cli.main,
        [
            "tissue-mask",
            "--img-input",
            str(source_img),
            "--method",
            "Morphological",
            "--mode",
            "save",
            "--output-path",
            output_path,
        ],
    )

    assert tissue_mask_result.exit_code == 0
    assert pathlib.Path(output_path, source_img.stem + ".png").is_file()

    tissue_mask_result = runner.invoke(
        cli.main,
        [
            "tissue-mask",
            "--img-input",
            str(source_img),
            "--method",
            "Morphological",
            "--mode",
            "save",
            "--output-path",
            output_path,
            "--resolution",
            1.25,
            "--units",
            "power",
        ],
    )

    assert tissue_mask_result.exit_code == 0
    assert pathlib.Path(output_path, source_img.stem + ".png").is_file()

    tissue_mask_result = runner.invoke(
        cli.main,
        [
            "tissue-mask",
            "--img-input",
            str(source_img),
            "--method",
            "Morphological",
            "--mode",
            "save",
            "--output-path",
            output_path,
            "--resolution",
            32,
            "--units",
            "mpp",
        ],
    )

    assert tissue_mask_result.exit_code == 0
    assert pathlib.Path(output_path, source_img.stem + ".png").is_file()

    tissue_mask_result = runner.invoke(
        cli.main,
        [
            "tissue-mask",
            "--img-input",
            str(source_img),
            "--method",
            "Morphological",
            "--mode",
            "save",
            "--output-path",
            output_path,
            "--kernel-size",
            "1",
            "1",
        ],
    )

    assert tissue_mask_result.exit_code == 0
    assert pathlib.Path(output_path, source_img.stem + ".png").is_file()


def test_cli_tissue_mask_method_not_supported(sample_svs):
    """Test method not supported for the tissue masking CLI."""
    source_img = pathlib.Path(sample_svs)
    runner = CliRunner()
    tissue_mask_result = runner.invoke(
        cli.main,
        [
            "tissue-mask",
            "--img-input",
            str(source_img),
            "--method",
            "Test",
        ],
    )

    assert tissue_mask_result.output == ""
    assert tissue_mask_result.exit_code == 1
    assert isinstance(tissue_mask_result.exception, MethodNotSupported)

    tissue_mask_result = runner.invoke(
        cli.main,
        [
            "tissue-mask",
            "--img-input",
            str(source_img),
            "--method",
            "Morphological",
            "--resolution",
            32,
            "--units",
            "Test",
        ],
    )

    assert tissue_mask_result.output == ""
    assert tissue_mask_result.exit_code == 1
    assert isinstance(tissue_mask_result.exception, MethodNotSupported)


def test_cli_tissue_mask_file_not_found_error(source_image):
    """Test file not found error for the tissue masking CLI."""
    source_img = pathlib.Path(source_image)
    runner = CliRunner()
    tissue_mask_result = runner.invoke(
        cli.main,
        [
            "tissue-mask",
            "--img-input",
            str(source_img)[:-1],
            "--method",
            "Otsu",
        ],
    )

    assert tissue_mask_result.output == ""
    assert tissue_mask_result.exit_code == 1
    assert isinstance(tissue_mask_result.exception, FileNotFoundError)
