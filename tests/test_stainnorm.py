"""Test for stain normalization code."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from tiatoolbox import cli
from tiatoolbox.data import _local_sample_path, stain_norm_target
from tiatoolbox.tools import stainextract
from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.utils import imread


def test_stain_extract() -> None:
    """Test stain extraction class."""
    stain_matrix = np.array([0.65, 0.70, 0.29])
    with pytest.raises(
        ValueError,
        match=r"Stain matrix must have shape \(2, 3\) or \(3, 3\).",
    ):
        _ = stainextract.CustomExtractor(stain_matrix)


def test_vectors_in_right_direction() -> None:
    """Test if eigenvectors are corrected in the right direction."""
    e_vect = np.ones([2, 2])
    e_vect = stainextract.vectors_in_correct_direction(e_vectors=e_vect)
    assert np.all(e_vect == 1)

    e_vect = np.ones([2, 2])
    e_vect[0, 0] = -1
    e_vect = stainextract.vectors_in_correct_direction(e_vectors=e_vect)
    assert np.all(e_vect[:, 1] == 1)
    assert e_vect[0, 0] == 1
    assert e_vect[1, 0] == -1

    e_vect = np.ones([2, 2])
    e_vect[0, 1] = -1
    e_vect = stainextract.vectors_in_correct_direction(e_vectors=e_vect)
    assert np.all(e_vect[:, 0] == 1)
    assert e_vect[0, 1] == 1
    assert e_vect[1, 1] == -1


def test_h_e_in_correct_order() -> None:
    """Test if H&E vectors are returned in the correct order."""
    v1 = np.ones(3)
    v2 = np.zeros(3)
    he = stainextract.h_and_e_in_right_order(v1, v2)
    assert np.all(he == np.array([v1, v2]))

    he = stainextract.h_and_e_in_right_order(v1=v2, v2=v1)
    assert np.all(he == np.array([v1, v2]))


def test_dl_output_for_h_and_e() -> None:
    """Test if correct value for H and E from dictionary learning output is returned."""
    dictionary = np.zeros([20, 15])
    dictionary1 = stainextract.dl_output_for_h_and_e(dictionary=dictionary)

    assert np.all(dictionary1 == dictionary)
    dictionary[1, :] = 1
    dictionary2 = stainextract.dl_output_for_h_and_e(dictionary=dictionary)

    assert dictionary2.shape == (2, 15)
    assert np.all(dictionary2 == dictionary[[1, 0], :])


def test_reinhard_normalize(source_image: Path, norm_reinhard: Path) -> None:
    """Test for Reinhard colour normalization."""
    source_img = imread(Path(source_image))
    target_img = stain_norm_target()
    reinhard_img = imread(Path(norm_reinhard))

    norm = get_normalizer("reinhard")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(reinhard_img / 255.0 - transform / 255.0)) < 1e-2


def test_custom_normalize(source_image: Path, norm_ruifrok: Path) -> None:
    """Test for stain normalization with user-defined stain matrix."""
    source_img = imread(Path(source_image))
    target_img = stain_norm_target()
    custom_img = imread(Path(norm_ruifrok))

    # init class with custom method - test with ruifrok stain matrix
    stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
    norm = get_normalizer("custom", stain_matrix=stain_matrix)
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(custom_img / 255.0 - transform / 255.0)) < 1e-2


def test_get_normalizer_assertion() -> None:
    """Test get normalizer assertion error."""
    stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
    with pytest.raises(
        ValueError,
        match=r"`stain_matrix` is only defined when using `method_name`=\"custom\".",
    ):
        _ = get_normalizer("ruifrok", stain_matrix)


def test_get_custom_normalizer_assertion() -> None:
    """Test get custom normalizer assertion error."""
    stain_matrix = None
    with pytest.raises(
        ValueError,
        match=r"`stain_matrix` is None when using `method_name`=\"custom\".",
    ):
        _ = get_normalizer("custom", stain_matrix)


def test_ruifrok_normalize(source_image: Path, norm_ruifrok: Path) -> None:
    """Test for stain normalization with stain matrix from Ruifrok and Johnston."""
    source_img = imread(Path(source_image))
    target_img = stain_norm_target()
    ruifrok_img = imread(Path(norm_ruifrok))

    # init class with Ruifrok & Johnston method
    norm = get_normalizer("ruifrok")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(ruifrok_img / 255.0 - transform / 255.0)) < 1e-2


def test_macenko_normalize(source_image: Path, norm_macenko: Path) -> None:
    """Test for stain normalization with stain matrix from Macenko et al."""
    source_img = imread(Path(source_image))
    target_img = stain_norm_target()
    macenko_img = imread(Path(norm_macenko))

    # init class with Macenko method
    norm = get_normalizer("macenko")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(macenko_img / 255.0 - transform / 255.0)) < 1e-2


def test_vahadane_normalize(
    source_image: Path, norm_vahadane: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test for stain normalization with stain matrix from Vahadane et al."""
    source_img = imread(Path(source_image))
    target_img = stain_norm_target()
    vahadane_img = imread(Path(norm_vahadane))

    # init class with Vahadane method
    norm = get_normalizer("vahadane")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image
    assert "Vahadane stain extraction/normalization algorithms" in caplog.text
    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(vahadane_img / 255.0 - transform / 255.0)) < 1e-1


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_stainnorm(source_image: Path, tmp_path: Path) -> None:
    """Test for the stain normalization CLI."""
    source_img = Path(source_image)
    target_img = _local_sample_path("target_image.png")
    runner = CliRunner()
    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stain-norm",
            "--img-input",
            source_img,
            "--target-input",
            target_img,
            "--output-path",
            str(tmp_path / "stainnorm_output"),
            "--method",
            "reinhard",
        ],
    )

    assert stainnorm_result.exit_code == 0

    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stain-norm",
            "--img-input",
            source_img,
            "--target-input",
            target_img,
            "--output-path",
            str(tmp_path / "stainnorm_output"),
            "--method",
            "ruifrok",
        ],
    )

    assert stainnorm_result.exit_code == 0

    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stain-norm",
            "--img-input",
            source_img,
            "--target-input",
            target_img,
            "--output-path",
            str(tmp_path / "stainnorm_output"),
            "--method",
            "macenko",
        ],
    )

    assert stainnorm_result.exit_code == 0

    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stain-norm",
            "--img-input",
            source_img,
            "--target-input",
            target_img,
            "--output-path",
            str(tmp_path / "stainnorm_output"),
            "--method",
            "vahadane",
        ],
    )

    assert stainnorm_result.exit_code == 0


def test_cli_stainnorm_dir(source_image: Path, tmp_path: Path) -> None:
    """Test directory input for the stain normalization CLI."""
    source_img = source_image.parent
    target_img = _local_sample_path("target_image.png")
    runner = CliRunner()
    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stain-norm",
            "--img-input",
            str(source_img),
            "--target-input",
            target_img,
            "--output-path",
            str(tmp_path / "stainnorm_ouput"),
            "--method",
            "ruifrok",
        ],
    )

    assert stainnorm_result.exit_code == 0


def test_cli_stainnorm_file_not_found_error(source_image: Path, tmp_path: Path) -> None:
    """Test file not found error for the stain normalization CLI."""
    source_img = Path(source_image)
    target_img = stain_norm_target()
    runner = CliRunner()
    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stain-norm",
            "--img-input",
            str(source_img)[:-1],
            "--target-input",
            target_img,
            "--output-path",
            str(tmp_path / "stainnorm_output"),
            "--method",
            "vahadane",
        ],
    )

    assert stainnorm_result.output == ""
    assert stainnorm_result.exit_code == 1
    assert isinstance(stainnorm_result.exception, FileNotFoundError)


def test_cli_stainnorm_method_not_supported(source_image: Path, tmp_path: Path) -> None:
    """Test method not supported for the stain normalization CLI."""
    source_img = Path(source_image)
    target_img = stain_norm_target()
    runner = CliRunner()
    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stain-norm",
            "--img-input",
            str(source_img),
            "--target-input",
            target_img,
            "--output-path",
            str(tmp_path / "stainnorm_output"),
            "--method",
            "Test",
        ],
    )

    assert "Invalid value for '--method'" in stainnorm_result.output
    assert stainnorm_result.exit_code != 0
    assert isinstance(stainnorm_result.exception, SystemExit)
