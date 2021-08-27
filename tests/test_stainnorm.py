"""Tests for stain normalisation code."""

from tiatoolbox.utils.misc import imread
from tiatoolbox.tools.stainnorm import get_normaliser
from tiatoolbox.tools import stainextract
from tiatoolbox import cli
from tiatoolbox.utils.exceptions import MethodNotSupported

import pathlib
import numpy as np
from click.testing import CliRunner
import pytest


def test_stain_extract():
    """Test stain extraction class."""
    stain_matrix = np.array([0.65, 0.70, 0.29])
    with pytest.raises(ValueError):
        _ = stainextract.CustomExtractor(stain_matrix)


def test_vectors_in_right_direction():
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


def test_h_e_in_correct_order():
    """Test if H&E vectors are returned in the correct order."""
    v1 = np.ones(3)
    v2 = np.zeros(3)
    he = stainextract.h_and_e_in_right_order(v1, v2)
    assert np.all(he == np.array([v1, v2]))

    he = stainextract.h_and_e_in_right_order(v2, v1)
    assert np.all(he == np.array([v1, v2]))


def test_dl_output_for_h_and_e():
    """Test if correct value for H and E from dictionary learning output is returned."""
    dictionary = np.zeros([20, 15])
    dictionary1 = stainextract.dl_output_for_h_and_e(dictionary=dictionary)

    assert np.all(dictionary1 == dictionary)
    dictionary[1, :] = 1
    dictionary2 = stainextract.dl_output_for_h_and_e(dictionary=dictionary)

    assert dictionary2.shape == (2, 15)
    assert np.all(dictionary2 == dictionary[[1, 0], :])


def test_reinhard_normalise(_source_image, _norm_reinhard):
    """Test for Reinhard colour normalisation."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(pathlib.Path(_source_image))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    reinhard_img = imread(pathlib.Path(_norm_reinhard))

    norm = get_normaliser("reinhard")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(reinhard_img / 255.0 - transform / 255.0)) < 1e-2


def test_custom_normalise(_source_image, _norm_ruifrok):
    """Test for stain normalisation with user-defined stain matrix."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(pathlib.Path(_source_image))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    custom_img = imread(pathlib.Path(_norm_ruifrok))

    # init class with custom method - test with ruifrok stain matrix
    stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
    norm = get_normaliser("custom", stain_matrix=stain_matrix)
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(custom_img / 255.0 - transform / 255.0)) < 1e-2


def test_get_normaliser_assertion():
    """Test get normaliser assertion error."""
    stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
    with pytest.raises(ValueError):
        _ = get_normaliser("ruifrok", stain_matrix)


def test_ruifrok_normalise(_source_image, _norm_ruifrok):
    """Test for stain normalisation with stain matrix from Ruifrok and Johnston."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(pathlib.Path(_source_image))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    ruifrok_img = imread(pathlib.Path(_norm_ruifrok))

    # init class with Ruifrok & Johnston method
    norm = get_normaliser("ruifrok")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(ruifrok_img / 255.0 - transform / 255.0)) < 1e-2


def test_macenko_normalise(_source_image, _norm_macenko):
    """Test for stain normalisation with stain matrix from Macenko et al."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(pathlib.Path(_source_image))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    macenko_img = imread(pathlib.Path(_norm_macenko))

    # init class with Macenko method
    norm = get_normaliser("macenko")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(macenko_img / 255.0 - transform / 255.0)) < 1e-2


def test_vahadane_normalise(_source_image, _norm_vahadane):
    """Test for stain normalisation with stain matrix from Vahadane et al."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(pathlib.Path(_source_image))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    vahadane_img = imread(pathlib.Path(_norm_vahadane))

    # init class with Vahadane method
    norm = get_normaliser("vahadane")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(vahadane_img / 255.0 - transform / 255.0)) < 1e-2


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_stainnorm(_source_image):
    """Test for the stain normalisation CLI."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = pathlib.Path(_source_image)
    target_img = file_parent_dir.joinpath("../data/target_image.png")
    runner = CliRunner()
    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stain-norm",
            "--source_input",
            source_img,
            "--target_input",
            target_img,
            "--method",
            "reinhard",
        ],
    )

    assert stainnorm_result.exit_code == 0

    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stain-norm",
            "--source_input",
            source_img,
            "--target_input",
            target_img,
            "--method",
            "ruifrok",
        ],
    )

    assert stainnorm_result.exit_code == 0

    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stain-norm",
            "--source_input",
            source_img,
            "--target_input",
            target_img,
            "--method",
            "macenko",
        ],
    )

    assert stainnorm_result.exit_code == 0

    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stain-norm",
            "--source_input",
            source_img,
            "--target_input",
            target_img,
            "--method",
            "vahadane",
        ],
    )

    assert stainnorm_result.exit_code == 0


def test_cli_stainnorm_dir(_source_image):
    """Test directory input for the stain normalisation CLI."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = _source_image.dirname
    target_img = file_parent_dir.joinpath("../data/target_image.png")
    runner = CliRunner()
    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stain-norm",
            "--source_input",
            str(source_img),
            "--target_input",
            target_img,
            "--method",
            "vahadane",
        ],
    )

    assert stainnorm_result.exit_code == 0


def test_cli_stainnorm_file_not_found_error(_source_image):
    """Test file not found error for the stain normalisation CLI."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = pathlib.Path(_source_image)
    target_img = file_parent_dir.joinpath("../data/target_image.png")
    runner = CliRunner()
    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stain-norm",
            "--source_input",
            str(source_img)[:-1],
            "--target_input",
            target_img,
            "--method",
            "vahadane",
        ],
    )

    assert stainnorm_result.output == ""
    assert stainnorm_result.exit_code == 1
    assert isinstance(stainnorm_result.exception, FileNotFoundError)


def test_cli_stainnorm_method_not_supported(_source_image):
    """Test method not supported for the stain normalisation CLI."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = pathlib.Path(_source_image)
    target_img = file_parent_dir.joinpath("../data/target_image.png")
    runner = CliRunner()
    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stain-norm",
            "--source_input",
            str(source_img),
            "--target_input",
            target_img,
            "--method",
            "Test",
        ],
    )

    assert stainnorm_result.output == ""
    assert stainnorm_result.exit_code == 1
    assert isinstance(stainnorm_result.exception, MethodNotSupported)
