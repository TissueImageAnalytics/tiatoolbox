from tiatoolbox.utils.misc import imread
from tiatoolbox.tools.stainnorm import get_normaliser
from tiatoolbox import cli
from tiatoolbox.utils.exceptions import MethodNotSupported

import pathlib
import numpy as np
from click.testing import CliRunner


def test_reinhard_normalise():
    """Test for Reinhard colour normalisation."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(file_parent_dir.joinpath("data/source_image.png"))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    reinhard_img = imread(file_parent_dir.joinpath("data/norm_reinhard.png"))

    norm = get_normaliser("reinhard")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(reinhard_img / 255.0 - transform / 255.0)) < 1e-2


def test_custom_normalise():
    """Test for stain normalisation with user-defined stain matrix."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(file_parent_dir.joinpath("data/source_image.png"))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    custom_img = imread(file_parent_dir.joinpath("data/norm_ruifrok.png"))

    # init class with custom method - test with ruifrok stain matrix
    stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
    norm = get_normaliser("custom", stain_matrix=stain_matrix)
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(custom_img / 255.0 - transform / 255.0)) < 1e-2


def test_ruifrok_normalise():
    """Test for stain normalisation with stain matrix from Ruifrok and Johnston."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(file_parent_dir.joinpath("data/source_image.png"))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    ruifrok_img = imread(file_parent_dir.joinpath("data/norm_ruifrok.png"))

    # init class with Ruifrok & Johnston method
    norm = get_normaliser("ruifrok")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(ruifrok_img / 255.0 - transform / 255.0)) < 1e-2


def test_macenko_normalise():
    """Test for stain normalisation with stain matrix from Macenko et al."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(file_parent_dir.joinpath("data/source_image.png"))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    macenko_img = imread(file_parent_dir.joinpath("data/norm_macenko.png"))

    # init class with Macenko method
    norm = get_normaliser("macenko")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(macenko_img / 255.0 - transform / 255.0)) < 1e-2


def test_vahadane_normalise():
    """Test for stain normalisation with stain matrix from Vahadane et al."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = imread(file_parent_dir.joinpath("data/source_image.png"))
    target_img = imread(file_parent_dir.joinpath("../data/target_image.png"))
    vahadane_img = imread(file_parent_dir.joinpath("data/norm_vahadane.png"))

    # init class with Vahadane method
    norm = get_normaliser("vahadane")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.mean(np.absolute(vahadane_img / 255.0 - transform / 255.0)) < 1e-2


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_stainnorm():
    """Test for the stain normalisation CLI."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = file_parent_dir.joinpath("data/source_image.png")
    target_img = file_parent_dir.joinpath("../data/target_image.png")
    runner = CliRunner()
    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stainnorm",
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
            "stainnorm",
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
            "stainnorm",
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
            "stainnorm",
            "--source_input",
            source_img,
            "--target_input",
            target_img,
            "--method",
            "vahadane",
        ],
    )

    assert stainnorm_result.exit_code == 0


def test_cli_stainnorm_dir():
    """Test directory input for the stain normalisation CLI."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = file_parent_dir.joinpath("data")
    target_img = file_parent_dir.joinpath("../data/target_image.png")
    runner = CliRunner()
    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stainnorm",
            "--source_input",
            str(source_img),
            "--target_input",
            target_img,
            "--method",
            "vahadane",
        ],
    )

    assert stainnorm_result.exit_code == 0


def test_cli_stainnorm_file_not_found_error():
    """Test file not found error for the stain normalisation CLI."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = file_parent_dir.joinpath("data/source_image.png")
    target_img = file_parent_dir.joinpath("../data/target_image.png")
    runner = CliRunner()
    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stainnorm",
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


def test_cli_stainnorm_method_not_supported():
    """Test method not supported for the stain normalisation CLI."""
    file_parent_dir = pathlib.Path(__file__).parent
    source_img = file_parent_dir.joinpath("data/source_image.png")
    target_img = file_parent_dir.joinpath("../data/target_image.png")
    runner = CliRunner()
    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stainnorm",
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
