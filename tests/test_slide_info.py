"""Tests for code related to obtaining slide information."""

from tiatoolbox.wsicore.slide_info import slide_info
from tiatoolbox import utils
from tiatoolbox import cli

import pathlib
from click.testing import CliRunner


def test_slide_info(_sample_all_wsis, tmp_path):
    """Test for slide_info as a python function."""
    file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
    files_all = utils.misc.grab_files_from_dir(
        input_path=_sample_all_wsis,
        file_types=file_types,
    )

    for curr_file in files_all:
        slide_param = slide_info(input_path=curr_file, verbose=True)
        out_path = tmp_path / slide_param.file_path.with_suffix(".yaml").name
        utils.misc.save_yaml(slide_param.as_dict(), out_path)


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_slide_info(_sample_all_wsis, tmp_path):
    """Test the Slide information CLI."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img_input",
            str(pathlib.Path(_sample_all_wsis)),
            "--mode",
            "save",
            "--file_types",
            "*.ndpi, *.svs",
            "--output_dir",
            str(tmp_path),
        ],
    )

    assert slide_info_result.exit_code == 0
    assert tmp_path.joinpath("CMU-1-Small-Region.yaml").exists()
    assert tmp_path.joinpath("CMU-1.yaml").exists()
    assert not tmp_path.joinpath("test1.yaml").exists()


def test_command_line_slide_info_jp2(_sample_all_wsis, tmp_path):
    """Test the Slide information CLI JP2, svs."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img_input",
            str(pathlib.Path(_sample_all_wsis)),
            "--mode",
            "save",
        ],
    )

    output_dir = pathlib.Path(_sample_all_wsis).parent
    assert slide_info_result.exit_code == 0
    assert output_dir.joinpath("meta/CMU-1-Small-Region.yaml").exists()
    assert output_dir.joinpath("meta/CMU-1.yaml").exists()
    assert output_dir.joinpath("meta/test1.yaml").exists()


def test_command_line_slide_info_svs(_sample_svs):
    """Test CLI slide info for single file."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img_input",
            _sample_svs,
            "--file_types",
            "*.ndpi, *.svs",
            "--mode",
            "show",
            "--verbose",
            "False",
        ],
    )

    assert slide_info_result.exit_code == 0


def test_command_line_slide_info_file_not_found(_sample_svs):
    """Test CLI slide info file not found error."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img_input",
            str(_sample_svs)[:-1],
            "--file_types",
            "*.ndpi, *.svs",
            "--mode",
            "show",
        ],
    )

    assert slide_info_result.output == ""
    assert slide_info_result.exit_code == 1
    assert isinstance(slide_info_result.exception, FileNotFoundError)


def test_command_line_slide_info_output_none_mode_save(_sample_svs):
    """Test CLI slide info for single file."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img_input",
            str(_sample_svs),
            "--file_types",
            "*.ndpi, *.svs",
            "--mode",
            "save",
        ],
    )

    assert slide_info_result.exit_code == 0
    assert (
        pathlib.Path(_sample_svs)
        .parent.parent.joinpath("meta/CMU-1-Small-Region.yaml")
        .exists()
    )
