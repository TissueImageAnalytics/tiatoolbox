from tiatoolbox.dataloader.slide_info import slide_info
from tiatoolbox import utils
from tiatoolbox import cli

import pathlib
from click.testing import CliRunner


def test_slide_info(_sample_all_wsis, tmp_path):
    """Test for slide_info as a python function."""
    file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
    files_all = utils.misc.grab_files_from_dir(
        input_path=_sample_all_wsis, file_types=file_types,
    )

    for curr_file in files_all:
        slide_param = slide_info(input_path=curr_file, verbose=True)
        out_path = tmp_path / slide_param.file_path.with_suffix(".yaml").name
        utils.misc.save_yaml(slide_param.as_dict(), out_path)


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_slide_info(_sample_all_wsis):
    """Test the Slide information CLI."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--wsi_input",
            str(pathlib.Path(_sample_all_wsis)),
            "--file_types",
            '"*.ndpi, *.svs"',
            "--mode",
            "save",
        ],
    )

    assert slide_info_result.exit_code == 0

    file_types = "*.svs"
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_sample_all_wsis)), file_types=file_types,
    )
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--wsi_input",
            files_all[0],
            "--file_types",
            '"*.ndpi, *.svs"',
            "--mode",
            "show",
        ],
    )

    assert slide_info_result.exit_code == 0
