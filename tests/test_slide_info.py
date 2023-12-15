"""Test for code related to obtaining slide information."""

from pathlib import Path

from click.testing import CliRunner

from tiatoolbox import cli

# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_slide_info(sample_all_wsis: Path, tmp_path: Path) -> None:
    """Test the Slide information CLI."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img-input",
            str(Path(sample_all_wsis)),
            "--mode",
            "save",
            "--file-types",
            "*.ndpi, *.svs",
            "--output-path",
            str(tmp_path),
            "--verbose",
            "True",
        ],
    )

    assert slide_info_result.exit_code == 0
    assert Path(tmp_path, "CMU-1-Small-Region.yaml").exists()
    assert Path(tmp_path, "CMU-1.yaml").exists()
    assert not Path(tmp_path, "test1.yaml").exists()


def test_command_line_slide_info_jp2(sample_all_wsis: Path) -> None:
    """Test the Slide information CLI JP2, svs."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img-input",
            str(Path(sample_all_wsis)),
            "--mode",
            "save",
        ],
    )

    output_dir = Path(sample_all_wsis).parent
    assert slide_info_result.exit_code == 0
    assert Path(output_dir, "meta-data", "CMU-1-Small-Region.yaml").exists()
    assert Path(output_dir, "meta-data", "CMU-1.yaml").exists()
    assert Path(
        output_dir,
        "meta-data",
        "CMU-1-Small-Region.omnyx.yaml",
    ).exists()


def test_command_line_slide_info_svs(sample_svs: Path) -> None:
    """Test CLI slide info for single file."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img-input",
            sample_svs,
            "--file-types",
            "*.ndpi, *.svs",
            "--mode",
            "show",
            "--verbose",
            "True",
        ],
    )

    assert slide_info_result.exit_code == 0


def test_command_line_slide_info_file_not_found(sample_svs: Path) -> None:
    """Test CLI slide info file not found error."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img-input",
            str(sample_svs)[:-1],
            "--file-types",
            "*.ndpi, *.svs",
            "--mode",
            "show",
        ],
    )

    assert slide_info_result.output == ""
    assert slide_info_result.exit_code == 1
    assert isinstance(slide_info_result.exception, FileNotFoundError)


def test_command_line_slide_info_output_none_mode_save(sample_svs: Path) -> None:
    """Test CLI slide info for single file."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img-input",
            str(sample_svs),
            "--file-types",
            "*.ndpi, *.svs",
            "--mode",
            "save",
            "--verbose",
            "False",
        ],
    )

    assert slide_info_result.exit_code == 0
    assert Path(
        sample_svs.parent,
        "meta-data",
        "CMU-1-Small-Region.yaml",
    ).exists()


def test_command_line_slide_info_no_input() -> None:
    """Test CLI slide info for single file."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--file-types",
            "*.ndpi, *.svs",
            "--mode",
            "save",
        ],
    )

    assert "No image input provided." in slide_info_result.output
    assert slide_info_result.exit_code != 0
