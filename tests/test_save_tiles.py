"""Test for code related to saving image tiles."""

from pathlib import Path

from click.testing import CliRunner

from tiatoolbox import cli

# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_save_tiles(sample_svs_ndpi_wsis: Path, tmp_path: Path) -> None:
    """Test for save_tiles CLI."""
    runner = CliRunner()
    save_tiles_result = runner.invoke(
        cli.main,
        [
            "save-tiles",
            "--img-input",
            str(Path(sample_svs_ndpi_wsis)),
            "--file-types",
            "*.ndpi, *.svs",
            "--tile-objective-value",
            "5",
            "--output-path",
            str(tmp_path / "all_tiles"),
        ],
    )

    tmp_path = Path(tmp_path)
    cmu_small_region = tmp_path / "all_tiles" / "CMU-1-Small-Region.svs"
    bioformatspull2759 = tmp_path / "all_tiles" / "bioformatspull2759.ndpi"

    assert save_tiles_result.exit_code == 0
    assert (cmu_small_region / "Output.csv").exists()
    assert (cmu_small_region / "slide_thumbnail.jpg").exists()
    assert (cmu_small_region / "Tile_5_0_0.jpg").exists()

    assert (bioformatspull2759 / "Output.csv").exists()
    assert (bioformatspull2759 / "slide_thumbnail.jpg").exists()
    assert (bioformatspull2759 / "Tile_5_0_0.jpg").exists()


def test_command_line_save_tiles_single_file(sample_svs: Path, tmp_path: Path) -> None:
    """Test for save_tiles CLI single file."""
    runner = CliRunner()
    save_svs_tiles_result = runner.invoke(
        cli.main,
        [
            "save-tiles",
            "--img-input",
            str(sample_svs),
            "--file-types",
            "*.ndpi, *.svs",
            "--tile-objective-value",
            "5",
            "--output-path",
            tmp_path,
            "--verbose",
            "True",
        ],
    )

    assert save_svs_tiles_result.exit_code == 0
    assert (
        Path(tmp_path)
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        Path(tmp_path)
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        Path(tmp_path)
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )


def test_command_line_save_tiles_file_not_found(
    sample_svs: Path,
    tmp_path: Path,
) -> None:
    """Test for save_tiles CLI file not found error."""
    runner = CliRunner()
    save_svs_tiles_result = runner.invoke(
        cli.main,
        [
            "save-tiles",
            "--img-input",
            str(sample_svs)[:-1],
            "--file-types",
            "*.ndpi, *.svs",
            "--tile-objective-value",
            "5",
            "--output-path",
            tmp_path,
        ],
    )

    assert save_svs_tiles_result.output == ""
    assert save_svs_tiles_result.exit_code == 1
    assert isinstance(save_svs_tiles_result.exception, FileNotFoundError)
