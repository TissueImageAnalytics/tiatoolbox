"""Tests for code related to saving image tiles."""

import os
import pathlib

from click.testing import CliRunner

from tiatoolbox import cli

# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_save_tiles(sample_all_wsis2, tmp_path):
    """Test for save_tiles CLI."""
    runner = CliRunner()
    save_tiles_result = runner.invoke(
        cli.main,
        [
            "save-tiles",
            "--img-input",
            str(pathlib.Path(sample_all_wsis2)),
            "--file-types",
            "*.ndpi, *.svs, *.jp2",
            "--tile-objective-value",
            "5",
            "--output-path",
            os.path.join(tmp_path, "all_tiles"),
        ],
    )

    tmp_path = pathlib.Path(tmp_path)
    cmu_small_region = tmp_path / "all_tiles" / "CMU-1-Small-Region.svs"
    bioformatspull2759 = tmp_path / "all_tiles" / "bioformatspull2759.ndpi"
    test1jp2 = tmp_path / "all_tiles" / "test1.jp2"
    assert save_tiles_result.exit_code == 0
    assert (cmu_small_region / "Output.csv").exists()
    assert (cmu_small_region / "slide_thumbnail.jpg").exists()
    assert (cmu_small_region / "Tile_5_0_0.jpg").exists()

    assert (bioformatspull2759 / "Output.csv").exists()
    assert (bioformatspull2759 / "slide_thumbnail.jpg").exists()
    assert (bioformatspull2759 / "Tile_5_0_0.jpg").exists()
    assert (test1jp2 / "Output.csv").exists()
    assert (test1jp2 / "slide_thumbnail.jpg").exists()
    assert (test1jp2 / "Tile_5_0_0.jpg").exists()


def test_command_line_save_tiles_single_file(sample_svs, tmp_path):
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
            "False",
        ],
    )

    assert save_svs_tiles_result.exit_code == 0
    assert (
        pathlib.Path(tmp_path)
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )


def test_command_line_save_tiles_file_not_found(sample_svs, tmp_path):
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
