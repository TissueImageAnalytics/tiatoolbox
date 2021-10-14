#!/usr/bin/env python

"""Pytests for `tiatoolbox` package."""
from click.testing import CliRunner

from tiatoolbox import __version__, cli
from tiatoolbox.cli import (
    patch_predictor,
    read_bounds,
    save_tiles,
    slide_info,
    slide_thumbnail,
    tissue_mask,
)

# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_help_interface():
    """Test the CLI help."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert help_result.output == result.output


def test_command_line_version():
    """Test for version check."""
    runner = CliRunner()
    version_result = runner.invoke(cli.main, ["-V"])
    assert __version__ in version_result.output


def test_cli_patch_predictor():
    runner = CliRunner()
    patch_predictor_result = runner.invoke(patch_predictor.main)
    assert patch_predictor_result.exit_code == 0


def test_cli_read_bounds():
    runner = CliRunner()
    read_bounds_result = runner.invoke(read_bounds.main)
    assert read_bounds_result.exit_code == 0


def test_cli_save_tiles():
    runner = CliRunner()
    save_tiles_result = runner.invoke(save_tiles.main)
    assert save_tiles_result.exit_code == 0


def test_cli_slide_info():
    runner = CliRunner()
    slide_info_result = runner.invoke(slide_info.main)
    assert slide_info_result.exit_code == 0


def test_cli_slide_thumbnail():
    runner = CliRunner()
    slide_thumbnail_result = runner.invoke(slide_thumbnail.main)
    assert slide_thumbnail_result.exit_code == 0


def test_cli_tissue_mask():
    runner = CliRunner()
    tissue_mask_result = runner.invoke(tissue_mask.main)
    assert tissue_mask_result.exit_code == 0
