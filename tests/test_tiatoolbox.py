#!/usr/bin/env python
"""Pytests for `tiatoolbox` package."""

from __future__ import annotations

from click.testing import CliRunner

from tiatoolbox import __version__, cli

# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_help_interface() -> None:
    """Test the CLI help."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "Computational pathology toolbox by TIA Centre." in help_result.output


def test_command_line_version() -> None:
    """Test for version check."""
    runner = CliRunner()
    version_result = runner.invoke(cli.main, ["-v"])
    assert __version__ in version_result.output
    version_result = runner.invoke(cli.main, ["--version"])
    assert __version__ in version_result.output
