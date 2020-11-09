#!/usr/bin/env python

"""Pytests for `tiatoolbox` package."""
from tiatoolbox import cli
from tiatoolbox import __version__

from click.testing import CliRunner


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
