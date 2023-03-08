#!/usr/bin/env python

"""Pytests for `tiatoolbox` package."""
import logging

from click.testing import CliRunner

from tiatoolbox import __version__, cli

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
    assert "Computational pathology toolbox by TIA Centre." in help_result.output


def test_command_line_version():
    """Test for version check."""
    runner = CliRunner()
    version_result = runner.invoke(cli.main, ["-v"])
    assert __version__ in version_result.output
    version_result = runner.invoke(cli.main, ["--version"])
    assert __version__ in version_result.output


def test_logger_output(capsys):
    """Tests if logger is writing output to correct value."""
    from tiatoolbox import logger

    logger.setLevel(logging.DEBUG)
    logger.debug("Test if debug is written to stdout.")

    logger.setLevel(logging.INFO)
    logger.info("Test if info written to stdout.")

    logger.setLevel(logging.WARNING)
    logger.warning("Test if warning written to stderr.")

    logger.setLevel(logging.ERROR)
    logger.error("Test if error is written to stderr.")

    logger.setLevel(logging.CRITICAL)
    logger.critical("Test if critical is written to stderr.")

    print("not working")
    capsys
