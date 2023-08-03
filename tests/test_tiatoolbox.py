#!/usr/bin/env python
"""Pytests for `tiatoolbox` package."""
from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

from tiatoolbox import __version__, cli
from tiatoolbox.cli.common import (
    cli_auto_generate_mask,
    cli_merge_predictions,
    cli_output_path,
    cli_return_labels,
    cli_return_probabilities,
    tiatoolbox_cli,
)

if TYPE_CHECKING:
    from pathlib import Path

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


@tiatoolbox_cli.command()
@cli_return_probabilities()
@cli_merge_predictions()
@cli_return_labels()
@cli_auto_generate_mask()
@cli_output_path()
def func_default_param(
    return_probabilities: bool,
    return_labels: bool,
    merge_predictions: bool,
    output_path: str,
    auto_generate_mask: bool,
) -> None:
    """Function to test cli with default None values."""
    from tiatoolbox.utils import save_yaml

    out_dict = {
        "return_probabilities": not return_probabilities,
        "return_labels": return_labels,
        "merge_predictions": merge_predictions,
        "auto_generate_mask": not auto_generate_mask,
    }

    save_yaml(input_dict=out_dict, output_path=output_path)


def test_cli_defaults(tmp_path: Path):
    """Check if the default bool values are correctly returned.

    The test checks if the default bool values are identified
    if None is passed as an argument.

    """
    import yaml

    cli.main.add_command(func_default_param)
    runner = CliRunner()

    yaml_output = tmp_path / "test.yaml"

    _ = runner.invoke(
        cli.main,
        [
            "func-default-param",
            "--output-path",
            str(yaml_output),
        ],
    )

    with yaml_output.open("r") as file:
        out = yaml.safe_load(file)

    assert all(out.values())
