"""Tests for cli inputs."""

import json
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from tiatoolbox.cli.common import (
    cli_class_dict,
    cli_input_resolutions,
    cli_output_resolutions,
    parse_bool_list,
    prepare_model_cli,
)


@click.command()
@cli_class_dict(default=None)
@cli_input_resolutions(default=None)
@cli_output_resolutions(default=None)
def predictor_specific_inputs(
    class_dict: dict[int, str] | None,
    input_resolutions: list[dict] | None,
    output_resolutions: list[dict] | None,
) -> None:
    """Helper to test predictor specific inputs."""
    click.echo((class_dict, input_resolutions, output_resolutions))


def test_cli_class_dict() -> None:
    """Test CLI class dictionary input."""
    runner = CliRunner()
    result = runner.invoke(
        predictor_specific_inputs, ["--class-dict", '{"1": "tumour", "2": "normal"}']
    )
    assert result.exit_code == 0
    assert "{1: 'tumour', 2: 'normal'}" in result.output


def test_cli_input_resolutions() -> None:
    """Test CLI input resolutions list of dicts."""
    runner = CliRunner()
    # Pass a JSON list of dicts
    resolutions = [
        {"units": "mpp", "resolution": 0.25},
        {"units": "level", "resolution": 1},
    ]
    result = runner.invoke(
        predictor_specific_inputs, ["--input-resolutions", json.dumps(resolutions)]
    )

    assert (
        "(None, [{'units': 'mpp', 'resolution': 0.25}, {'units': 'level',"
    ) in result.output
    output = result.output.strip()
    # Check that our resolutions appear in the output
    assert "'units': 'mpp'" in output
    assert "'resolution': 0.25" in output
    assert "'units': 'level'" in output
    assert "'resolution': 1" in output
    # And class_dict should be None
    assert "None" in output


def test_cli_both_options() -> None:
    """Test both CLI options together."""
    runner = CliRunner()
    resolutions = [{"units": "mpp", "resolution": 0.25}]
    result = runner.invoke(
        predictor_specific_inputs,
        [
            "--class-dict",
            '{"1": "tumour"}',
            "--input-resolutions",
            json.dumps(resolutions),
        ],
    )

    assert result.exit_code == 0
    # predictor_specific_inputs should echo the tuple (class_dict, input_resolutions)
    output = result.output.strip()

    # Check that the class_dict appears in the output
    assert "'tumour'" in output
    # Check that the resolutions appear in the output
    assert "'units': 'mpp'" in output
    assert "'resolution': 0.25" in output


def test_cli_invalid_json() -> None:
    """Test invalid JSON raises error."""
    runner = CliRunner()
    result = runner.invoke(
        predictor_specific_inputs,
        ["--class-dict", "{invalid json}"],
    )

    assert result.exit_code != 0
    # The error message should be in the output
    assert "Invalid JSON" in result.output

    result = runner.invoke(
        predictor_specific_inputs,
        ["--input-resolutions", "{invalid json}"],
    )

    assert result.exit_code != 0
    # The error message should be in the output
    assert "Invalid JSON" in result.output

    result = runner.invoke(
        predictor_specific_inputs,
        ["--output-resolutions", "{invalid json}"],
    )

    assert result.exit_code != 0
    # The error message should be in the output
    assert "Invalid JSON" in result.output


def test_cli_output_resolutions() -> None:
    """Test CLI output resolutions list of dicts."""
    runner = CliRunner()
    resolutions = [{"units": "mpp", "resolution": 0.5}]
    result = runner.invoke(
        predictor_specific_inputs,
        ["--output-resolutions", json.dumps(resolutions)],
    )
    assert result.exit_code == 0
    output = result.output.strip()
    assert "'units': 'mpp'" in output
    assert "'resolution': 0.5" in output


@pytest.mark.parametrize("option", ["--input-resolutions", "--output-resolutions"])
def test_cli_resolutions_not_list(option: str) -> None:
    """Test that non-list JSON raises BadParameter."""
    runner = CliRunner()
    # Pass a dict instead of a list (valid JSON, wrong type)
    bad_value = '{"units": "mpp", "resolution": 0.25}'
    result = runner.invoke(
        predictor_specific_inputs,
        [option, bad_value],
    )
    assert result.exit_code != 0
    assert "Must be a JSON list of dictionaries" in result.output


def test_parse_bool_list_none() -> None:
    """parse_bool_list should return None when value is None."""
    result = parse_bool_list(_ctx=None, _param=None, value=None)
    assert result is None


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        ("true,false", (True, False)),
        ("1,0", (True, False)),
        ("yes,no", (True, False)),
        ("y,n", (True, False)),
        (" true , 0 , YES ", (True, False, True)),
    ],
)
def test_parse_bool_list_valid(
    input_str: str,
    expected: tuple[bool, ...],
) -> None:
    """parse_bool_list should correctly parse valid boolean lists."""
    result = parse_bool_list(_ctx=None, _param=None, value=input_str)
    assert result == expected


@pytest.mark.parametrize("bad_value", ["foo", "true,bar", "1,2", "yes,maybe"])
def test_parse_bool_list_invalid(bad_value: str) -> None:
    """parse_bool_list should raise BadParameter on invalid tokens."""
    with pytest.raises(click.BadParameter):
        parse_bool_list(_ctx=None, _param=None, value=bad_value)


def test_output_path_exists_raises() -> None:
    """Ensure FileExistsError when the output_path already exists on disk."""
    img_input = Path("input.jpg")
    output_path = Path("out")
    masks = None

    # First call: output_path.exists() → True
    # Second call: img_input.exists() → True (never reached)
    with (
        patch.object(Path, "exists", side_effect=[True, True]),
        pytest.raises(FileExistsError),
    ):
        prepare_model_cli(img_input, output_path, masks, "*.jpg")


def test_img_input_not_found_raises() -> None:
    """Ensure FileNotFoundError when the img_input path does not exist."""
    img_input = Path("missing.jpg")
    output_path = Path("out")
    masks = None

    # output_path.exists() → False
    # img_input.exists() → False
    with (
        patch.object(Path, "exists", side_effect=[False, False]),
        pytest.raises(FileNotFoundError),
    ):
        prepare_model_cli(img_input, output_path, masks, "*.jpg")


def test_masks_is_file() -> None:
    """Verify that when masks is a file a list containing that mask file is returned."""
    img_input = Path("input.jpg")
    output_path = Path("out")
    masks = Path("mask.png")

    # output_path.exists() → False
    # img_input.exists() → True
    with (
        patch.object(Path, "exists", side_effect=[False, True]),
        patch.object(Path, "is_file", return_value=True),
        patch.object(Path, "is_dir", return_value=False),
    ):
        files, masks_all, out = prepare_model_cli(
            img_input, output_path, masks, "*.jpg"
        )

    assert files == [img_input]
    assert masks_all == [masks]
    assert out == output_path
