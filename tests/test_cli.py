"""Tests for cli inputs."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from tiatoolbox.cli.common import (
    cli_class_dict,
    cli_input_resolutions,
    cli_output_resolutions,
    parse_bool_list,
    prepare_ioconfig,
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


class FakeIOConfig:
    """Minimal stand-in for a ModelIOConfigABC subclass."""

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize a FakeIOConfig object."""
        self.kwargs = kwargs


def test_prepare_ioconfig_with_pretrained_weights(
    track_tmp_path: Path,
) -> None:
    """Test the branch where ``pretrained_weights`` is provided.

    This test verifies that:
    - the YAML file is read,
    - the parsed YAML is passed into ``config_class``,
    - a config object is returned instead of ``None``.

    Args:
        track_tmp_path (Path):
            Temporary directory used to create the YAML fixture file.

    Returns:
        None:
            Assertions validate the expected behavior.

    """
    yaml_path = track_tmp_path / "config.yaml"
    yaml_path.write_text(
        "patch_input_shape: [224, 224]\n"
        "stride_shape: [112, 112]\n"
        "input_resolutions:\n"
        "  - units: mpp\n"
        "    resolution: 0.5\n",
        encoding="utf-8",
    )

    ioconfig = prepare_ioconfig(
        config_class=FakeIOConfig,
        pretrained_weights=track_tmp_path / "weights.pth",
        yaml_config_path=yaml_path,
    )

    assert isinstance(ioconfig, FakeIOConfig)
    assert ioconfig.kwargs["patch_input_shape"] == [224, 224]
    assert ioconfig.kwargs["stride_shape"] == [112, 112]
    assert ioconfig.kwargs["input_resolutions"][0]["units"] == "mpp"
    assert ioconfig.kwargs["input_resolutions"][0]["resolution"] == 0.5


def _fake_grab_files_from_dir(
    input_path: Path,
    file_types: tuple[str, ...],
) -> list:
    """Fake grab_files_from_dir function."""
    _ = input_path, file_types
    msg = "grab_files_from_dir should not be called"
    raise AssertionError(msg)


def _fake_string_to_tuple(in_str: str) -> tuple[str, ...]:
    """Fake string to tuple."""
    return tuple(part.strip() for part in in_str.split(","))


def test_prepare_model_cli_with_input_dir_and_mask_dir(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test ``prepare_model_cli`` when both input and masks are directories.

    This test covers:
    - ``if masks.is_dir():``
    - ``if Path.is_dir(img_input):``

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Pytest fixture used to replace filesystem helper functions.
        track_tmp_path (Path):
            Temporary directory used to create input, mask, and output fixtures.

    Returns:
        None:
            Assertions verify the expected behavior.

    """
    img_dir = track_tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "sample_1.png").write_bytes(b"fake image")
    (img_dir / "sample_2.png").write_bytes(b"fake image")

    mask_dir = track_tmp_path / "masks"
    mask_dir.mkdir()
    (mask_dir / "mask_1.png").write_bytes(b"fake mask")
    (mask_dir / "mask_2.jpg").write_bytes(b"fake mask")

    output_path = track_tmp_path / "output"
    file_types = "*.png, *.jpg"

    def _fake_grab_files_from_dir_for_dirs(
        input_path: Path,
        file_types: tuple[str, ...],
    ) -> list:
        """Fake grab_files_from_dir function."""
        _ = file_types
        if input_path == img_dir:
            return [img_dir / "sample_1.png", img_dir / "sample_2.png"]
        if input_path == mask_dir:
            return [mask_dir / "mask_1.png", mask_dir / "mask_2.jpg"]
        msg = f"Unexpected path: {input_path}"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "tiatoolbox.utils.misc.grab_files_from_dir",
        _fake_grab_files_from_dir_for_dirs,
    )
    monkeypatch.setattr(
        "tiatoolbox.utils.misc.string_to_tuple",
        _fake_string_to_tuple,
    )

    files_all, masks_all, returned_output = prepare_model_cli(
        img_input=img_dir,
        output_path=output_path,
        masks=mask_dir,
        file_types=file_types,
    )

    assert files_all == [img_dir / "sample_1.png", img_dir / "sample_2.png"]
    assert masks_all == [mask_dir / "mask_1.png", mask_dir / "mask_2.jpg"]
    assert returned_output == output_path


def test_prepare_model_cli_with_single_input_and_mask_file(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test ``prepare_model_cli`` when the mask is provided as a file.

    This test covers:
    - ``if masks.is_file():``

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Pytest fixture used to replace filesystem helper functions.
        track_tmp_path (Path):
            Temporary directory used to create input, mask, and output fixtures.

    Returns:
        None:
            Assertions verify the expected behavior.

    """
    img_file = track_tmp_path / "sample.png"
    img_file.write_bytes(b"fake image")

    mask_file = track_tmp_path / "mask.png"
    mask_file.write_bytes(b"fake mask")

    output_path = track_tmp_path / "output"

    monkeypatch.setattr(
        "tiatoolbox.utils.misc.grab_files_from_dir",
        _fake_grab_files_from_dir,
    )
    monkeypatch.setattr(
        "tiatoolbox.utils.misc.string_to_tuple",
        _fake_string_to_tuple,
    )

    files_all, masks_all, returned_output = prepare_model_cli(
        img_input=img_file,
        output_path=output_path,
        masks=mask_file,
        file_types="*.png",
    )

    assert files_all == [img_file]
    assert masks_all == [mask_file]
    assert returned_output == output_path
