"""Tests for code related to obtaining slide thumbnails."""

import os
import pathlib

import numpy as np
import pytest
from click.testing import CliRunner

from tiatoolbox import cli
from tiatoolbox.utils.env_detection import running_on_ci
from tiatoolbox.wsicore import wsireader


def test_wsireader_get_thumbnail_openslide(sample_svs):
    """Test for get_thumbnail as a python function."""
    wsi = wsireader.OpenSlideWSIReader(sample_svs)
    slide_thumbnail = wsi.slide_thumbnail()
    assert isinstance(slide_thumbnail, np.ndarray)
    assert slide_thumbnail.dtype == "uint8"


def test_wsireader_get_thumbnail_jp2(sample_jp2):
    """Test for get_thumbnail as a python function."""
    wsi = wsireader.OmnyxJP2WSIReader(sample_jp2)
    slide_thumbnail = wsi.slide_thumbnail()
    assert isinstance(slide_thumbnail, np.ndarray)
    assert slide_thumbnail.dtype == "uint8"


def command_line_slide_thumbnail(runner, sample, tmp_path, mode="save"):
    """Command line slide thumbnail helper."""
    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--img-input",
            str(pathlib.Path(sample)),
            "--mode",
            mode,
            "--output-path",
            str(pathlib.Path(tmp_path)),
        ],
    )

    assert slide_thumb_result.exit_code == 0
    if mode == "save":
        assert (pathlib.Path(tmp_path) / (sample.stem + ".jpg")).is_file()


def test_command_line_slide_thumbnail(sample_ndpi, tmp_path):
    """Test for the slide_thumbnail CLI."""
    runner = CliRunner()

    command_line_slide_thumbnail(runner, sample=sample_ndpi, tmp_path=tmp_path)


def test_command_line_slide_thumbnail_output_none(sample_svs, tmp_path):
    """Test cli slide thumbnail with output dir None."""
    runner = CliRunner()
    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--img-input",
            str(pathlib.Path(sample_svs)),
            "--mode",
            "save",
        ],
    )

    assert slide_thumb_result.exit_code == 0
    assert (
        pathlib.Path(sample_svs).parent / "slide-thumbnail" / (sample_svs.stem + ".jpg")
    ).is_file()


def test_command_line_jp2_slide_thumbnail(sample_jp2, tmp_path):
    """Test for the jp2 slide_thumbnail CLI."""
    runner = CliRunner()

    command_line_slide_thumbnail(runner, sample=sample_jp2, tmp_path=tmp_path)


@pytest.mark.skipif(running_on_ci(), reason="No display on CI.")
@pytest.mark.skipif(
    not os.environ.get("SHOW_TESTS"),
    reason="Visual tests disabled, set SHOW_TESTS to enable.",
)
def test_command_line_jp2_slide_thumbnail_mode_show(sample_jp2, tmp_path):
    """Test for the jp2 slide_thumbnail CLI mode='show'."""
    runner = CliRunner()

    command_line_slide_thumbnail(
        runner, sample=sample_jp2, tmp_path=tmp_path, mode="show"
    )


def test_command_line_jp2_slide_thumbnail_file_not_supported(sample_jp2, tmp_path):
    """Test for the jp2 slide_thumbnail CLI."""
    runner = CliRunner()

    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--img-input",
            str(pathlib.Path(sample_jp2))[:-1],
            "--mode",
            "save",
            "--output-path",
            str(pathlib.Path(tmp_path)),
        ],
    )

    assert slide_thumb_result.output == ""
    assert slide_thumb_result.exit_code == 1
    assert isinstance(slide_thumb_result.exception, FileNotFoundError)
