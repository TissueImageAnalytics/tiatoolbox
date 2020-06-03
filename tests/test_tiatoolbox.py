#!/usr/bin/env python

"""Tests for `tiatoolbox` package."""
import pytest

from tiatoolbox.dataloader.slide_info import slide_info
from tiatoolbox.utils import misc_utils as misc
from tiatoolbox import cli
from tiatoolbox import __version__

from click.testing import CliRunner
import requests
import os
import pathlib


@pytest.fixture
def response_ndpi():
    """
    Sample pytest fixture for ndpi images
    Download ndpi image for pytest
    """
    if not os.path.isfile("./CMU-1.ndpi"):
        r = requests.get(
            "http://openslide.cs.cmu.edu/download/openslide-testdata/Hamamatsu/CMU-1.ndpi"
        )
        with open("./CMU-1.ndpi", "wb") as f:
            f.write(r.content)


@pytest.fixture
def response_svs():
    """
    Sample pytest fixture for svs images
    Download ndpi image for pytest
    """
    if not os.path.isfile("./CMU-1.svs"):
        r = requests.get(
            "http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs"
        )
        with open("./CMU-1.svs", "wb") as f:
            f.write(r.content)


def test_slide_info(response_ndpi, response_svs):
    """
    pytest for slide_info as a python function
    """
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
    file_types = ("*.ndpi", "*.svs", "*.mrxs")
    files_all = misc.grab_files_from_dir(
        input_path=str(pathlib.Path(r".")), file_types=file_types,
    )
    _ = slide_info(input_path=files_all, workers=2, mode="save")


def test_command_line_help_interface():
    """
    Test the CLI help
    """
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert help_result.output == result.output


def test_command_line_version():
    """
    pytest for version check
    """
    runner = CliRunner()
    version_result = runner.invoke(cli.main, ["-V"])
    assert __version__ in version_result.output


def test_command_line_slide_info(response_ndpi, response_svs):
    """
    Test the Slide information CLI.
    """
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--wsi_input",
            ".",
            "--file_types",
            '"*.ndpi, *.svs"',
            "--mode",
            "show",
            "--workers",
            "2",
        ],
    )

    assert slide_info_result.exit_code == 0
