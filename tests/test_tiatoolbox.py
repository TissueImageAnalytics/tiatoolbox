#!/usr/bin/env python

"""Tests for `tiatoolbox` package."""
import pytest

from tiatoolbox.dataloader.slide_info import slide_info
from tiatoolbox.dataloader import wsireader
from tiatoolbox import utils
from tiatoolbox import cli
from tiatoolbox import __version__

from click.testing import CliRunner
import requests
import os
import pathlib
import numpy as np


@pytest.fixture
def _response_ndpi(request):
    """
    Sample pytest fixture for ndpi images
    Download ndpi image for pytest
    """
    ndpi_file_path = pathlib.Path(__file__).parent.joinpath("CMU-1.ndpi")
    if not pathlib.Path.is_file(ndpi_file_path):
        r = requests.get(
            "http://openslide.cs.cmu.edu/download/openslide-testdata"
            "/Hamamatsu/CMU-1.ndpi"
        )
        with open(ndpi_file_path, "wb") as f:
            f.write(r.content)

    def close_ndpi():
        if pathlib.Path.is_file(ndpi_file_path):
            os.remove(str(ndpi_file_path))

    request.addfinalizer(close_ndpi)
    return _response_ndpi


@pytest.fixture
def _response_svs(request):
    """
    Sample pytest fixture for svs images
    Download ndpi image for pytest
    """
    svs_file_path = pathlib.Path(__file__).parent.joinpath("CMU-1.svs")
    if not pathlib.Path.is_file(svs_file_path):
        r = requests.get(
            "http://openslide.cs.cmu.edu/download/openslide-testdata"
            "/Aperio/CMU-1.svs"
        )
        with open(svs_file_path, "wb") as f:
            f.write(r.content)

    def close_ndpi():
        if pathlib.Path.is_file(svs_file_path):
            os.remove(str(svs_file_path))

    request.addfinalizer(close_ndpi)
    return _response_svs


def test_slide_info(_response_ndpi, _response_svs):
    """pytest for slide_info as a python function"""
    file_types = ("*.ndpi", "*.svs", "*.mrxs")
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(r".")), file_types=file_types,
    )
    slide_params = slide_info(input_path=files_all, workers=2)

    for slide_param in slide_params:
        utils.misc.save_yaml(slide_param, slide_param["file_name"] + ".yaml")


def test_wsireader_slide_info(_response_svs):
    """pytest for slide_info as a python function"""
    wsi_obj = wsireader.WSIReader(str(pathlib.Path(r".")), "CMU-1.svs")
    slide_param = wsi_obj.slide_info()
    utils.misc.save_yaml(slide_param, slide_param["file_name"] + ".yaml")


def test_wsireader_read_region(_response_ndpi, _response_svs):
    """pytest for slide_info as a python function"""
    wsi_obj = wsireader.WSIReader(str(pathlib.Path(r".")), "CMU-1.svs")
    level = 0
    region = [13000, 17000, 15000, 19000]
    im_region = wsi_obj.read_region(region[0], region[1], region[2], region[3], level)
    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype=='uint8'
    assert im_region.shape==(2000, 2000, 4)


def test_command_line_help_interface():
    """Test the CLI help"""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert help_result.output == result.output


def test_command_line_version():
    """pytest for version check"""
    runner = CliRunner()
    version_result = runner.invoke(cli.main, ["-V"])
    assert __version__ in version_result.output


def test_command_line_slide_info(_response_ndpi, _response_svs):
    """Test the Slide information CLI."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--wsi_input",
            ".",
            "--file_types",
            '"*.ndpi, *.svs"',
            "--workers",
            "2",
        ],
    )

    assert slide_info_result.exit_code == 0
