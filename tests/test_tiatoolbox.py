#!/usr/bin/env python

"""Tests for `tiatoolbox` package."""
import pytest

from tiatoolbox.dataloader.slide_info import slide_info

# from tiatoolbox.dataloader.save_tiles import save_tiles
from tiatoolbox.dataloader import wsireader
from tiatoolbox import utils

# from tiatoolbox.utils.exceptions import FileNotSupported
from tiatoolbox import cli
from tiatoolbox import __version__

from click.testing import CliRunner
import requests
import os
import pathlib
import shutil

# import numpy as np


# -------------------------------------------------------------------------------------
# Pytest Fixtures
# -------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _response_ndpi(request, tmpdir_factory):
    """
    Sample pytest fixture for ndpi images
    Download ndpi image for pytest
    """
    ndpi_file_path = tmpdir_factory.mktemp("data").join("CMU-1.ndpi")
    if not os.path.isfile(ndpi_file_path):
        print("Downloading NDPI")
        r = requests.get(
            "http://openslide.cs.cmu.edu/download/openslide-testdata"
            "/Hamamatsu/CMU-1.ndpi"
        )
        with open(ndpi_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("Skipping NDPI")

    def close_ndpi():
        if ndpi_file_path.isfile():
            os.remove(str(ndpi_file_path))

    request.addfinalizer(close_ndpi)
    return ndpi_file_path


@pytest.fixture(scope="session")
def _response_svs(request, tmpdir_factory):
    """
    Sample pytest fixture for svs images
    Download ndpi image for pytest
    """
    svs_file_path = tmpdir_factory.mktemp("data").join("CMU-1-Small-Region.svs")
    if not os.path.isfile(svs_file_path):
        print("Downloading SVS")
        r = requests.get(
            "http://openslide.cs.cmu.edu/download/openslide-testdata"
            "/Aperio/CMU-1-Small-Region.svs"
        )
        with open(svs_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("Skipping SVS")

    def close_svs():
        if svs_file_path.isfile():
            os.remove(str(svs_file_path))

    request.addfinalizer(close_svs)
    return svs_file_path


@pytest.fixture(scope="session")
def _response_jp2(request, tmpdir_factory):
    """
    Sample pytest fixture for svs images
    Download ndpi image for pytest
    """
    jp2_file_path = tmpdir_factory.mktemp("data").join("test1.jp2")
    if not os.path.isfile(jp2_file_path):
        print("Downloading JP2")
        r = requests.get(
            "https://warwick.ac.uk/fac/sci/dcs/research/tia/tiatoolbox/files"
            "/test1.jp2"
        )
        with open(jp2_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("Skipping JP2")

    def close_jp2():
        if jp2_file_path.isfile():
            os.remove(str(jp2_file_path))

    request.addfinalizer(close_jp2)
    return jp2_file_path


@pytest.fixture(scope="session")
def _response_all_wsis(request, _response_ndpi, _response_svs, tmpdir_factory):
    dir_path = pathlib.Path(tmpdir_factory.mktemp("data"))

    try:
        dir_path.joinpath(_response_ndpi.basename).symlink_to(_response_ndpi)
        dir_path.joinpath(_response_svs.basename).symlink_to(_response_svs)
    except OSError:
        shutil.copy(_response_ndpi, dir_path.joinpath(_response_ndpi.basename))
        shutil.copy(_response_svs, dir_path.joinpath(_response_svs.basename))

    def close_all_wsi():
        if dir_path.is_dir():
            shutil.rmtree(dir_path)

    request.addfinalizer(close_all_wsi)

    return dir_path


# -------------------------------------------------------------------------------------
# Python API tests
# -------------------------------------------------------------------------------------


def test_slide_info(_response_all_wsis, tmp_path):
    """pytest for slide_info as a python function"""
    file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
    files_all = utils.misc.grab_files_from_dir(
        input_path=_response_all_wsis, file_types=file_types,
    )
    slide_params = slide_info(input_path=files_all, workers=2, verbose=True)

    for _, slide_param in enumerate(slide_params):
        utils.misc.save_yaml(
            slide_param.as_dict(), tmp_path / (slide_param.file_name + ".yaml")
        )

    unwrapped_slide_info = slide_info.__closure__[0].cell_contents
    utils.misc.save_yaml(
        unwrapped_slide_info(input_path=files_all[0], verbose=True),
        tmp_path / "test.yaml",
    )


def test_wsireader_slide_info(_response_svs, tmp_path):
    """pytest for slide_info in WSIReader class as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent), file_types=file_types,
    )
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(files_all[0]))
    wsi_obj = wsireader.OpenSlideWSIReader(input_dir, file_name + ext)
    slide_param = wsi_obj.slide_info
    utils.misc.save_yaml(
        slide_param.as_dict(), tmp_path / (slide_param.file_name + ".yaml")
    )


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


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
