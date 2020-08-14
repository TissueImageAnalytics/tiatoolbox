#!/usr/bin/env python

"""Tests for `tiatoolbox` package."""
import pytest

# from tiatoolbox.dataloader.slide_info import slide_info
# from tiatoolbox.dataloader.save_tiles import save_tiles
# from tiatoolbox.dataloader import wsireader
# from tiatoolbox import utils
# from tiatoolbox.utils.exceptions import FileNotSupported
# from tiatoolbox import cli
# from tiatoolbox import __version__
#
# from click.testing import CliRunner
import requests
import os
import pathlib
# import numpy as np


@pytest.fixture(scope="session")
def _response_ndpi(request):
    """
    Sample pytest fixture for ndpi images
    Download ndpi image for pytest
    """
    ndpi_file_path = pathlib.Path(__file__).parent.joinpath("CMU-1.ndpi")
    if not pathlib.Path.is_file(ndpi_file_path):
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
        if pathlib.Path.is_file(ndpi_file_path):
            os.remove(str(ndpi_file_path))

    request.addfinalizer(close_ndpi)
    return _response_ndpi


@pytest.fixture(scope="session")
def _response_svs(request):
    """
    Sample pytest fixture for svs images
    Download ndpi image for pytest
    """
    svs_file_path = pathlib.Path(__file__).parent.joinpath("CMU-1-Small-Region.svs")
    if not pathlib.Path.is_file(svs_file_path):
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
        if pathlib.Path.is_file(svs_file_path):
            os.remove(str(svs_file_path))

    request.addfinalizer(close_svs)
    return _response_svs


@pytest.fixture(scope="session")
def _response_jp2(request):
    """
    Sample pytest fixture for svs images
    Download ndpi image for pytest
    """
    jp2_file_path = pathlib.Path(__file__).parent.joinpath("test1.jp2")
    if not pathlib.Path.is_file(jp2_file_path):
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
        if pathlib.Path.is_file(jp2_file_path):
            os.remove(str(jp2_file_path))

    request.addfinalizer(close_jp2)
    return _response_jp2


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------



