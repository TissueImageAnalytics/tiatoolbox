#!/usr/bin/env python

"""Tests for `tiatoolbox` metadata."""
import pathlib

import pytest

from tiatoolbox import utils
from tiatoolbox.dataloader import wsireader, wsimeta


def test_wsimeta_init_fail():
    with pytest.raises(TypeError):
        wsimeta.WSIMeta()


def test_wsimeta_validate_fail():
    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512), level_dimensions=[])
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(
        slide_dimensions=(512, 512),
        level_dimensions=[(512, 512), (256, 256)],
        level_count=3,
    )
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512), level_downsamples=[1, 2],)
    assert meta.validate() is False


def test_wsimeta_validate_init_pass():
    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512))
    assert meta.validate()

    meta = wsimeta.WSIMeta(
        slide_dimensions=(512, 512),
        level_dimensions=[(512, 512), (256, 256)],
        level_downsamples=[1, 2],
    )
    assert meta.validate()


def test_wsimeta_openslidewsireader_ndpi(_response_ndpi, tmp_path):
    file_types = ("*.ndpi",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(files_all[0]))
    wsi_obj = wsireader.OpenSlideWSIReader(input_dir, file_name + ext)
    meta = wsi_obj.slide_info
    assert meta.validate()


def test_wsimeta_openslidewsireader_svs(_response_svs, tmp_path):
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(files_all[0]))
    wsi_obj = wsireader.OpenSlideWSIReader(input_dir, file_name + ext)
    meta = wsi_obj.slide_info
    assert meta.validate()
