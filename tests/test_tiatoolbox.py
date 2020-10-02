#!/usr/bin/env python

"""Tests for `tiatoolbox` package."""
import pytest

from tiatoolbox.dataloader.slide_info import slide_info

from tiatoolbox.dataloader.save_tiles import save_tiles
from tiatoolbox.dataloader import wsireader, wsimeta
from tiatoolbox import utils

from tiatoolbox.utils.exceptions import FileNotSupported
from tiatoolbox import cli
from tiatoolbox import __version__

from click.testing import CliRunner
import requests
import os
import pathlib
import numpy as np
import shutil


# -------------------------------------------------------------------------------------
# Pytest Fixtures
# -------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _response_ndpi(tmpdir_factory):
    """
    Sample pytest fixture for ndpi images
    Download ndpi image for pytest
    """
    ndpi_file_path = tmpdir_factory.mktemp("data").join("CMU-1.ndpi")
    if not os.path.isfile(ndpi_file_path):
        print("\nDownloading NDPI")
        r = requests.get(
            "http://openslide.cs.cmu.edu/download/openslide-testdata"
            "/Hamamatsu/CMU-1.ndpi"
        )
        with open(ndpi_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping NDPI")

    return ndpi_file_path


@pytest.fixture(scope="session")
def _response_svs(tmpdir_factory):
    """
    Sample pytest fixture for svs images
    Download ndpi image for pytest
    """
    svs_file_path = tmpdir_factory.mktemp("data").join("CMU-1-Small-Region.svs")
    if not os.path.isfile(svs_file_path):
        print("\nDownloading SVS")
        r = requests.get(
            "http://openslide.cs.cmu.edu/download/openslide-testdata"
            "/Aperio/CMU-1-Small-Region.svs"
        )
        with open(svs_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping SVS")

    return svs_file_path


@pytest.fixture(scope="session")
def _response_jp2(tmpdir_factory):
    """
    Sample pytest fixture for svs images
    Download ndpi image for pytest
    """
    jp2_file_path = tmpdir_factory.mktemp("data").join("test1.jp2")
    if not os.path.isfile(jp2_file_path):
        print("\nDownloading JP2")
        r = requests.get(
            "https://warwick.ac.uk/fac/sci/dcs/research/tia/tiatoolbox/files"
            "/test1.jp2"
        )
        with open(jp2_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("\nSkipping JP2")

    return jp2_file_path


@pytest.fixture(scope="session")
def _response_all_wsis(_response_ndpi, _response_svs, _response_jp2, tmpdir_factory):
    dir_path = pathlib.Path(tmpdir_factory.mktemp("data"))

    try:
        dir_path.joinpath(_response_ndpi.basename).symlink_to(_response_ndpi)
        dir_path.joinpath(_response_svs.basename).symlink_to(_response_svs)
        dir_path.joinpath(_response_jp2.basename).symlink_to(_response_jp2)
    except OSError:
        shutil.copy(_response_ndpi, dir_path.joinpath(_response_ndpi.basename))
        shutil.copy(_response_svs, dir_path.joinpath(_response_svs.basename))
        shutil.copy(_response_jp2, dir_path.joinpath(_response_jp2.basename))

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

    for curr_file in files_all:
        slide_param = slide_info(input_path=curr_file, verbose=True)
        out_path = tmp_path / slide_param.file_path.with_suffix(".yaml").name
        utils.misc.save_yaml(slide_param.as_dict(), out_path)


def test_wsireader_slide_info(_response_svs, tmp_path):
    """pytest for slide_info in WSIReader class as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent), file_types=file_types,
    )
    wsi_obj = wsireader.OpenSlideWSIReader(files_all[0])
    slide_param = wsi_obj.slide_info
    out_path = tmp_path / slide_param.file_path.with_suffix(".yaml").name
    utils.misc.save_yaml(slide_param.as_dict(), out_path)


def test_wsireader_read_region(_response_svs):
    """pytest for read region as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent), file_types=file_types,
    )
    wsi_obj = wsireader.OpenSlideWSIReader(files_all[0])
    level = 0
    region = [13000, 17000, 15000, 19000]
    im_region = wsi_obj.read_region(region[0], region[1], region[2], region[3], level)
    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (2000, 2000, 3)


def test_wsireader_slide_thumbnail(_response_svs):
    """pytest for slide_thumbnail as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent), file_types=file_types,
    )
    wsi_obj = wsireader.OpenSlideWSIReader(files_all[0])
    slide_thumbnail = wsi_obj.slide_thumbnail()
    assert isinstance(slide_thumbnail, np.ndarray)
    assert slide_thumbnail.dtype == "uint8"


def test_wsireader_save_tiles(_response_svs, tmp_path):
    """pytest for save_tiles in wsireader as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent), file_types=file_types,
    )
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(files_all[0]))
    wsi_obj = wsireader.OpenSlideWSIReader(
        files_all[0],
        output_dir=str(pathlib.Path(tmp_path).joinpath("test_wsireader_save_tiles")),
        tile_objective_value=5,
    )
    wsi_obj.save_tiles(verbose=True)
    assert (
        pathlib.Path(tmp_path)
        .joinpath("test_wsireader_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("test_wsireader_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("test_wsireader_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )


def test_save_tiles(_response_all_wsis, tmp_path):
    """pytest for save_tiles as a python function"""
    file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_all_wsis)), file_types=file_types,
    )

    for curr_file in files_all:
        save_tiles(
            input_path=curr_file,
            tile_objective_value=5,
            output_dir=str(pathlib.Path(tmp_path).joinpath("tiles_save_tiles")),
            verbose=True,
        )

    assert (
        pathlib.Path(tmp_path)
        .joinpath("tiles_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("tiles_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("tiles_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("tiles_save_tiles")
        .joinpath("CMU-1.ndpi")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("tiles_save_tiles")
        .joinpath("CMU-1.ndpi")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("tiles_save_tiles")
        .joinpath("CMU-1.ndpi")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )


def test_wsireader_jp2_save_tiles(_response_jp2, tmp_path):
    """pytest for save_tiles in wsireader as a python function"""
    wsi_obj = wsireader.OmnyxJP2WSIReader(
        _response_jp2,
        output_dir=str(
            pathlib.Path(tmp_path).joinpath("test_wsireader_jp2_save_tiles")
        ),
        tile_objective_value=5,
    )
    wsi_obj.save_tiles(verbose=True)
    assert (
        pathlib.Path(tmp_path)
        .joinpath("test_wsireader_jp2_save_tiles")
        .joinpath("test1.jp2")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("test_wsireader_jp2_save_tiles")
        .joinpath("test1.jp2")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        pathlib.Path(tmp_path)
        .joinpath("test_wsireader_jp2_save_tiles")
        .joinpath("test1.jp2")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )


def test_exception_tests():
    with pytest.raises(FileNotSupported):
        utils.misc.save_yaml(
            slide_info(input_path="/mnt/test/sample.txt", verbose=True).as_dict(),
            "test.yaml",
        )

    with pytest.raises(FileNotSupported):
        save_tiles(
            input_path="/mnt/test/sample.txt",
            tile_objective_value=5,
            output_dir=str(pathlib.Path(__file__).parent.joinpath("tiles_save_tiles")),
            verbose=True,
        )


def test_imresize():
    """pytest for imresize"""
    img = np.zeros((2000, 2000, 3))
    resized_img = utils.transforms.imresize(img, 0.5)
    assert resized_img.shape == (1000, 1000, 3)


def test_background_composite():
    """pytest for background composite"""
    new_im = np.zeros((2000, 2000, 4)).astype("uint8")
    new_im[:1000, :, 3] = 255
    im = utils.transforms.background_composite(new_im)
    assert np.all(im[1000:, :, :] == 255)
    assert np.all(im[:1000, :, :] == 0)


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


def test_command_line_slide_info(_response_all_wsis):
    """Test the Slide information CLI."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--wsi_input",
            str(pathlib.Path(_response_all_wsis)),
            "--file_types",
            '"*.ndpi, *.svs"',
            "--mode",
            "save",
        ],
    )

    assert slide_info_result.exit_code == 0

    file_types = "*.svs"
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_all_wsis)), file_types=file_types,
    )
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--wsi_input",
            files_all[0],
            "--file_types",
            '"*.ndpi, *.svs"',
            "--mode",
            "show",
        ],
    )

    assert slide_info_result.exit_code == 0


def test_command_line_read_region(_response_ndpi, tmp_path):
    """Test the Read Region CLI."""
    runner = CliRunner()
    read_region_result = runner.invoke(
        cli.main,
        [
            "read-region",
            "--wsi_input",
            str(pathlib.Path(_response_ndpi)),
            "--level",
            "0",
            "--mode",
            "save",
            "--region",
            "0",
            "0",
            "2000",
            "2000",
            "--output_path",
            str(pathlib.Path(tmp_path).joinpath("im_region.jpg")),
        ],
    )

    assert read_region_result.exit_code == 0
    assert os.path.isfile(str(pathlib.Path(tmp_path).joinpath("im_region.jpg")))

    read_region_result = runner.invoke(
        cli.main,
        [
            "read-region",
            "--wsi_input",
            str(pathlib.Path(_response_ndpi)),
            "--level",
            "0",
            "--mode",
            "save",
            "--output_path",
            str(pathlib.Path(tmp_path).joinpath("im_region2.jpg")),
        ],
    )

    assert read_region_result.exit_code == 0
    assert os.path.isfile(str(pathlib.Path(tmp_path).joinpath("im_region2.jpg")))


def test_command_line_slide_thumbnail(_response_ndpi, tmp_path):
    """pytest for the slide_thumbnail CLI."""
    runner = CliRunner()
    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--wsi_input",
            str(pathlib.Path(_response_ndpi)),
            "--mode",
            "save",
            "--output_path",
            str(pathlib.Path(tmp_path).joinpath("slide_thumb.jpg")),
        ],
    )

    assert slide_thumb_result.exit_code == 0
    assert pathlib.Path(tmp_path).joinpath("slide_thumb.jpg").is_file()


def test_command_line_save_tiles(_response_all_wsis, tmp_path):
    """pytest for save_tiles CLI."""
    runner = CliRunner()
    save_tiles_result = runner.invoke(
        cli.main,
        [
            "save-tiles",
            "--wsi_input",
            str(pathlib.Path(_response_all_wsis)),
            "--file_types",
            '"*.ndpi, *.svs"',
            "--tile_objective_value",
            "5",
            "--output_dir",
            os.path.join(tmp_path, "all_tiles"),
        ],
    )

    assert save_tiles_result.exit_code == 0

    file_types = "*.svs"
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_all_wsis)), file_types=file_types,
    )
    save_svs_tiles_result = runner.invoke(
        cli.main,
        [
            "save-tiles",
            "--wsi_input",
            files_all[0],
            "--file_types",
            '"*.ndpi, *.svs"',
            "--tile_objective_value",
            "5",
            "--output_dir",
            tmp_path,
        ],
    )

    assert save_svs_tiles_result.exit_code == 0


def test_wsimeta_init_fail():
    with pytest.raises(TypeError):
        wsimeta.WSIMeta(slide_dimensions=None)


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

    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512), level_downsamples=[1, 2],)
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512))
    meta.level_dimensions = None
    assert meta.validate() is False

    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512))
    meta.level_downsamples = None
    assert meta.validate() is False


def test_wsimeta_validate_pass():
    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512))
    assert meta.validate()

    meta = wsimeta.WSIMeta(
        slide_dimensions=(512, 512),
        level_dimensions=[(512, 512), (256, 256)],
        level_downsamples=[1, 2],
    )
    assert meta.validate()


def test_wsimeta_openslidewsireader_ndpi(_response_ndpi, tmp_path):
    wsi_obj = wsireader.OpenSlideWSIReader(_response_ndpi)
    meta = wsi_obj.slide_info
    assert meta.validate()


def test_wsimeta_openslidewsireader_svs(_response_svs, tmp_path):
    wsi_obj = wsireader.OpenSlideWSIReader(_response_svs)
    meta = wsi_obj.slide_info
    assert meta.validate()
