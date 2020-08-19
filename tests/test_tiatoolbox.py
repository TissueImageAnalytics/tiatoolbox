#!/usr/bin/env python

"""Tests for `tiatoolbox` package."""
import pytest

from tiatoolbox.dataloader.slide_info import slide_info
from tiatoolbox.dataloader.save_tiles import save_tiles
from tiatoolbox.dataloader import wsireader
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
    svs_file_path = pathlib.Path(__file__).parent.joinpath("CMU-1-Small-Region.svs")
    if not pathlib.Path.is_file(svs_file_path):
        r = requests.get(
            "http://openslide.cs.cmu.edu/download/openslide-testdata"
            "/Aperio/CMU-1-Small-Region.svs"
        )
        with open(svs_file_path, "wb") as f:
            f.write(r.content)

    def close_ndpi():
        if pathlib.Path.is_file(svs_file_path):
            os.remove(str(svs_file_path))

    request.addfinalizer(close_ndpi)
    return _response_svs


@pytest.fixture
def _response_jp2(request):
    """
    Sample pytest fixture for svs images
    Download ndpi image for pytest
    """
    jp2_file_path = pathlib.Path(__file__).parent.joinpath("test1.jp2")
    if not pathlib.Path.is_file(jp2_file_path):
        r = requests.get(
            "https://warwick.ac.uk/fac/sci/dcs/research/tia/tiatoolbox/files"
            "/test1.jp2"
        )
        with open(jp2_file_path, "wb") as f:
            f.write(r.content)

    def close_jp2():
        if pathlib.Path.is_file(jp2_file_path):
            os.remove(str(jp2_file_path))

    request.addfinalizer(close_jp2)
    return _response_jp2


def test_slide_info(_response_ndpi, _response_svs):
    """pytest for slide_info as a python function"""
    file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    slide_params = slide_info(input_path=files_all, workers=2, verbose=True)

    for _, slide_param in enumerate(slide_params):
        utils.misc.save_yaml(slide_param.as_dict(), slide_param.file_name + ".yaml")

    unwrapped_slide_info = slide_info.__closure__[0].cell_contents
    unwrapped_slide_info(input_path=files_all[0], verbose=True)


def test_wsireader_slide_info(_response_svs):
    """pytest for slide_info in WSIReader class as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(files_all[0]))
    wsi_obj = wsireader.OpenSlideWSIReader(input_dir, file_name + ext)
    slide_param = wsi_obj.slide_info
    utils.misc.save_yaml(slide_param.as_dict(), slide_param.file_name + ".yaml")


def test_wsireader_read_region(_response_svs):
    """pytest for read region as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(files_all[0]))
    wsi_obj = wsireader.OpenSlideWSIReader(input_dir, file_name + ext)
    level = 0
    region = [13000, 17000, 15000, 19000]
    im_region = wsi_obj.read_region(region[0], region[1], region[2], region[3], level)
    im_region = im_region[:, :, 0:3]
    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (2000, 2000, 3)


def test_wsireader_openslide_thumbnail(_response_svs):
    """pytest for slide_thumbnail as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(files_all[0]))
    wsi_obj = wsireader.OpenSlideWSIReader(input_dir, file_name + ext)
    slide_thumbnail = wsi_obj.slide_thumbnail()
    assert isinstance(slide_thumbnail, np.ndarray)
    assert slide_thumbnail.dtype == "uint8"


def test_wsireader_jp2slide_thumbnail(_response_jp2):
    """pytest for slide_thumbnail as a python function"""
    file_types = ("*.jp2",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(files_all[0]))
    wsi_obj = wsireader.OmnyxJP2WSIReader(input_dir, file_name + ext)
    slide_thumbnail = wsi_obj.slide_thumbnail()
    assert isinstance(slide_thumbnail, np.ndarray)
    assert slide_thumbnail.dtype == "uint8"


def test_imresize():
    """pytest for imresize"""
    img = np.zeros((2000, 2000, 3))
    resized_img = utils.transforms.imresize(img, 0.5)
    assert resized_img.shape == (1000, 1000, 3)


def test_background_composite():
    """pytest for background composit"""
    new_im = np.zeros((2000, 2000, 4)).astype("uint8")
    new_im[:1000, :, 3] = 255
    im = utils.transforms.background_composite(new_im)
    assert np.all(im[1000:, :, :] == 255)
    assert np.all(im[:1000, :, :] == 0)


def test_wsireader_save_tiles(_response_svs):
    """pytest for save_tiles in wsireader as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(files_all[0]))
    wsi_obj = wsireader.OpenSlideWSIReader(
        input_dir,
        file_name + ext,
        output_dir=str(
            pathlib.Path(__file__).parent.joinpath("test_wsireader_save_tiles")
        ),
        tile_objective_value=5,
    )
    wsi_obj.save_tiles(verbose=True)
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("test_wsireader_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("test_wsireader_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("test_wsireader_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )
    shutil.rmtree(pathlib.Path(__file__).parent.joinpath("test_wsireader_save_tiles"))


def test_save_tiles(_response_ndpi, _response_svs):
    """pytest for save_tiles as a python function"""
    file_types = ("*.ndpi", "*.svs", "*.mrxs")
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    save_tiles(
        input_path=files_all,
        workers=2,
        tile_objective_value=5,
        output_dir=str(pathlib.Path(__file__).parent.joinpath("tiles_save_tiles")),
        verbose=True,
    )
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("tiles_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("tiles_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("tiles_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("tiles_save_tiles")
        .joinpath("CMU-1.ndpi")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("tiles_save_tiles")
        .joinpath("CMU-1.ndpi")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("tiles_save_tiles")
        .joinpath("CMU-1.ndpi")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )
    shutil.rmtree(pathlib.Path(__file__).parent.joinpath("tiles_save_tiles"))


def test_save_tiles_unwrap(_response_svs):
    file_types = "*.svs"
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    unwrapped_save_tiles = save_tiles.__closure__[0].cell_contents
    unwrapped_save_tiles(
        input_path=files_all[0],
        tile_objective_value=5,
        output_dir=str(pathlib.Path(__file__).parent.joinpath("tiles_save_tiles")),
        verbose=True,
    )
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("tiles_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("tiles_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("tiles_save_tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )
    shutil.rmtree(pathlib.Path(__file__).parent.joinpath("tiles_save_tiles"))


def test_save_tiles_jp2_unwrap(_response_jp2):
    file_types = "*.jp2"
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    unwrapped_save_tiles = save_tiles.__closure__[0].cell_contents
    unwrapped_save_tiles(
        input_path=files_all[0],
        tile_objective_value=5,
        output_dir=str(
            pathlib.Path(__file__).parent.joinpath("test_save_tiles_jp2_unwrap")
        ),
        verbose=True,
    )
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("test_save_tiles_jp2_unwrap")
        .joinpath("test1.jp2")
        .joinpath("Output.csv")
        .exists()
    )
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("test_save_tiles_jp2_unwrap")
        .joinpath("test1.jp2")
        .joinpath("slide_thumbnail.jpg")
        .exists()
    )
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("test_save_tiles_jp2_unwrap")
        .joinpath("test1.jp2")
        .joinpath("Tile_5_0_0.jpg")
        .exists()
    )
    shutil.rmtree(pathlib.Path(__file__).parent.joinpath("test_save_tiles_jp2_unwrap"))


def test_exception_tests():
    unwrapped_slide_info = slide_info.__closure__[0].cell_contents
    with pytest.raises(FileNotSupported):
        utils.misc.save_yaml(
            unwrapped_slide_info(input_path="/mnt/test/sample.txt", verbose=True),
            "test.yaml",
        )

    unwrapped_save_tiles = save_tiles.__closure__[0].cell_contents
    with pytest.raises(FileNotSupported):
        unwrapped_save_tiles(
            input_path="/mnt/test/sample.txt",
            tile_objective_value=5,
            output_dir=str(pathlib.Path(__file__).parent.joinpath("tiles_save_tiles")),
            verbose=True,
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


def test_command_line_slide_info(_response_ndpi, _response_svs):
    """Test the Slide information CLI."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--wsi_input",
            str(pathlib.Path(__file__).parent),
            "--file_types",
            "*.ndpi, *.svs",
            "--workers",
            "2",
            "--mode",
            "save",
        ],
    )

    assert slide_info_result.exit_code == 0

    file_types = "*.svs"
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--wsi_input",
            files_all[0],
            "--file_types",
            "*.ndpi, *.svs",
            "--workers",
            "2",
            "--mode",
            "show",
        ],
    )

    assert slide_info_result.exit_code == 0


def test_command_line_read_region(_response_ndpi):
    """Test the Read Region CLI."""
    runner = CliRunner()
    read_region_result = runner.invoke(
        cli.main,
        [
            "read-region",
            "--wsi_input",
            str(pathlib.Path(__file__).parent.joinpath("CMU-1.ndpi")),
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
            str(pathlib.Path(__file__).parent.joinpath("im_region.jpg")),
        ],
    )

    assert read_region_result.exit_code == 0
    assert os.path.isfile(str(pathlib.Path(__file__).parent.joinpath("im_region.jpg")))

    read_region_result = runner.invoke(
        cli.main,
        [
            "read-region",
            "--wsi_input",
            str(pathlib.Path(__file__).parent.joinpath("CMU-1.ndpi")),
            "--level",
            "0",
            "--mode",
            "save",
            "--output_path",
            str(pathlib.Path(__file__).parent.joinpath("im_region2.jpg")),
        ],
    )

    assert read_region_result.exit_code == 0
    assert os.path.isfile(str(pathlib.Path(__file__).parent.joinpath("im_region2.jpg")))


def test_command_line_slide_thumbnail(_response_ndpi):
    """Test the Slide Thumbnail CLI."""
    runner = CliRunner()
    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--wsi_input",
            str(pathlib.Path(__file__).parent.joinpath("CMU-1.ndpi")),
            "--mode",
            "save",
            "--output_path",
            str(pathlib.Path(__file__).parent.joinpath("slide_thumb.jpg")),
        ],
    )

    assert slide_thumb_result.exit_code == 0
    assert pathlib.Path(__file__).parent.joinpath("slide_thumb.jpg").is_file()


def test_command_line_slide_thumbnail_jp2(_response_jp2):
    """Test the Slide Thumbnail CLI."""
    runner = CliRunner()
    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--wsi_input",
            str(pathlib.Path(__file__).parent.joinpath("test1.jp2")),
            "--mode",
            "save",
            "--output_path",
            str(pathlib.Path(__file__).parent.joinpath("slide_thumbjp2.jpg")),
        ],
    )

    assert slide_thumb_result.exit_code == 0
    assert pathlib.Path(__file__).parent.joinpath("slide_thumbjp2.jpg").is_file()


def test_command_line_save_tiles(_response_ndpi, _response_svs):
    """Test the Save tiles CLI."""
    runner = CliRunner()
    save_tiles_result = runner.invoke(
        cli.main,
        [
            "save-tiles",
            "--wsi_input",
            str(pathlib.Path(__file__).parent),
            "--file_types",
            "*.ndpi, *.svs",
            "--workers",
            "2",
            "--tile_objective_value",
            "5",
        ],
    )

    assert save_tiles_result.exit_code == 0
    shutil.rmtree(pathlib.Path(__file__).parent.joinpath("../tiles"))


def test_command_line_svs_save_tiles(_response_svs):
    runner = CliRunner()
    file_types = "*.svs"
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    save_tiles_result = runner.invoke(
        cli.main,
        [
            "save-tiles",
            "--wsi_input",
            files_all[0],
            "--file_types",
            '"*.ndpi, *.svs"',
            "--workers",
            "2",
            "--tile_objective_value",
            "5",
        ],
    )

    assert save_tiles_result.exit_code == 0
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("../tiles")
        .joinpath("CMU-1-Small-Region.svs")
        .joinpath("slide_thumbnail.jpg")
        .is_file()
    )
    shutil.rmtree(
        pathlib.Path(__file__)
        .parent.joinpath("../tiles")
        .joinpath("CMU-1-Small-Region.svs")
    )


def test_command_line_ndpi_save_tiles(_response_ndpi):
    runner = CliRunner()
    file_types = "*.ndpi"
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    save_tiles_result = runner.invoke(
        cli.main,
        [
            "save-tiles",
            "--wsi_input",
            files_all[0],
            "--file_types",
            '"*.ndpi, *.svs"',
            "--workers",
            "2",
            "--tile_objective_value",
            "5",
        ],
    )

    assert save_tiles_result.exit_code == 0
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("../tiles")
        .joinpath("CMU-1.ndpi")
        .joinpath("slide_thumbnail.jpg")
        .is_file()
    )
    shutil.rmtree(
        pathlib.Path(__file__).parent.joinpath("../tiles").joinpath("CMU-1.ndpi")
    )


def test_command_line_jp2_save_tiles(_response_jp2):
    runner = CliRunner()
    file_types = "*.jp2"
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(__file__).parent), file_types=file_types,
    )
    save_tiles_result = runner.invoke(
        cli.main,
        [
            "save-tiles",
            "--wsi_input",
            files_all[0],
            "--file_types",
            '"*.jp2, *.ndpi"',
            "--workers",
            "2",
            "--tile_objective_value",
            "5",
        ],
    )

    assert save_tiles_result.exit_code == 0
    assert (
        pathlib.Path(__file__)
        .parent.joinpath("../tiles")
        .joinpath("test1.jp2")
        .joinpath("slide_thumbnail.jpg")
        .is_file()
    )
    shutil.rmtree(
        pathlib.Path(__file__).parent.joinpath("../tiles").joinpath("test1.jp2")
    )
