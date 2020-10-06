#!/usr/bin/env python

"""pytests for `tiatoolbox` package."""
import pytest

from tiatoolbox.dataloader.slide_info import slide_info
from tiatoolbox.dataloader.save_tiles import save_tiles
from tiatoolbox.dataloader import wsireader
from tiatoolbox.tools.stainnorm import get_normaliser
from tiatoolbox import utils
from tiatoolbox.utils.exceptions import FileNotSupported
from tiatoolbox.utils.misc import imread
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
        print("Downloading NDPI")
        r = requests.get(
            "http://openslide.cs.cmu.edu/download/openslide-testdata"
            "/Hamamatsu/CMU-1.ndpi"
        )
        with open(ndpi_file_path, "wb") as f:
            f.write(r.content)
    else:
        print("Skipping NDPI")

    return ndpi_file_path


@pytest.fixture(scope="session")
def _response_svs(tmpdir_factory):
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

    return svs_file_path


@pytest.fixture(scope="session")
def _response_jp2(tmpdir_factory):
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

    return jp2_file_path


@pytest.fixture(scope="session")
def _response_all_wsis(_response_ndpi, _response_svs, tmpdir_factory):
    dir_path = pathlib.Path(tmpdir_factory.mktemp("data"))

    try:
        dir_path.joinpath(_response_ndpi.basename).symlink_to(_response_ndpi)
        dir_path.joinpath(_response_svs.basename).symlink_to(_response_svs)
    except OSError:
        shutil.copy(_response_ndpi, dir_path.joinpath(_response_ndpi.basename))
        shutil.copy(_response_svs, dir_path.joinpath(_response_svs.basename))

    return dir_path


@pytest.fixture(scope="session")
def _response_stainnorm_source(tmpdir_factory):
    """
    Sample pytest fixture for source image for stain normalisation
    Download png image for pytest
    """
    source_file_path = tmpdir_factory.mktemp("data").join("source.png")
    if not os.path.isfile(source_file_path):
        print("Downloading Source Image for stain normalisation")
        r = requests.get(
            "https://warwick.ac.uk/fac/sci/dcs/research/tia/tiatoolbox"
            "/files/source.png"
        )
        with open(source_file_path, "wb") as f:
            f.write(r.content)

    else:
        print("Skipping Source Image")

    return source_file_path


@pytest.fixture(scope="session")
def _response_stainnorm_target(tmpdir_factory):
    """
    Sample pytest fixture for target image for stain normalisation
    Download png image for pytest
    """
    target_file_path = tmpdir_factory.mktemp("data").join("target.png")
    if not os.path.isfile(target_file_path):
        print("Downloading Target Image for stain normalisation")
        r = requests.get(
            "https://warwick.ac.uk/fac/sci/dcs/research/tia/tiatoolbox"
            "/files/target.png"
        )
        with open(target_file_path, "wb") as f:
            f.write(r.content)

    else:
        print("Skipping Target Image")

    return target_file_path


@pytest.fixture(scope="session")
def _response_reinhard(tmpdir_factory):
    """
    Sample pytest fixture for reinhard normalised image
    Download png image for pytest
    """
    reinhard_file_path = tmpdir_factory.mktemp("data").join("reinhard.png")
    print("Downloading Reinhard Image for stain normalisation")
    if not os.path.isfile(reinhard_file_path):
        r = requests.get(
            "https://warwick.ac.uk/fac/sci/dcs/research/tia/tiatoolbox"
            "/files/reinhard.png"
        )
        with open(reinhard_file_path, "wb") as f:
            f.write(r.content)

    else:
        print("Skipping Reinhard Image")

    return reinhard_file_path


@pytest.fixture(scope="session")
def _response_ruifrok(tmpdir_factory):
    """
    Sample pytest fixture for ruifrok normalised image
    Download png image for pytest
    """
    ruifrok_file_path = tmpdir_factory.mktemp("data").join("ruifrok.png")
    if not os.path.isfile(ruifrok_file_path):
        r = requests.get(
            "https://warwick.ac.uk/fac/sci/dcs/research/tia/tiatoolbox"
            "/files/ruifrok.png"
        )
        with open(ruifrok_file_path, "wb") as f:
            f.write(r.content)

    else:
        print("Skipping Reinhard Image")

    return ruifrok_file_path


# -------------------------------------------------------------------------------------
# Python API tests
# -------------------------------------------------------------------------------------


def test_slide_info(_response_all_wsis, tmp_path):
    """pytest for slide_info as a python function"""
    file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
    files_all = utils.misc.grab_files_from_dir(
        input_path=_response_all_wsis,
        file_types=file_types,
    )

    for curr_file in files_all:
        slide_param = slide_info(input_path=curr_file, verbose=True)
        utils.misc.save_yaml(
            slide_param.as_dict(), tmp_path / (slide_param.file_name + ".yaml")
        )


def test_wsireader_slide_info(_response_svs, tmp_path):
    """pytest for slide_info in WSIReader class as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent),
        file_types=file_types,
    )
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(files_all[0]))
    wsi_obj = wsireader.OpenSlideWSIReader(input_dir, file_name + ext)
    slide_param = wsi_obj.slide_info
    utils.misc.save_yaml(
        slide_param.as_dict(), tmp_path / (slide_param.file_name + ".yaml")
    )


def test_wsireader_read_region(_response_svs):
    """pytest for read region as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent),
        file_types=file_types,
    )
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(files_all[0]))
    wsi_obj = wsireader.OpenSlideWSIReader(input_dir, file_name + ext)
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
        input_path=str(pathlib.Path(_response_svs).parent),
        file_types=file_types,
    )
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(files_all[0]))
    wsi_obj = wsireader.OpenSlideWSIReader(input_dir, file_name + ext)
    slide_thumbnail = wsi_obj.slide_thumbnail()
    assert isinstance(slide_thumbnail, np.ndarray)
    assert slide_thumbnail.dtype == "uint8"


def test_wsireader_save_tiles(_response_svs, tmp_path):
    """pytest for save_tiles in wsireader as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent),
        file_types=file_types,
    )
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(files_all[0]))
    wsi_obj = wsireader.OpenSlideWSIReader(
        input_dir,
        file_name + ext,
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
        input_path=str(pathlib.Path(_response_all_wsis)),
        file_types=file_types,
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


def test_reinhard_normalise(
    _response_stainnorm_source, _response_stainnorm_target, _response_reinhard
):
    """pytest for Reinhard colour normalisation."""

    source_img = imread(pathlib.Path(_response_stainnorm_source))
    target_img = imread(pathlib.Path(_response_stainnorm_target))
    reinhard_img = imread(pathlib.Path(_response_reinhard))

    norm = get_normaliser("reinhard")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.sum(reinhard_img - transform) < 1e-3


def test_ruifrok_normalise(
    _response_stainnorm_source, _response_stainnorm_target, _response_ruifrok
):
    """pytest for stain normalisation with stain matrix from Ruifrok and Johnston."""

    source_img = imread(pathlib.Path(_response_stainnorm_source))
    target_img = imread(pathlib.Path(_response_stainnorm_target))
    ruifrok_img = imread(pathlib.Path(_response_ruifrok))

    # init class with Ruifrok method
    norm = get_normaliser("ruifrok")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.sum(ruifrok_img - transform) < 1e-3


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_help_interface():
    """pytest the CLI help"""
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
    """pytest the Slide information CLI."""
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
        input_path=str(pathlib.Path(_response_all_wsis)),
        file_types=file_types,
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
    """pytest the Read Region CLI."""
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
        ],
    )

    assert save_tiles_result.exit_code == 0

    file_types = "*.svs"
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_all_wsis)),
        file_types=file_types,
    )
    save_tiles_result = runner.invoke(
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

    assert save_tiles_result.exit_code == 0


def test_command_line_stainnorm(_response_stainnorm_source, _response_stainnorm_target):
    """pytest for the stain normalisation CLI."""
    runner = CliRunner()
    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stainnorm",
            "--source_input",
            pathlib.Path(_response_stainnorm_source),
            "--target_input",
            pathlib.Path(_response_stainnorm_target),
            "--method",
            "reinhard",
        ],
    )

    assert stainnorm_result.exit_code == 0

    stainnorm_result = runner.invoke(
        cli.main,
        [
            "stainnorm",
            "--source_input",
            pathlib.Path(_response_stainnorm_source),
            "--target_input",
            pathlib.Path(_response_stainnorm_target),
            "--method",
            "ruifrok",
        ],
    )

    assert stainnorm_result.exit_code == 0
