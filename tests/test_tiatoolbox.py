#!/usr/bin/env python

"""pytests for `tiatoolbox` package."""
import pytest
from pytest import approx

from tiatoolbox.dataloader.slide_info import slide_info

from tiatoolbox.dataloader.save_tiles import save_tiles
from tiatoolbox.dataloader import wsireader, wsimeta
from tiatoolbox.tools.stainnorm import get_normaliser
from tiatoolbox import utils

from tiatoolbox.utils.exceptions import FileNotSupported, MethodNotSupported
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
            "/test2.jp2"
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
    """Test for slide_info as a python function"""
    file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
    files_all = utils.misc.grab_files_from_dir(
        input_path=_response_all_wsis, file_types=file_types,
    )

    for curr_file in files_all:
        slide_param = slide_info(input_path=curr_file, verbose=True)
        out_path = tmp_path / slide_param.file_path.with_suffix(".yaml").name
        utils.misc.save_yaml(slide_param.as_dict(), out_path)


def test_wsireader_slide_info(_response_svs, tmp_path):
    """Test for slide_info in WSIReader class as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent), file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    slide_param = wsi.slide_info
    out_path = tmp_path / slide_param.file_path.with_suffix(".yaml").name
    utils.misc.save_yaml(slide_param.as_dict(), out_path)


def test_wsireader_read_region(_response_svs):
    """Test for read region as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent), file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    level = 0
    region = [1000, 2000, 2000, 3000]
    im_region = wsi.read_region(region[0], region[1], region[2], region[3], level)

    assert isinstance(im_region, np.ndarray)
    assert im_region.dtype == "uint8"
    assert im_region.shape == (1000, 1000, 3)


def test_wsireader_slide_thumbnail(_response_svs):
    """Test for slide_thumbnail as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent), file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(files_all[0])
    slide_thumbnail = wsi.slide_thumbnail()
    assert isinstance(slide_thumbnail, np.ndarray)
    assert slide_thumbnail.dtype == "uint8"


def test_wsireader_save_tiles(_response_svs, tmp_path):
    """Test for save_tiles in wsireader as a python function"""
    file_types = ("*.svs",)
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_response_svs).parent), file_types=file_types,
    )
    wsi = wsireader.OpenSlideWSIReader(
        files_all[0],
        output_dir=str(pathlib.Path(tmp_path).joinpath("test_wsireader_save_tiles")),
        tile_objective_value=5,
    )
    wsi.save_tiles(verbose=True)
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
    """Test for save_tiles as a python function"""
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
    wsi = wsireader.OmnyxJP2WSIReader(
        _response_jp2,
        output_dir=str(
            pathlib.Path(tmp_path).joinpath("test_wsireader_jp2_save_tiles")
        ),
        tile_objective_value=5,
    )
    wsi.save_tiles(verbose=True)
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
    """Test for Exceptions"""
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

    with pytest.raises(MethodNotSupported):
        get_normaliser(method_name="invalid_normaliser")


def test_imresize():
    """Test for imresize"""
    img = np.zeros((2000, 2000, 3))
    resized_img = utils.transforms.imresize(img, 0.5)
    assert resized_img.shape == (1000, 1000, 3)


def test_background_composite():
    """Test for background composite"""
    new_im = np.zeros((2000, 2000, 4)).astype("uint8")
    new_im[:1000, :, 3] = 255
    im = utils.transforms.background_composite(new_im)
    assert np.all(im[1000:, :, :] == 255)
    assert np.all(im[:1000, :, :] == 0)

    im = utils.transforms.background_composite(new_im, alpha=True)
    assert np.all(im[:, :, 3] == 255)


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


def test_reinhard_normalise(
    _response_stainnorm_source, _response_stainnorm_target, _response_reinhard
):
    """Test for Reinhard colour normalisation."""
    source_img = imread(pathlib.Path(_response_stainnorm_source))
    target_img = imread(pathlib.Path(_response_stainnorm_target))
    reinhard_img = imread(pathlib.Path(_response_reinhard))

    norm = get_normaliser("reinhard")
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.sum(reinhard_img - transform) < 1e-3


def test_custom_normalise(
    _response_stainnorm_source, _response_stainnorm_target, _response_ruifrok
):
    """Test for stain normalisation with user-defined stain matrix."""
    source_img = imread(pathlib.Path(_response_stainnorm_source))
    target_img = imread(pathlib.Path(_response_stainnorm_target))
    ruifrok_img = imread(pathlib.Path(_response_ruifrok))

    # init class with custom method - test with ruifrok stain matrix
    stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
    norm = get_normaliser("custom", stain_matrix=stain_matrix)
    norm.fit(target_img)  # get stain information of target image
    transform = norm.transform(source_img)  # transform source image

    assert np.shape(transform) == np.shape(source_img)
    assert np.sum(ruifrok_img - transform) < 1e-3


def test_ruifrok_normalise(
    _response_stainnorm_source, _response_stainnorm_target, _response_ruifrok
):
    """Test for stain normalisation with stain matrix from Ruifrok and Johnston."""
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
    """Test the CLI help"""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert help_result.output == result.output


def test_command_line_version():
    """Test for version check"""
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
    """pytest OpenSlide read_region CLI."""
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


def test_command_line_jp2_read_region(_response_jp2, tmp_path):
    """pytest JP2 read_region"""
    runner = CliRunner()
    read_region_result = runner.invoke(
        cli.main,
        [
            "read-region",
            "--wsi_input",
            str(pathlib.Path(_response_jp2)),
            "--level",
            "0",
            "--mode",
            "save",
            "--output_path",
            str(pathlib.Path(tmp_path).joinpath("im_region.jpg")),
        ],
    )

    assert read_region_result.exit_code == 0
    assert os.path.isfile(str(pathlib.Path(tmp_path).joinpath("im_region.jpg")))


def test_command_line_slide_thumbnail(_response_ndpi, tmp_path):
    """Test for the slide_thumbnail CLI."""
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


def test_command_line_jp2_slide_thumbnail(_response_jp2, tmp_path):
    """pytest for the jp2 slide_thumbnail CLI."""
    runner = CliRunner()
    slide_thumb_result = runner.invoke(
        cli.main,
        [
            "slide-thumbnail",
            "--wsi_input",
            str(pathlib.Path(_response_jp2)),
            "--mode",
            "save",
            "--output_path",
            str(pathlib.Path(tmp_path).joinpath("slide_thumb.jpg")),
        ],
    )

    assert slide_thumb_result.exit_code == 0
    assert pathlib.Path(tmp_path).joinpath("slide_thumb.jpg").is_file()


def test_command_line_save_tiles(_response_all_wsis, tmp_path):
    """Test for save_tiles CLI."""
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


@pytest.mark.filterwarnings("ignore")
def test_wsimeta_validate_fail():
    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512), level_dimensions=[])
    assert meta.validate() is False


def test_command_line_stainnorm(_response_stainnorm_source, _response_stainnorm_target):
    """Test for the stain normalisation CLI."""
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


@pytest.mark.filterwarnings("ignore")
def test_wsimeta_validate_pass():
    meta = wsimeta.WSIMeta(slide_dimensions=(512, 512))
    assert meta.validate()

    meta = wsimeta.WSIMeta(
        slide_dimensions=(512, 512),
        level_dimensions=[(512, 512), (256, 256)],
        level_downsamples=[1, 2],
    )


def test_wsimeta_openslidewsireader_ndpi(_response_ndpi, tmp_path):
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(_response_ndpi))
    wsi_obj = wsireader.OpenSlideWSIReader(_response_ndpi)
    meta = wsi_obj.slide_info
    assert meta.validate()


def test_wsimeta_openslidewsireader_svs(_response_svs, tmp_path):
    input_dir, file_name, ext = utils.misc.split_path_name_ext(str(_response_svs))
    wsi_obj = wsireader.OpenSlideWSIReader(_response_svs)
    meta = wsi_obj.slide_info
    assert meta.validate()


def test_openslidewsireader_relative_level_scales_mpp(_response_ndpi):
    path = pathlib.Path(_response_ndpi)
    wsi = wsireader.OpenSlideWSIReader(path)
    level_scales = wsi.relative_level_scales(0.5, "mpp")
    expected = np.array(
        [
            [0.91282519, 0.91012514],
            [1.82565039, 1.82025028],
            [3.65130078, 3.64050057],
            [7.30260155, 7.28100114],
            [14.6052031, 14.56200228],
            [29.21040621, 29.12400455],
            [58.42081241, 58.2480091],
            [116.84162483, 116.4960182],
            [233.68324966, 232.99203641],
        ]
    )
    assert level_scales == approx(expected)


def test_openslidewsireader_relative_level_scales_power(_response_ndpi):
    path = pathlib.Path(_response_ndpi)
    wsi = wsireader.OpenSlideWSIReader(path)
    level_scales = wsi.relative_level_scales(wsi.slide_info.objective_power, "power")
    assert level_scales[0] == 1
    downsamples = np.array(wsi.slide_info.level_downsamples)
    assert np.all(level_scales == downsamples)


def test_openslidewsireader_relative_level_scales_level(_response_ndpi):
    path = pathlib.Path(_response_ndpi)
    wsi = wsireader.OpenSlideWSIReader(path)
    level_scales = wsi.relative_level_scales(3, "level")
    assert level_scales[3] == 1.0
    downsamples = np.array(wsi.slide_info.level_downsamples)
    expected = downsamples / downsamples[3]
    assert np.all(level_scales == expected)


def test_openslidewsireader_relative_level_scales_baseline(_response_ndpi):
    path = pathlib.Path(_response_ndpi)
    wsi = wsireader.OpenSlideWSIReader(path)
    level_scales = wsi.relative_level_scales(0.125, "baseline")
    assert level_scales[3] == 1.0
    downsamples = np.array(wsi.slide_info.level_downsamples)
    expected = downsamples * 0.125
    assert np.all(level_scales == expected)


def test_openslidewsireader_optimal_relative_level_scale_mpp(_response_ndpi):
    path = pathlib.Path(_response_ndpi)
    wsi = wsireader.OpenSlideWSIReader(path)

    level_mpp_05, level_rescale_mpp_05 = wsi.optimal_relative_level_scale(0.5, "mpp")
    level_mpp_10, level_rescale_mpp_10 = wsi.optimal_relative_level_scale(10, "mpp")

    assert np.all(level_mpp_05 == 0)
    assert np.all(level_mpp_10 == 4)
    assert level_rescale_mpp_05 == approx([0.91282519, 0.91012514])
    assert level_rescale_mpp_10 == approx([0.73026016, 0.72810011])


def test_openslidewsireader_optimal_relative_level_scales_power(_response_ndpi):
    path = pathlib.Path(_response_ndpi)
    wsi = wsireader.OpenSlideWSIReader(path)

    level_power_05, level_rescale_power_05 = wsi.optimal_relative_level_scale(
        2.5, "power"
    )
    level_power_10, level_rescale_power_10 = wsi.optimal_relative_level_scale(
        10, "power"
    )

    assert level_power_05 == 3
    assert level_rescale_power_05 == 1.0
    assert level_power_10 == 1
    assert level_rescale_power_10 == 1.0


def test_openslidewsireader_read_rect_params_for_scale_power(_response_ndpi):
    path = pathlib.Path(_response_ndpi)
    wsi = wsireader.OpenSlideWSIReader(path)

    location = (0, 0)
    target_size = (256, 256)
    # Test a range of objective powers
    for target_scale in [1.25, 2.5, 5, 10, 20]:
        (
            level_location,
            level,
            read_size,
            post_read_scale,
        ) = wsi.read_rect_params_for_scale(
            location=location,
            target_size=target_size,
            target_scale=target_scale,
            units="power",
        )
        assert level >= 0
        assert level < wsi.slide_info.level_count
        # Check that read_size * scale == target_size
        post_read_downscaled_size = np.round(read_size * post_read_scale).astype(int)
        assert np.array_equal(post_read_downscaled_size, np.array(target_size))


def test_openslidewsireader_read_rect_params_for_scale_mpp(_response_ndpi):
    path = pathlib.Path(_response_ndpi)
    wsi = wsireader.OpenSlideWSIReader(path)

    location = (0, 0)
    target_size = (256, 256)
    # Test a range of MPP
    for target_scale in range(1, 10):
        (
            level_location,
            level,
            read_size,
            post_read_scale,
        ) = wsi.read_rect_params_for_scale(
            location=location,
            target_size=target_size,
            target_scale=target_scale,
            units="mpp",
        )
        assert level >= 0
        assert level < wsi.slide_info.level_count
        # Check that read_size * scale == target_size
        post_read_downscaled_size = np.round(read_size * post_read_scale).astype(int)
        assert np.array_equal(post_read_downscaled_size, np.array(target_size))

