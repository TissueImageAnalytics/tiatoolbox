from tiatoolbox.dataloader.save_tiles import save_tiles
from tiatoolbox import utils
from tiatoolbox import cli

import pathlib
import os
from click.testing import CliRunner


def test_save_tiles(_sample_all_wsis, tmp_path):
    """Test for save_tiles as a python function."""
    file_types = ("*.ndpi", "*.svs", "*.mrxs", "*.jp2")
    files_all = utils.misc.grab_files_from_dir(
        input_path=str(pathlib.Path(_sample_all_wsis)), file_types=file_types,
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


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_save_tiles(_sample_all_wsis, tmp_path):
    """Test for save_tiles CLI."""
    runner = CliRunner()
    save_tiles_result = runner.invoke(
        cli.main,
        [
            "save-tiles",
            "--wsi_input",
            str(pathlib.Path(_sample_all_wsis)),
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
        input_path=str(pathlib.Path(_sample_all_wsis)), file_types=file_types,
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
