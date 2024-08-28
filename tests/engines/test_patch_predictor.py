"""Test for Patch Predictor."""

from __future__ import annotations

import copy
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np
import zarr
from click.testing import CliRunner

from tiatoolbox import cli
from tiatoolbox.models import IOPatchPredictorConfig
from tiatoolbox.models.architecture.vanilla import CNNModel
from tiatoolbox.models.engine.patch_predictor import PatchPredictor
from tiatoolbox.utils import download_data, imwrite
from tiatoolbox.utils import env_detection as toolbox_env

device = "cuda" if toolbox_env.has_gpu() else "cpu"
ON_GPU = toolbox_env.has_gpu()
RNG = np.random.default_rng()  # Numpy Random Generator


# -------------------------------------------------------------------------------------
# Engine
# -------------------------------------------------------------------------------------


def test_io_config_delegation(remote_sample: Callable, tmp_path: Path) -> None:
    """Test for delegating args to io config."""
    mini_wsi_svs = Path(remote_sample("wsi2_4k_4k_svs"))
    model = CNNModel("resnet50")
    predictor = PatchPredictor(model=model, weights=None)
    kwargs = {
        "patch_input_shape": [512, 512],
        "resolution": 1.75,
        "units": "mpp",
    }

    # test providing config / full input info for default models without weights
    ioconfig = IOPatchPredictorConfig(
        patch_input_shape=(512, 512),
        stride_shape=(256, 256),
        input_resolutions=[{"resolution": 1.35, "units": "mpp"}],
    )
    predictor.run(
        images=[mini_wsi_svs],
        ioconfig=ioconfig,
        patch_mode=False,
        save_dir=f"{tmp_path}/dump",
    )
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    predictor.run(
        images=[mini_wsi_svs],
        patch_mode=False,
        save_dir=f"{tmp_path}/dump",
        **kwargs,
    )
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    # test overwriting pretrained ioconfig
    predictor = PatchPredictor(model="resnet18-kather100k", batch_size=1)
    predictor.run(
        images=[mini_wsi_svs],
        patch_input_shape=(300, 300),
        patch_mode=False,
        save_dir=f"{tmp_path}/dump",
    )
    assert predictor._ioconfig.patch_input_shape == (300, 300)
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    predictor.run(
        images=[mini_wsi_svs],
        stride_shape=(300, 300),
        patch_mode=False,
        save_dir=f"{tmp_path}/dump",
    )
    assert predictor._ioconfig.stride_shape == (300, 300)
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    predictor.run(
        images=[mini_wsi_svs],
        resolution=1.99,
        patch_mode=False,
        save_dir=f"{tmp_path}/dump",
    )
    assert predictor._ioconfig.input_resolutions[0]["resolution"] == 1.99
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    predictor.run(
        images=[mini_wsi_svs],
        units="baseline",
        patch_mode=False,
        save_dir=f"{tmp_path}/dump",
    )
    assert predictor._ioconfig.input_resolutions[0]["units"] == "baseline"
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    predictor.run(
        images=[mini_wsi_svs],
        units="level",
        resolution=0,
        patch_mode=False,
        save_dir=f"{tmp_path}/dump",
    )
    assert predictor._ioconfig.input_resolutions[0]["units"] == "level"
    assert predictor._ioconfig.input_resolutions[0]["resolution"] == 0
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    predictor.run(
        images=[mini_wsi_svs],
        units="power",
        resolution=20,
        patch_mode=False,
        save_dir=f"{tmp_path}/dump",
    )
    assert predictor._ioconfig.input_resolutions[0]["units"] == "power"
    assert predictor._ioconfig.input_resolutions[0]["resolution"] == 20
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)


def test_patch_predictor_api(
    sample_patch1: Path,
    sample_patch2: Path,
    tmp_path: Path,
) -> None:
    """Helper function to get the model output using API 1."""
    save_dir_path = tmp_path

    # convert to pathlib Path to prevent reader complaint
    inputs = [Path(sample_patch1), Path(sample_patch2)]
    predictor = PatchPredictor(model="resnet18-kather100k", batch_size=1)
    # don't run test on GPU
    # Default run
    output = predictor.run(
        inputs,
        device="cpu",
    )
    assert sorted(output.keys()) == ["probabilities"]
    assert len(output["probabilities"]) == 2
    shutil.rmtree(save_dir_path, ignore_errors=True)

    # whether to return labels
    output = predictor.run(
        inputs,
        labels=["1", "a"],
        return_labels=True,
    )
    assert sorted(output.keys()) == sorted(["labels", "probabilities"])
    assert len(output["probabilities"]) == len(output["labels"])
    assert output["labels"].tolist() == ["1", "a"]
    shutil.rmtree(save_dir_path, ignore_errors=True)

    # test loading user weight
    pretrained_weights_url = (
        "https://tiatoolbox.dcs.warwick.ac.uk/models/pc/resnet18-kather100k.pth"
    )

    # remove prev generated data
    shutil.rmtree(save_dir_path, ignore_errors=True)
    save_dir_path.mkdir(parents=True)
    pretrained_weights = (
        save_dir_path / "tmp_pretrained_weigths" / "resnet18-kather100k.pth"
    )

    download_data(pretrained_weights_url, pretrained_weights)

    predictor = PatchPredictor(
        model="resnet18-kather100k",
        weights=pretrained_weights,
        batch_size=1,
    )
    ioconfig = predictor.ioconfig

    # --- test different using user model
    model = CNNModel(backbone="resnet18", num_classes=9)
    # test prediction
    predictor = PatchPredictor(model=model, batch_size=1, verbose=False)
    output = predictor.run(
        inputs,
        labels=[1, 2],
        return_labels=True,
        ioconfig=ioconfig,
    )
    assert sorted(output.keys()) == sorted(["labels", "probabilities"])
    assert len(output["probabilities"]) == len(output["labels"])
    assert output["labels"].tolist() == [1, 2]


def test_wsi_predictor_api(
    sample_wsi_dict: dict,
    tmp_path: Path,
) -> None:
    """Test normal run of wsi predictor."""
    save_dir_path = tmp_path

    # convert to pathlib Path to prevent wsireader complaint
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    mini_wsi_jpg = Path(sample_wsi_dict["wsi2_4k_4k_jpg"])
    mini_wsi_msk = Path(sample_wsi_dict["wsi2_4k_4k_msk"])

    patch_size = np.array([224, 224])
    predictor = PatchPredictor(model="resnet18-kather100k", batch_size=32)

    save_dir = f"{save_dir_path}/model_wsi_output"

    # wrapper to make this more clean
    kwargs = {
        "patch_input_shape": patch_size,
        "stride_shape": patch_size,
        "resolution": 1.0,
        "units": "baseline",
        "save_dir": save_dir,
    }
    # ! add this test back once the read at `baseline` is fixed
    # sanity check, both output should be the same with same resolution read args
    # remove previously generated data

    _kwargs = copy.deepcopy(kwargs)
    # test reading of multiple whole-slide images
    output = predictor.run(
        images=[mini_wsi_svs, mini_wsi_jpg],
        masks=[mini_wsi_msk, mini_wsi_msk],
        patch_mode=False,
        **_kwargs,
    )

    wsi_pred = zarr.open(str(output[mini_wsi_svs]), mode="r")
    tile_pred = zarr.open(str(output[mini_wsi_jpg]), mode="r")
    diff = tile_pred["probabilities"][:] == wsi_pred["probabilities"][:]
    accuracy = np.sum(diff) / np.size(wsi_pred["probabilities"][:])
    assert accuracy > 0.99, np.nonzero(~diff)

    shutil.rmtree(_kwargs["save_dir"], ignore_errors=True)


def _test_predictor_output(
    inputs: list,
    model: str,
    probabilities_check: list | None = None,
    predictions_check: list | None = None,
) -> None:
    """Test the predictions of multiple models included in tiatoolbox."""
    predictor = PatchPredictor(
        model=model,
        batch_size=32,
        verbose=False,
    )
    # don't run test on GPU
    output = predictor.run(
        inputs,
        return_probabilities=True,
        return_labels=False,
        device=device,
    )
    predictions = output["probabilities"]
    for idx, probabilities_ in enumerate(predictions):
        probabilities_max = max(probabilities_)
        assert np.abs(probabilities_max - probabilities_check[idx]) <= 1e-3, (
            model,
            probabilities_max,
            probabilities_check[idx],
            probabilities_,
            predictions_check[idx],
        )
        assert np.argmax(probabilities_) == predictions_check[idx], (
            model,
            probabilities_max,
            probabilities_check[idx],
            probabilities_,
            predictions_check[idx],
        )


def test_patch_predictor_kather100k_output(
    sample_patch1: Path,
    sample_patch2: Path,
) -> None:
    """Test the output of patch prediction models on Kather100K dataset."""
    inputs = [Path(sample_patch1), Path(sample_patch2)]
    pretrained_info = {
        "alexnet-kather100k": [1.0, 0.9999735355377197],
        "resnet18-kather100k": [1.0, 0.9999911785125732],
        "resnet34-kather100k": [1.0, 0.9979840517044067],
        "resnet50-kather100k": [1.0, 0.9999986886978149],
        "resnet101-kather100k": [1.0, 0.9999932050704956],
        "resnext50_32x4d-kather100k": [1.0, 0.9910059571266174],
        "resnext101_32x8d-kather100k": [1.0, 0.9999971389770508],
        "wide_resnet50_2-kather100k": [1.0, 0.9953408241271973],
        "wide_resnet101_2-kather100k": [1.0, 0.9999831914901733],
        "densenet121-kather100k": [1.0, 1.0],
        "densenet161-kather100k": [1.0, 0.9999959468841553],
        "densenet169-kather100k": [1.0, 0.9999934434890747],
        "densenet201-kather100k": [1.0, 0.9999983310699463],
        "mobilenet_v2-kather100k": [0.9999998807907104, 0.9999126195907593],
        "mobilenet_v3_large-kather100k": [0.9999996423721313, 0.9999878406524658],
        "mobilenet_v3_small-kather100k": [0.9999998807907104, 0.9999997615814209],
        "googlenet-kather100k": [1.0, 0.9999639987945557],
    }
    for model, expected_prob in pretrained_info.items():
        _test_predictor_output(
            inputs,
            model,
            probabilities_check=expected_prob,
            predictions_check=[6, 3],
        )
        # only test 1 on travis to limit runtime
        if toolbox_env.running_on_ci():
            break


def _validate_probabilities(predictions: list | dict) -> bool:
    """Helper function to test if the probabilities value are valid."""
    if isinstance(predictions, dict):
        return all(0 <= probability <= 1 for _, probability in predictions.items())

    for row in predictions:
        for element in row:
            if not (0 <= element <= 1):
                return False
    return True


def test_wsi_predictor_zarr(sample_wsi_dict: dict, tmp_path: Path) -> None:
    """Test normal run of patch predictor for WSIs."""
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])

    predictor = PatchPredictor(
        model="alexnet-kather100k",
        batch_size=32,
        verbose=False,
    )
    # don't run test on GPU
    output = predictor.run(
        images=[mini_wsi_svs],
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=tmp_path / "wsi_out_check",
    )

    assert output[mini_wsi_svs].exists()

    output_ = zarr.open(output[mini_wsi_svs])

    assert output_["probabilities"].shape == (70, 9)  # number of patches x classes
    assert output_["probabilities"].ndim == 2
    # number of patches x [start_x, start_y, end_x, end_y]
    assert output_["coordinates"].shape == (70, 4)
    assert output_["coordinates"].ndim == 2
    assert _validate_probabilities(predictions=output_["probabilities"])


def test_wsi_predictor_zarr_baseline(sample_wsi_dict: dict, tmp_path: Path) -> None:
    """Test normal run of patch predictor for WSIs."""
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])

    predictor = PatchPredictor(
        model="alexnet-kather100k",
        batch_size=32,
        verbose=False,
    )
    # don't run test on GPU
    output = predictor.run(
        images=[mini_wsi_svs],
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=tmp_path / "wsi_out_check",
        units="baseline",
        resolution=1.0,
    )

    assert output[mini_wsi_svs].exists()

    output_ = zarr.open(output[mini_wsi_svs])

    assert output_["probabilities"].shape == (244, 9)  # number of patches x classes
    assert output_["probabilities"].ndim == 2
    # number of patches x [start_x, start_y, end_x, end_y]
    assert output_["coordinates"].shape == (244, 4)
    assert output_["coordinates"].ndim == 2
    assert _validate_probabilities(predictions=output_["probabilities"])


def _extract_probabilities_from_annotation_store(dbfile: str) -> dict:
    """Helper function to extract probabilities from Annotation Store."""
    probs_dict = {}
    con = sqlite3.connect(dbfile)
    cur = con.cursor()
    annotations_properties = list(cur.execute("SELECT properties FROM annotations"))

    for item in annotations_properties:
        for json_str in item:
            probs_dict = json.loads(json_str)
            probs_dict.pop("prob_0")

    return probs_dict


def test_engine_run_wsi_annotation_store(
    sample_wsi_dict: dict,
    tmp_path: Path,
) -> None:
    """Test the engine run for Whole slide images."""
    # convert to pathlib Path to prevent wsireader complaint
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    mini_wsi_msk = Path(sample_wsi_dict["wsi2_4k_4k_msk"])

    eng = PatchPredictor(model="alexnet-kather100k")

    patch_size = np.array([224, 224])
    save_dir = f"{tmp_path}/model_wsi_output"

    kwargs = {
        "patch_input_shape": patch_size,
        "stride_shape": patch_size,
        "resolution": 0.5,
        "save_dir": save_dir,
        "units": "mpp",
        "scale_factor": (2.0, 2.0),
    }

    output = eng.run(
        images=[mini_wsi_svs],
        masks=[mini_wsi_msk],
        patch_mode=False,
        output_type="AnnotationStore",
        **kwargs,
    )

    output_ = output[mini_wsi_svs]

    assert output_.exists()
    assert output_.suffix == ".db"
    predictions = _extract_probabilities_from_annotation_store(output_)
    assert _validate_probabilities(predictions)

    shutil.rmtree(save_dir)


def test_engine_run_wsi_annotation_store_power(
    sample_wsi_dict: dict,
    tmp_path: Path,
) -> None:
    """Test the engine run for Whole slide images."""
    # convert to pathlib Path to prevent wsireader complaint
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    mini_wsi_msk = Path(sample_wsi_dict["wsi2_4k_4k_msk"])

    eng = PatchPredictor(model="alexnet-kather100k")

    patch_size = np.array([224, 224])
    save_dir = f"{tmp_path}/model_wsi_output"

    kwargs = {
        "patch_input_shape": patch_size,
        "stride_shape": patch_size,
        "resolution": 20,
        "save_dir": save_dir,
        "units": "power",
    }

    output = eng.run(
        images=[mini_wsi_svs],
        masks=[mini_wsi_msk],
        patch_mode=False,
        output_type="AnnotationStore",
        **kwargs,
    )

    output_ = output[mini_wsi_svs]

    assert output_.exists()
    assert output_.suffix == ".db"
    predictions = _extract_probabilities_from_annotation_store(output_)
    assert _validate_probabilities(predictions)

    shutil.rmtree(save_dir)


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_models_file_not_found(sample_svs: Path, tmp_path: Path) -> None:
    """Test for models CLI file not found error."""
    runner = CliRunner()
    model_file_not_found_result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--img-input",
            str(sample_svs)[:-1],
            "--file-types",
            '"*.ndpi, *.svs"',
            "--output-path",
            str(tmp_path.joinpath("output")),
        ],
    )

    assert model_file_not_found_result.output == ""
    assert model_file_not_found_result.exit_code == 1
    assert isinstance(model_file_not_found_result.exception, FileNotFoundError)


def test_command_line_models_incorrect_mode(sample_svs: Path, tmp_path: Path) -> None:
    """Test for models CLI mode not in wsi, tile."""
    runner = CliRunner()
    mode_not_in_wsi_tile_result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--img-input",
            str(sample_svs),
            "--file-types",
            '"*.ndpi, *.svs"',
            "--patch-mode",
            '"patch"',
            "--output-path",
            str(tmp_path.joinpath("output")),
        ],
    )

    assert "Invalid value for '--patch-mode'" in mode_not_in_wsi_tile_result.output
    assert mode_not_in_wsi_tile_result.exit_code != 0
    assert isinstance(mode_not_in_wsi_tile_result.exception, SystemExit)


def test_cli_model_single_file(sample_svs: Path, tmp_path: Path) -> None:
    """Test for models CLI single file."""
    runner = CliRunner()
    models_wsi_result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--img-input",
            str(sample_svs),
            "--patch-mode",
            "False",
            "--output-path",
            str(tmp_path / "output"),
        ],
    )

    assert models_wsi_result.exit_code == 0
    assert (tmp_path / "output" / (sample_svs.stem + ".db")).exists()


def test_cli_model_multiple_file_mask(remote_sample: Callable, tmp_path: Path) -> None:
    """Test for models CLI multiple file with mask."""
    mini_wsi_svs = Path(remote_sample("svs-1-small"))
    sample_wsi_msk = remote_sample("small_svs_tissue_mask")
    sample_wsi_msk = np.load(sample_wsi_msk).astype(np.uint8)
    imwrite(f"{tmp_path}/small_svs_tissue_mask.jpg", sample_wsi_msk)
    mini_wsi_msk = tmp_path.joinpath("small_svs_tissue_mask.jpg")

    # Make multiple copies for test
    dir_path = tmp_path.joinpath("new_copies")
    dir_path.mkdir()

    dir_path_masks = tmp_path.joinpath("new_copies_masks")
    dir_path_masks.mkdir()

    try:
        dir_path.joinpath("1_" + mini_wsi_svs.name).symlink_to(mini_wsi_svs)
        dir_path.joinpath("2_" + mini_wsi_svs.name).symlink_to(mini_wsi_svs)
        dir_path.joinpath("3_" + mini_wsi_svs.name).symlink_to(mini_wsi_svs)
    except OSError:
        shutil.copy(mini_wsi_svs, dir_path / ("1_" + mini_wsi_svs.name))
        shutil.copy(mini_wsi_svs, dir_path / ("2_" + mini_wsi_svs.name))
        shutil.copy(mini_wsi_svs, dir_path / ("3_" + mini_wsi_svs.name))

    try:
        dir_path_masks.joinpath("1_" + mini_wsi_msk.name).symlink_to(mini_wsi_msk)
        dir_path_masks.joinpath("2_" + mini_wsi_msk.name).symlink_to(mini_wsi_msk)
        dir_path_masks.joinpath("3_" + mini_wsi_msk.name).symlink_to(mini_wsi_msk)
    except OSError:
        shutil.copy(mini_wsi_msk, dir_path_masks / ("1_" + mini_wsi_msk.name))
        shutil.copy(mini_wsi_msk, dir_path_masks / ("2_" + mini_wsi_msk.name))
        shutil.copy(mini_wsi_msk, dir_path_masks / ("3_" + mini_wsi_msk.name))

    runner = CliRunner()
    models_tiles_result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--img-input",
            str(dir_path),
            "--patch-mode",
            str(False),
            "--masks",
            str(dir_path_masks),
            "--output-path",
            str(tmp_path / "output"),
            "--output-type",
            "zarr",
        ],
    )

    assert models_tiles_result.exit_code == 0
    assert (tmp_path / "output" / ("1_" + mini_wsi_svs.stem + ".zarr")).exists()
    assert (tmp_path / "output" / ("2_" + mini_wsi_svs.stem + ".zarr")).exists()
    assert (tmp_path / "output" / ("3_" + mini_wsi_svs.stem + ".zarr")).exists()
