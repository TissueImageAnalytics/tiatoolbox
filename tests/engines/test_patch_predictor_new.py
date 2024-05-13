"""Test for Patch Predictor."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from click.testing import CliRunner

from tiatoolbox import cli
from tiatoolbox.models import IOPatchPredictorConfig
from tiatoolbox.models.architecture.vanilla import CNNModel
from tiatoolbox.models.engine.patch_predictor_new import PatchPredictor
from tiatoolbox.utils import download_data, imwrite
from tiatoolbox.utils import env_detection as toolbox_env

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

    predictor.predict(
        [mini_wsi_svs],
        stride_shape=(300, 300),
        mode="wsi",
        on_gpu=ON_GPU,
        save_dir=f"{tmp_path}/dump",
    )
    assert predictor._ioconfig.stride_shape == (300, 300)
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    predictor.predict(
        [mini_wsi_svs],
        resolution=1.99,
        mode="wsi",
        save_dir=f"{tmp_path}/dump",
    )
    assert predictor._ioconfig.input_resolutions[0]["resolution"] == 1.99
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    predictor.predict(
        [mini_wsi_svs],
        units="baseline",
        mode="wsi",
        save_dir=f"{tmp_path}/dump",
    )
    assert predictor._ioconfig.input_resolutions[0]["units"] == "baseline"
    shutil.rmtree(tmp_path / "dump", ignore_errors=True)

    predictor = PatchPredictor(pretrained_model="resnet18-kather100k")
    predictor.predict(
        [mini_wsi_svs],
        mode="wsi",
        merge_predictions=True,
        save_dir=f"{tmp_path}/dump",
    )
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
    predictor = PatchPredictor(pretrained_model="resnet18-kather100k", batch_size=1)
    # don't run test on GPU
    output = predictor.predict(
        inputs,
        on_gpu=ON_GPU,
        save_dir=save_dir_path,
    )
    assert sorted(output.keys()) == ["predictions"]
    assert len(output["predictions"]) == 2
    shutil.rmtree(save_dir_path, ignore_errors=True)

    output = predictor.predict(
        inputs,
        labels=[1, "a"],
        return_labels=True,
        on_gpu=ON_GPU,
        save_dir=save_dir_path,
    )
    assert sorted(output.keys()) == sorted(["labels", "predictions"])
    assert len(output["predictions"]) == len(output["labels"])
    assert output["labels"] == [1, "a"]
    shutil.rmtree(save_dir_path, ignore_errors=True)

    output = predictor.predict(
        inputs,
        return_probabilities=True,
        on_gpu=ON_GPU,
        save_dir=save_dir_path,
    )
    assert sorted(output.keys()) == sorted(["predictions", "probabilities"])
    assert len(output["predictions"]) == len(output["probabilities"])
    shutil.rmtree(save_dir_path, ignore_errors=True)

    output = predictor.predict(
        inputs,
        return_probabilities=True,
        labels=[1, "a"],
        return_labels=True,
        on_gpu=ON_GPU,
        save_dir=save_dir_path,
    )
    assert sorted(output.keys()) == sorted(["labels", "predictions", "probabilities"])
    assert len(output["predictions"]) == len(output["labels"])
    assert len(output["predictions"]) == len(output["probabilities"])

    # test saving output, should have no effect
    _ = predictor.predict(
        inputs,
        on_gpu=ON_GPU,
        save_dir="special_dir_not_exist",
    )
    assert not Path.is_dir(Path("special_dir_not_exist"))

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

    _ = PatchPredictor(
        pretrained_model="resnet18-kather100k",
        pretrained_weights=pretrained_weights,
        batch_size=1,
    )

    # --- test different using user model
    model = CNNModel(backbone="resnet18", num_classes=9)
    # test prediction
    predictor = PatchPredictor(model=model, batch_size=1, verbose=False)
    output = predictor.predict(
        inputs,
        return_probabilities=True,
        labels=[1, "a"],
        return_labels=True,
        on_gpu=ON_GPU,
        save_dir=save_dir_path,
    )
    assert sorted(output.keys()) == sorted(["labels", "predictions", "probabilities"])
    assert len(output["predictions"]) == len(output["labels"])
    assert len(output["predictions"]) == len(output["probabilities"])


def test_wsi_predictor_api(
    sample_wsi_dict: dict,
    tmp_path: Path,
    chdir: Callable,
) -> None:
    """Test normal run of wsi predictor."""
    save_dir_path = tmp_path

    # convert to pathlib Path to prevent wsireader complaint
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    mini_wsi_jpg = Path(sample_wsi_dict["wsi2_4k_4k_jpg"])
    mini_wsi_msk = Path(sample_wsi_dict["wsi2_4k_4k_msk"])

    patch_size = np.array([224, 224])
    predictor = PatchPredictor(pretrained_model="resnet18-kather100k", batch_size=32)

    save_dir = f"{save_dir_path}/model_wsi_output"

    # wrapper to make this more clean
    kwargs = {
        "return_probabilities": True,
        "return_labels": True,
        "on_gpu": ON_GPU,
        "patch_input_shape": patch_size,
        "stride_shape": patch_size,
        "resolution": 1.0,
        "units": "baseline",
        "save_dir": save_dir,
    }
    # ! add this test back once the read at `baseline` is fixed
    # sanity check, both output should be the same with same resolution read args
    wsi_output = predictor.predict(
        [mini_wsi_svs],
        masks=[mini_wsi_msk],
        mode="wsi",
        **kwargs,
    )

    shutil.rmtree(save_dir, ignore_errors=True)

    tile_output = predictor.predict(
        [mini_wsi_jpg],
        masks=[mini_wsi_msk],
        mode="tile",
        **kwargs,
    )

    wpred = np.array(wsi_output[0]["predictions"])
    tpred = np.array(tile_output[0]["predictions"])
    diff = tpred == wpred
    accuracy = np.sum(diff) / np.size(wpred)
    assert accuracy > 0.9, np.nonzero(~diff)

    # remove previously generated data
    shutil.rmtree(save_dir, ignore_errors=True)

    kwargs = {
        "return_probabilities": True,
        "return_labels": True,
        "on_gpu": ON_GPU,
        "patch_input_shape": patch_size,
        "stride_shape": patch_size,
        "resolution": 0.5,
        "save_dir": save_dir,
        "merge_predictions": True,  # to test the api coverage
        "units": "mpp",
    }

    _kwargs = copy.deepcopy(kwargs)
    _kwargs["merge_predictions"] = False
    # test reading of multiple whole-slide images
    output = predictor.predict(
        [mini_wsi_svs, mini_wsi_svs],
        masks=[mini_wsi_msk, mini_wsi_msk],
        mode="wsi",
        **_kwargs,
    )
    for output_info in output.values():
        assert Path(output_info["raw"]).exists()
        assert "merged" not in output_info
    shutil.rmtree(_kwargs["save_dir"], ignore_errors=True)

    # coverage test
    _kwargs = copy.deepcopy(kwargs)
    _kwargs["merge_predictions"] = True
    # test reading of multiple whole-slide images
    predictor.predict(
        [mini_wsi_svs, mini_wsi_svs],
        masks=[mini_wsi_msk, mini_wsi_msk],
        mode="wsi",
        **_kwargs,
    )
    _kwargs = copy.deepcopy(kwargs)
    with pytest.raises(FileExistsError):
        predictor.predict(
            [mini_wsi_svs, mini_wsi_svs],
            masks=[mini_wsi_msk, mini_wsi_msk],
            mode="wsi",
            **_kwargs,
        )
    # remove previously generated data
    shutil.rmtree(_kwargs["save_dir"], ignore_errors=True)

    with chdir(save_dir_path):
        # test reading of multiple whole-slide images
        _kwargs = copy.deepcopy(kwargs)
        _kwargs["save_dir"] = None  # default coverage
        _kwargs["return_probabilities"] = False
        output = predictor.predict(
            [mini_wsi_svs, mini_wsi_svs],
            masks=[mini_wsi_msk, mini_wsi_msk],
            mode="wsi",
            **_kwargs,
        )
        assert Path.exists(Path("output"))
        for output_info in output.values():
            assert Path(output_info["raw"]).exists()
            assert "merged" in output_info
            assert Path(output_info["merged"]).exists()

        # remove previously generated data
        shutil.rmtree("output", ignore_errors=True)


def test_wsi_predictor_merge_predictions(sample_wsi_dict: dict) -> None:
    """Test normal run of wsi predictor with merge predictions option."""
    # convert to pathlib Path to prevent reader complaint
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    mini_wsi_jpg = Path(sample_wsi_dict["wsi2_4k_4k_jpg"])
    mini_wsi_msk = Path(sample_wsi_dict["wsi2_4k_4k_msk"])

    # blind test
    # pseudo output dict from model with 2 patches
    output = {
        "resolution": 1.0,
        "units": "baseline",
        "probabilities": [[0.45, 0.55], [0.90, 0.10]],
        "predictions": [1, 0],
        "coordinates": [[0, 0, 2, 2], [2, 2, 4, 4]],
    }
    merged = PatchPredictor.merge_predictions(
        np.zeros([4, 4]),
        output,
        resolution=1.0,
        units="baseline",
    )
    _merged = np.array([[2, 2, 0, 0], [2, 2, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
    assert np.sum(merged - _merged) == 0

    # blind test for merging probabilities
    merged = PatchPredictor.merge_predictions(
        np.zeros([4, 4]),
        output,
        resolution=1.0,
        units="baseline",
        return_raw=True,
    )
    _merged = np.array(
        [
            [0.45, 0.45, 0, 0],
            [0.45, 0.45, 0, 0],
            [0, 0, 0.90, 0.90],
            [0, 0, 0.90, 0.90],
        ],
    )
    assert merged.shape == (4, 4, 2)
    assert np.mean(np.abs(merged[..., 0] - _merged)) < 1.0e-6

    # integration test
    predictor = PatchPredictor(pretrained_model="resnet18-kather100k", batch_size=1)

    kwargs = {
        "return_probabilities": True,
        "return_labels": True,
        "on_gpu": ON_GPU,
        "patch_input_shape": np.array([224, 224]),
        "stride_shape": np.array([224, 224]),
        "resolution": 1.0,
        "units": "baseline",
        "merge_predictions": True,
    }
    # sanity check, both output should be the same with same resolution read args
    wsi_output = predictor.predict(
        [mini_wsi_svs],
        masks=[mini_wsi_msk],
        mode="wsi",
        **kwargs,
    )

    # mock up to change the preproc func and
    # force to use the default in merge function
    # still should have the same results
    kwargs["merge_predictions"] = False
    tile_output = predictor.predict(
        [mini_wsi_jpg],
        masks=[mini_wsi_msk],
        mode="tile",
        **kwargs,
    )
    merged_tile_output = predictor.merge_predictions(
        mini_wsi_jpg,
        tile_output[0],
        resolution=kwargs["resolution"],
        units=kwargs["units"],
    )
    tile_output.append(merged_tile_output)

    # first make sure nothing breaks with predictions
    wpred = np.array(wsi_output[0]["predictions"])
    tpred = np.array(tile_output[0]["predictions"])
    diff = tpred == wpred
    accuracy = np.sum(diff) / np.size(wpred)
    assert accuracy > 0.9, np.nonzero(~diff)

    merged_wsi = wsi_output[1]
    merged_tile = tile_output[1]
    # ensure shape of merged predictions of tile and wsi input are the same
    assert merged_wsi.shape == merged_tile.shape
    # ensure consistent predictions between tile and wsi mode
    diff = merged_tile == merged_wsi
    accuracy = np.sum(diff) / np.size(merged_wsi)
    assert accuracy > 0.9, np.nonzero(~diff)


def _test_predictor_output(
    inputs: list,
    pretrained_model: str,
    probabilities_check: list | None = None,
    predictions_check: list | None = None,
    *,
    on_gpu: bool = ON_GPU,
) -> None:
    """Test the predictions of multiple models included in tiatoolbox."""
    predictor = PatchPredictor(
        pretrained_model=pretrained_model,
        batch_size=32,
        verbose=False,
    )
    # don't run test on GPU
    output = predictor.predict(
        inputs,
        return_probabilities=True,
        return_labels=False,
        on_gpu=on_gpu,
    )
    predictions = output["predictions"]
    probabilities = output["probabilities"]
    for idx, probabilities_ in enumerate(probabilities):
        probabilities_max = max(probabilities_)
        assert np.abs(probabilities_max - probabilities_check[idx]) <= 1e-3, (
            pretrained_model,
            probabilities_max,
            probabilities_check[idx],
            predictions[idx],
            predictions_check[idx],
        )
        assert predictions[idx] == predictions_check[idx], (
            pretrained_model,
            probabilities_max,
            probabilities_check[idx],
            predictions[idx],
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
    for pretrained_model, expected_prob in pretrained_info.items():
        _test_predictor_output(
            inputs,
            pretrained_model,
            probabilities_check=expected_prob,
            predictions_check=[6, 3],
            on_gpu=ON_GPU,
        )
        # only test 1 on travis to limit runtime
        if toolbox_env.running_on_ci():
            break


def test_patch_predictor_pcam_output(sample_patch3: Path, sample_patch4: Path) -> None:
    """Test the output of patch prediction models on PCam dataset."""
    inputs = [Path(sample_patch3), Path(sample_patch4)]
    pretrained_info = {
        "alexnet-pcam": [0.999980092048645, 0.9769067168235779],
        "resnet18-pcam": [0.999992847442627, 0.9466130137443542],
        "resnet34-pcam": [1.0, 0.9976525902748108],
        "resnet50-pcam": [0.9999270439147949, 0.9999996423721313],
        "resnet101-pcam": [1.0, 0.9997289776802063],
        "resnext50_32x4d-pcam": [0.9999996423721313, 0.9984435439109802],
        "resnext101_32x8d-pcam": [0.9997072815895081, 0.9969086050987244],
        "wide_resnet50_2-pcam": [0.9999837875366211, 0.9959040284156799],
        "wide_resnet101_2-pcam": [1.0, 0.9945427179336548],
        "densenet121-pcam": [0.9999251365661621, 0.9997479319572449],
        "densenet161-pcam": [0.9999969005584717, 0.9662821292877197],
        "densenet169-pcam": [0.9999998807907104, 0.9993504881858826],
        "densenet201-pcam": [0.9999942779541016, 0.9950824975967407],
        "mobilenet_v2-pcam": [0.9999876022338867, 0.9942564368247986],
        "mobilenet_v3_large-pcam": [0.9999922513961792, 0.9719613790512085],
        "mobilenet_v3_small-pcam": [0.9999963045120239, 0.9747149348258972],
        "googlenet-pcam": [0.9999929666519165, 0.8701475858688354],
    }
    for pretrained_model, expected_prob in pretrained_info.items():
        _test_predictor_output(
            inputs,
            pretrained_model,
            probabilities_check=expected_prob,
            predictions_check=[1, 0],
            on_gpu=ON_GPU,
        )
        # only test 1 on travis to limit runtime
        if toolbox_env.running_on_ci():
            break


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
            "--mode",
            '"patch"',
            "--output-path",
            str(tmp_path.joinpath("output")),
        ],
    )

    assert "Invalid value for '--mode'" in mode_not_in_wsi_tile_result.output
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
            "--mode",
            "wsi",
            "--output-path",
            str(tmp_path.joinpath("output")),
        ],
    )

    assert models_wsi_result.exit_code == 0
    assert tmp_path.joinpath("output/0.merged.npy").exists()
    assert tmp_path.joinpath("output/0.raw.json").exists()
    assert tmp_path.joinpath("output/results.json").exists()


def test_cli_model_single_file_mask(remote_sample: Callable, tmp_path: Path) -> None:
    """Test for models CLI single file with mask."""
    mini_wsi_svs = Path(remote_sample("svs-1-small"))
    sample_wsi_msk = remote_sample("small_svs_tissue_mask")
    sample_wsi_msk = np.load(sample_wsi_msk).astype(np.uint8)
    imwrite(f"{tmp_path}/small_svs_tissue_mask.jpg", sample_wsi_msk)
    sample_wsi_msk = f"{tmp_path}/small_svs_tissue_mask.jpg"

    runner = CliRunner()
    models_tiles_result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--img-input",
            str(mini_wsi_svs),
            "--mode",
            "wsi",
            "--masks",
            str(sample_wsi_msk),
            "--output-path",
            str(tmp_path.joinpath("output")),
        ],
    )

    assert models_tiles_result.exit_code == 0
    assert tmp_path.joinpath("output/0.merged.npy").exists()
    assert tmp_path.joinpath("output/0.raw.json").exists()
    assert tmp_path.joinpath("output/results.json").exists()


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
        shutil.copy(mini_wsi_svs, dir_path.joinpath("1_" + mini_wsi_svs.name))
        shutil.copy(mini_wsi_svs, dir_path.joinpath("2_" + mini_wsi_svs.name))
        shutil.copy(mini_wsi_svs, dir_path.joinpath("3_" + mini_wsi_svs.name))

    try:
        dir_path_masks.joinpath("1_" + mini_wsi_msk.name).symlink_to(mini_wsi_msk)
        dir_path_masks.joinpath("2_" + mini_wsi_msk.name).symlink_to(mini_wsi_msk)
        dir_path_masks.joinpath("3_" + mini_wsi_msk.name).symlink_to(mini_wsi_msk)
    except OSError:
        shutil.copy(mini_wsi_msk, dir_path_masks.joinpath("1_" + mini_wsi_msk.name))
        shutil.copy(mini_wsi_msk, dir_path_masks.joinpath("2_" + mini_wsi_msk.name))
        shutil.copy(mini_wsi_msk, dir_path_masks.joinpath("3_" + mini_wsi_msk.name))

    tmp_path = tmp_path.joinpath("output")

    runner = CliRunner()
    models_tiles_result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--img-input",
            str(dir_path),
            "--mode",
            "wsi",
            "--masks",
            str(dir_path_masks),
            "--output-path",
            str(tmp_path),
        ],
    )

    assert models_tiles_result.exit_code == 0
    assert tmp_path.joinpath("0.merged.npy").exists()
    assert tmp_path.joinpath("0.raw.json").exists()
    assert tmp_path.joinpath("1.merged.npy").exists()
    assert tmp_path.joinpath("1.raw.json").exists()
    assert tmp_path.joinpath("2.merged.npy").exists()
    assert tmp_path.joinpath("2.raw.json").exists()
    assert tmp_path.joinpath("results.json").exists()
