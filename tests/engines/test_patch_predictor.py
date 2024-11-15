"""Test for Patch Predictor."""

from __future__ import annotations

import copy
import json
import shutil
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import zarr
from click.testing import CliRunner

from tiatoolbox import cli
from tiatoolbox.models import IOPatchPredictorConfig
from tiatoolbox.models.architecture.vanilla import CNNModel
from tiatoolbox.models.engine.patch_predictor import PatchPredictor
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.misc import download_data, get_zarr_array, imwrite

if TYPE_CHECKING:
    import pytest

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def _test_predictor_output(
    inputs: list,
    model: str,
    probabilities_check: list | None = None,
    classification_check: list | None = None,
    output_type: str = "dict",
    tmp_path: Path | None = None,
) -> None:
    """Test the predictions of multiple models included in tiatoolbox."""
    cache_mode = None if tmp_path is None else True
    save_dir = None if tmp_path is None else tmp_path / "output"
    predictor = PatchPredictor(
        model=model,
        batch_size=32,
        verbose=False,
    )
    # don't run test on GPU
    output = predictor.run(
        inputs,
        return_labels=False,
        device=device,
        cache_mode=cache_mode,
        save_dir=save_dir,
        output_type=output_type,
    )

    if tmp_path is not None:
        output = zarr.open(output, mode="r")

    probabilities = output["probabilities"]
    classification = output["predictions"]
    for idx, probabilities_ in enumerate(probabilities):
        probabilities_max = max(probabilities_)
        assert np.abs(probabilities_max - probabilities_check[idx]) <= 1e-3, (
            model,
            probabilities_max,
            probabilities_check[idx],
            probabilities_,
            classification_check[idx],
        )
        assert classification[idx] == classification_check[idx], (
            model,
            probabilities_max,
            probabilities_check[idx],
            probabilities_,
            classification_check[idx],
        )
    if save_dir:
        shutil.rmtree(save_dir)


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
    """Test PatchPredictor API."""
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
    assert sorted(output.keys()) == ["predictions", "probabilities"]
    assert len(output["probabilities"]) == 2
    shutil.rmtree(save_dir_path, ignore_errors=True)

    # whether to return labels
    output = predictor.run(
        inputs,
        labels=["1", "a"],
        return_labels=True,
    )
    assert sorted(output.keys()) == sorted(["labels", "predictions", "probabilities"])
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
    assert sorted(output.keys()) == sorted(["labels", "predictions", "probabilities"])
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

    wsi_out = zarr.open(str(output[mini_wsi_svs]), mode="r")
    tile_out = zarr.open(str(output[mini_wsi_jpg]), mode="r")
    diff = tile_out["probabilities"][:] == wsi_out["probabilities"][:]
    accuracy = np.sum(diff) / np.size(wsi_out["probabilities"][:])
    assert accuracy > 0.99, np.nonzero(~diff)

    diff = tile_out["predictions"][:] == wsi_out["predictions"][:]
    accuracy = np.sum(diff) / np.size(wsi_out["predictions"][:])
    assert accuracy > 0.99, np.nonzero(~diff)

    shutil.rmtree(_kwargs["save_dir"], ignore_errors=True)


def test_patch_predictor_kather100k_output(
    sample_patch1: Path,
    sample_patch2: Path,
    tmp_path: Path,
) -> None:
    """Test the output of patch classification models on Kather100K dataset."""
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
            classification_check=[6, 3],
        )

    # cache mode
    for model, expected_prob in pretrained_info.items():
        _test_predictor_output(
            inputs,
            model,
            probabilities_check=expected_prob,
            classification_check=[6, 3],
            tmp_path=tmp_path,
        )


def _extract_probabilities_from_annotation_store(dbfile: str) -> dict:
    """Helper function to extract probabilities from Annotation Store."""
    con = sqlite3.connect(dbfile)
    cur = con.cursor()
    annotations_properties = list(cur.execute("SELECT properties FROM annotations"))

    output = {"probabilities": [], "predictions": []}

    for item in annotations_properties:
        for json_str in item:
            probs_dict = json.loads(json_str)
            if "proba_0" in probs_dict:
                output["probabilities"].append(probs_dict.pop("prob_0"))
            output["predictions"].append(probs_dict.pop("type"))

    return output


def _validate_probabilities(output: list | dict | zarr.group) -> bool:
    """Helper function to test if the probabilities value are valid."""
    probabilities = np.array([0.5])

    if "probabilities" in output:
        probabilities = output["probabilities"]

    predictions = output["predictions"]
    if isinstance(probabilities, dict):
        return all(0 <= probability <= 1 for _, probability in probabilities.items())

    predictions = np.array(get_zarr_array(predictions)).astype(int)
    probabilities = get_zarr_array(probabilities)

    if not np.all(np.array(probabilities) <= 1):
        return False

    if not np.all(np.array(probabilities) >= 0):
        return False

    return np.all(predictions[:][0:5] == [7, 3, 2, 3, 3])


def test_wsi_predictor_zarr(
    sample_wsi_dict: dict, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
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
    # prediction for each patch
    assert output_["predictions"].shape == (70,)
    assert output_["predictions"].ndim == 1
    assert _validate_probabilities(output=output_)
    assert "Output file saved at " in caplog.text

    output = predictor.run(
        images=[mini_wsi_svs],
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=tmp_path / "wsi_out_check_no_probabilities",
    )

    assert output[mini_wsi_svs].exists()

    output_ = zarr.open(output[mini_wsi_svs])

    assert "probabilities" not in output_
    # number of patches x [start_x, start_y, end_x, end_y]
    assert output_["coordinates"].shape == (70, 4)
    assert output_["coordinates"].ndim == 2
    # prediction for each patch
    assert output_["predictions"].shape == (70,)
    assert output_["predictions"].ndim == 1
    assert _validate_probabilities(output=output_)
    assert "Output file saved at " in caplog.text


def test_patch_predictor_patch_mode_annotation_store(
    sample_patch1: Path,
    sample_patch2: Path,
    tmp_path: Path,
) -> None:
    """Test the output of patch classification models on Kather100K dataset."""
    inputs = [Path(sample_patch1), Path(sample_patch2)]

    predictor = PatchPredictor(
        model="alexnet-kather100k",
        batch_size=32,
        verbose=False,
    )
    # don't run test on GPU
    output = predictor.run(
        images=inputs,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
        save_dir=tmp_path / "patch_out_check",
        output_type="annotationstore",
    )

    assert output.exists()
    output = _extract_probabilities_from_annotation_store(output)
    assert np.all(output["predictions"] == [6, 3])
    assert np.all(np.array(output["probabilities"]) <= 1)
    assert np.all(np.array(output["probabilities"]) >= 0)


def test_patch_predictor_patch_mode_no_probabilities(
    sample_patch1: Path,
    sample_patch2: Path,
    tmp_path: Path,
) -> None:
    """Test the output of patch classification models on Kather100K dataset."""
    inputs = [Path(sample_patch1), Path(sample_patch2)]

    predictor = PatchPredictor(
        model="alexnet-kather100k",
        batch_size=32,
        verbose=False,
    )

    output = predictor.run(
        images=inputs,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
    )

    assert "probabilities" not in output

    # don't run test on GPU
    output = predictor.run(
        images=inputs,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
        save_dir=tmp_path / "patch_out_check",
        output_type="annotationstore",
    )

    assert output.exists()
    output = _extract_probabilities_from_annotation_store(output)
    assert np.all(output["predictions"] == [6, 3])
    assert output["probabilities"] == []


def test_engine_run_wsi_annotation_store(
    sample_wsi_dict: dict,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
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
        batch_size=4,
        **kwargs,
    )

    output_ = output[mini_wsi_svs]

    assert output_.exists()
    assert output_.suffix == ".db"
    output_ = _extract_probabilities_from_annotation_store(output_)

    # prediction for each patch
    assert np.array(output_["predictions"]).shape == (69,)
    assert _validate_probabilities(output_)

    assert "Output file saved at " in caplog.text

    shutil.rmtree(save_dir)


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


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
