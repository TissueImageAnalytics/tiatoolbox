"""Test for Patch Predictor."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import zarr

from tiatoolbox.models import IOPatchClassifierConfig
from tiatoolbox.models.architecture.vanilla import CNNModel
from tiatoolbox.models.engine.patch_predictor import PatchPredictor
from tiatoolbox.utils import download_data
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
    ioconfig = IOPatchClassifierConfig(
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
