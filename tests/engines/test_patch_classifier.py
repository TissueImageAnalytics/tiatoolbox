"""Test for Patch Classifier."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import zarr

from tiatoolbox.models.engine.patch_classifier import PatchClassifier
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.misc import get_zarr_array

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def _test_classifier_output(
    inputs: list,
    model: str,
    probabilities_check: list | None = None,
    classification_check: list | None = None,
    tmp_path: Path | None = None,
) -> None:
    """Test the predictions of multiple models included in tiatoolbox."""
    cache_mode = None if tmp_path is None else True
    save_dir = None if tmp_path is None else tmp_path / "output"
    classifier = PatchClassifier(
        model=model,
        batch_size=32,
        verbose=False,
    )
    # don't run test on GPU
    output = classifier.run(
        inputs,
        return_labels=False,
        device=device,
        cache_mode=cache_mode,
        save_dir=save_dir,
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
        _test_classifier_output(
            inputs,
            model,
            probabilities_check=expected_prob,
            classification_check=[6, 3],
        )

    # cache mode
    for model, expected_prob in pretrained_info.items():
        _test_classifier_output(
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
            output["probabilities"].append(probs_dict.pop("prob_0"))
            output["predictions"].append(probs_dict.pop("type"))

    return output


def _validate_probabilities(output: list | dict | zarr.group) -> bool:
    """Helper function to test if the probabilities value are valid."""
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


def test_wsi_predictor_zarr(sample_wsi_dict: dict, tmp_path: Path) -> None:
    """Test normal run of patch predictor for WSIs."""
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])

    classifier = PatchClassifier(
        model="alexnet-kather100k",
        batch_size=32,
        verbose=False,
    )
    # don't run test on GPU
    output = classifier.run(
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


def test_engine_run_wsi_annotation_store(
    sample_wsi_dict: dict,
    tmp_path: Path,
) -> None:
    """Test the engine run for Whole slide images."""
    # convert to pathlib Path to prevent wsireader complaint
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    mini_wsi_msk = Path(sample_wsi_dict["wsi2_4k_4k_msk"])

    eng = PatchClassifier(model="alexnet-kather100k")

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
    output_ = _extract_probabilities_from_annotation_store(output_)

    # prediction for each patch
    assert np.array(output_["predictions"]).shape == (69,)
    assert _validate_probabilities(output_)

    shutil.rmtree(save_dir)
