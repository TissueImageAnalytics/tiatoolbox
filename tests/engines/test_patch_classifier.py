"""Test for Patch Classifier."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import zarr

from tiatoolbox.models.engine.patch_classifier import PatchClassifier
from tiatoolbox.utils import env_detection as toolbox_env

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
