"""Test SemanticSegmentor."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import zarr

from tiatoolbox.models.engine.semantic_segmentor_new import SemanticSegmentor
from tiatoolbox.utils import env_detection as toolbox_env

if TYPE_CHECKING:
    from pathlib import Path

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def test_semantic_segmentor_init() -> None:
    """Tests SemanticSegmentor initialization."""
    segmentor = SemanticSegmentor(model="fcn-tissue_mask", device=device)

    assert isinstance(segmentor, SemanticSegmentor)
    assert isinstance(segmentor.model, torch.nn.Module)


def test_semantic_segmentor_patches(remote_sample: Callable, tmp_path: Path) -> None:
    """Tests SemanticSegmentor on image patches."""
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask", batch_size=32, verbose=False, device=device
    )

    sample_image = remote_sample("thumbnail-1k-1k")

    inputs = [sample_image, sample_image]

    assert segmentor.cache_mode is False

    output = segmentor.run(
        images=inputs,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
    )

    assert 0.62 < np.mean(output["predictions"][:]) < 0.66
    assert 0.495 < np.mean(output["probabilities"][:]) < 0.505

    assert (
        tuple(segmentor._ioconfig.patch_output_shape)
        == output["probabilities"][0].shape[:-1]
    )

    assert (
        tuple(segmentor._ioconfig.patch_output_shape) == output["predictions"][0].shape
    )

    output = segmentor.run(
        images=inputs,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
        cache_mode=True,
        save_dir=tmp_path / "output0",
    )

    assert output == tmp_path / "output0" / "output.zarr"

    output = zarr.open(output, mode="r")
    assert 0.62 < np.mean(output["predictions"][:]) < 0.66
    assert 0.495 < np.mean(output["probabilities"][:]) < 0.505

    output = segmentor.run(
        images=inputs,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
        cache_mode=True,
        save_dir=tmp_path / "output1",
    )

    assert output == tmp_path / "output1" / "output.zarr"

    output = zarr.open(output, mode="r")
    assert 0.62 < np.mean(output["predictions"][:]) < 0.66
    assert "probabilities" not in output.keys()  # noqa: SIM118


def test_save_annotation_store(remote_sample: Callable, tmp_path: Path) -> None:
    """Test for saving output as annotation store."""
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask", batch_size=32, verbose=False, device=device
    )

    sample_image = remote_sample("thumbnail-1k-1k")

    inputs = [sample_image, sample_image]

    output = segmentor.run(
        images=inputs,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
        cache_mode=True,
        save_dir=tmp_path / "output1",
        output_type="annotationstore",
    )

    assert output == tmp_path / "output1" / "output.db"
    con = sqlite3.connect(output)
    cur = con.cursor()
    annotations_properties = list(cur.execute("SELECT properties FROM annotations"))

    out = []

    for item in annotations_properties:
        for json_str in item:
            probs = json.loads(json_str)
            if "type" in probs:
                out.append(probs.pop("type"))

    assert "mask" in out

    assert annotations_properties is not None
