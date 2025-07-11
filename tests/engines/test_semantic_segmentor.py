"""Test SemanticSegmentor."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import zarr

from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.models.engine.semantic_segmentor_new import SemanticSegmentor
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.misc import imread

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
    assert 0.48 < np.mean(output["probabilities"][:]) < 0.52

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
    assert 0.48 < np.mean(output["probabilities"][:]) < 0.52

    output = segmentor.run(
        images=inputs,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
        cache_mode=True,
        output_type="zarr",
        save_dir=tmp_path / "output1",
    )

    assert output == tmp_path / "output1" / "output.zarr"

    output = zarr.open(output, mode="r")
    assert 0.62 < np.mean(output["predictions"][:]) < 0.66
    assert "probabilities" not in output.keys()  # noqa: SIM118

    output = segmentor.run(
        images=inputs,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
        cache_mode=False,
        save_dir=tmp_path / "output2",
        output_type="zarr",
    )

    assert output == tmp_path / "output2" / "output.zarr"

    output = zarr.open(output, mode="r")
    assert 0.62 < np.mean(output["predictions"][:]) < 0.66
    assert "probabilities" not in output
    assert "predictions" in output


def _test_store_output_patch(output: Path) -> None:
    """Helper method to test annotation store output for a patch."""
    store_ = SQLiteStore.open(output)
    annotations_ = store_.values()
    annotations_geometry_type = [
        str(annotation_.geometry_type) for annotation_ in annotations_
    ]
    assert "Polygon" in annotations_geometry_type

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


def test_save_annotation_store(remote_sample: Callable, tmp_path: Path) -> None:
    """Test for saving output as annotation store."""
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask", batch_size=32, verbose=False, device=device
    )

    sample_image = remote_sample("thumbnail-1k-1k")

    inputs = [sample_image]

    output = segmentor.run(
        images=inputs,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
        cache_mode=False,
        save_dir=tmp_path / "output1",
        output_type="annotationstore",
    )

    assert output[0] == tmp_path / "output1" / (sample_image.stem + ".db")
    assert len(output) == 1
    _test_store_output_patch(output[0])


def test_save_annotation_store_nparray(remote_sample: Callable, tmp_path: Path) -> None:
    """Test for saving output as annotation store using a numpy array."""
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask", batch_size=32, verbose=False, device=device
    )

    sample_image = remote_sample("thumbnail-1k-1k")

    input_image = imread(sample_image)
    inputs_list = [input_image, input_image]

    output = segmentor.run(
        images=inputs_list,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
        cache_mode=True,
        save_dir=tmp_path / "output1",
        output_type="annotationstore",
    )

    assert output[0] == tmp_path / "output1" / "0.db"
    assert output[1] == tmp_path / "output1" / "1.db"

    assert output[2] == tmp_path / "output1.zarr"

    _test_store_output_patch(output[0])
    _test_store_output_patch(output[1])

    output = segmentor.run(
        images=inputs_list,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
        cache_mode=True,
        save_dir=tmp_path / "output2",
        output_type="annotationstore",
    )

    assert output[0] == tmp_path / "output2" / "0.db"
    assert output[1] == tmp_path / "output2" / "1.db"

    assert len(output) == 2

    _test_store_output_patch(output[0])
    _test_store_output_patch(output[1])


def test_wsi_segmentor_zarr(
    remote_sample: Callable, sample_svs: Path, tmp_path: Path
) -> None:
    """Test SemanticSegmentor for WSIs with zarr output."""
    wsi_with_artifacts = Path(remote_sample("wsi3_20k_20k_svs"))

    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask",
        batch_size=32,
        verbose=False,
    )
    # Return Probabilitis is False
    output = segmentor.run(
        images=[wsi_with_artifacts],
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=tmp_path / "wsi_out_check",
        output_type="zarr",
    )

    assert output[wsi_with_artifacts].exists()

    # Return Probabilities is True
    # Using small image for faster run
    output = segmentor.run(
        images=[sample_svs],
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=tmp_path / "wsi_out_check_prob",
        output_type="zarr",
    )

    assert output[sample_svs].exists()
