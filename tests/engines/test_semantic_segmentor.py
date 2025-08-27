"""Test SemanticSegmentor."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import zarr
from click.testing import CliRunner

from tiatoolbox import cli
from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.misc import imread

if TYPE_CHECKING:
    import pytest

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

    assert not segmentor.patch_mode

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
        save_dir=tmp_path / "output1",
        output_type="annotationstore",
        verbose=True,
    )

    assert output[0] == tmp_path / "output1" / (sample_image.stem + ".db")
    assert len(output) == 1
    _test_store_output_patch(output[0])


def test_save_annotation_store_nparray(
    remote_sample: Callable, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test for saving output as annotation store using a numpy array."""
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask", batch_size=32, verbose=False, device=device
    )

    sample_image = remote_sample("thumbnail-1k-1k")

    input_image = imread(sample_image)
    inputs_list = np.array([input_image, input_image])

    output = segmentor.run(
        images=inputs_list,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
        save_dir=tmp_path / "output1",
        output_type="annotationstore",
    )

    assert output[0] == tmp_path / "output1" / "0.db"
    assert output[1] == tmp_path / "output1" / "1.db"

    assert (tmp_path / "output1" / "output.zarr").exists()

    zarr_group = zarr.open(str(tmp_path / "output1" / "output.zarr"), mode="r")
    assert "probabilities" in zarr_group

    assert "Probability maps cannot be saved as AnnotationStore." in caplog.text

    _test_store_output_patch(output[0])
    _test_store_output_patch(output[1])

    output = segmentor.run(
        images=inputs_list,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
        save_dir=tmp_path / "output2",
        output_type="annotationstore",
    )

    assert output[0] == tmp_path / "output2" / "0.db"
    assert output[1] == tmp_path / "output2" / "1.db"
    assert not (tmp_path / "output2" / "output.zarr").exists()

    assert len(output) == 2

    _test_store_output_patch(output[0])
    _test_store_output_patch(output[1])


def test_wsi_segmentor_zarr(
    remote_sample: Callable,
    sample_svs: Path,
    tmp_path: Path,
) -> None:
    """Test SemanticSegmentor for WSIs with zarr output."""
    wsi1_2k_2k_svs = Path(remote_sample("wsi1_2k_2k_svs"))

    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask",
        batch_size=64,
        verbose=False,
        num_workers=1,
    )
    # Return Probabilities is False
    output = segmentor.run(
        images=[sample_svs],
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=tmp_path / "wsi_out_check",
        batch_size=2,
        output_type="zarr",
        memory_threshold=1,
    )

    output_ = zarr.open(output[sample_svs], mode="r")
    assert 0.17 < np.mean(output_["predictions"][:]) < 0.19
    assert "probabilities" not in output_
    assert "canvas" not in output_
    assert "count" not in output_

    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask",
        batch_size=64,
        verbose=False,
        num_workers=1,
    )
    # Return Probabilities is False
    output = segmentor.run(
        images=[sample_svs],
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=tmp_path / "task_length_cache",
        batch_size=2,
        output_type="zarr",
        da_length_threshold=1,
    )

    output_ = zarr.open(output[sample_svs], mode="r")
    assert 0.17 < np.mean(output_["predictions"][:]) < 0.19
    assert "probabilities" in output_
    assert "canvas" not in output_
    assert "count" not in output_

    # Return Probabilities is True
    # Using small image for faster run
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask",
        batch_size=32,
        verbose=False,
        num_workers=1,
    )
    segmentor.drop_keys = []
    output = segmentor.run(
        images=[sample_svs, wsi1_2k_2k_svs],
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=tmp_path / "wsi_out_check_prob",
        output_type="zarr",
    )

    output_ = zarr.open(output[sample_svs], mode="r")
    assert 0.17 < np.mean(output_["predictions"][:]) < 0.19
    assert 0.52 < np.mean(output_["probabilities"][:]) < 0.56

    output_ = zarr.open(output[wsi1_2k_2k_svs], mode="r")
    assert 0.24 < np.mean(output_["predictions"][:]) < 0.25
    assert 0.48 < np.mean(output_["probabilities"][:]) < 0.52


def test_wsi_segmentor_annotationstore(
    sample_svs: Path, tmp_path: Path, caplog: pytest.CaptureFixture
) -> None:
    """Test SemanticSegmentor for WSIs with AnnotationStore output."""
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask",
        batch_size=32,
        verbose=False,
    )
    # Return Probabilities is False
    output = segmentor.run(
        images=[sample_svs],
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=tmp_path / "wsi_out_check",
        verbose=True,
        output_type="annotationstore",
    )

    assert output[sample_svs] == tmp_path / "wsi_out_check" / (sample_svs.stem + ".db")

    # Return Probabilities
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask",
        batch_size=32,
        verbose=False,
    )
    # Return Probabilities is False
    output = segmentor.run(
        images=[sample_svs],
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=tmp_path / "wsi_prob_out_check",
        verbose=True,
        output_type="annotationstore",
    )

    assert output[sample_svs] == tmp_path / "wsi_prob_out_check" / (
        sample_svs.stem + ".db"
    )
    assert output[sample_svs].with_suffix(".zarr").exists()

    zarr_group = zarr.open(output[sample_svs].with_suffix(".zarr"), mode="r")
    assert "probabilities" in zarr_group
    assert "Probability maps cannot be saved as AnnotationStore." in caplog.text


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_cli_model_single_file(sample_svs: Path, tmp_path: Path) -> None:
    """Test for models CLI single file."""
    runner = CliRunner()
    models_wsi_result = runner.invoke(
        cli.main,
        [
            "semantic-segmentor",
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
