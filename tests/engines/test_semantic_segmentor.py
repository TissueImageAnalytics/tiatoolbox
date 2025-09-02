"""Test SemanticSegmentor."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from unittest import mock

import dask.array as da
import numpy as np
import torch
import zarr
from click.testing import CliRunner

from tiatoolbox import cli
from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.models.engine import semantic_segmentor
from tiatoolbox.models.engine.semantic_segmentor import (
    SemanticSegmentor,
    merge_vertical_chunkwise,
)
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore import WSIReader

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


def test_non_overlapping_blocks() -> None:
    """Test for non-overlapping merge to canvas."""
    blocks = np.array([np.ones((2, 2, 1)), np.ones((2, 2, 1)) * 2])
    output_locations = np.array([[0, 0, 2, 2], [2, 0, 4, 2]])
    merged_shape = (2, 4, 1)
    canvas, count = semantic_segmentor.merge_batch_to_canvas(
        blocks, output_locations, merged_shape
    )
    assert np.array_equal(canvas[:, :2, :], np.ones((2, 2, 1)))
    assert np.array_equal(canvas[:, 2:, :], np.ones((2, 2, 1)) * 2)
    assert np.array_equal(count, np.ones((2, 4, 1)))


def test_overlapping_blocks() -> None:
    """Test for overlapping merge to canvas."""
    blocks = np.array([np.ones((2, 2, 1)), np.ones((2, 2, 1)) * 3])
    output_locations = np.array([[0, 0, 2, 2], [1, 0, 3, 2]])
    merged_shape = (2, 3, 1)
    canvas, count = semantic_segmentor.merge_batch_to_canvas(
        blocks, output_locations, merged_shape
    )
    expected_canvas = np.array([[[1], [4], [3]], [[1], [4], [3]]])
    expected_count = np.array([[[1], [2], [1]], [[1], [2], [1]]])
    assert np.array_equal(canvas, expected_canvas)
    assert np.array_equal(count, expected_count)


def test_zero_block() -> None:
    """Test for zero merge to canvas."""
    blocks = np.array([np.zeros((2, 2, 1)), np.ones((2, 2, 1))])
    output_locations = np.array([[0, 0, 2, 2], [2, 0, 4, 2]])
    merged_shape = (2, 4, 1)
    canvas, count = semantic_segmentor.merge_batch_to_canvas(
        blocks, output_locations, merged_shape
    )
    assert np.array_equal(canvas[:, :2, :], np.zeros((2, 2, 1)))
    assert np.array_equal(canvas[:, 2:, :], np.ones((2, 2, 1)))
    assert np.array_equal(count[:, :2, :], np.zeros((2, 2, 1)))
    assert np.array_equal(count[:, 2:, :], np.ones((2, 2, 1)))


def test_empty_blocks() -> None:
    """Test for empty merge to canvas."""
    blocks = np.empty((0, 2, 2, 1))
    output_locations = np.empty((0, 4))
    merged_shape = (2, 2, 1)
    canvas, count = semantic_segmentor.merge_batch_to_canvas(
        blocks, output_locations, merged_shape
    )
    assert np.array_equal(canvas, np.zeros((2, 2, 1)))
    assert np.array_equal(count, np.zeros((2, 2, 1), dtype=np.uint8))


def test_merge_vertical_chunkwise_memory_threshold_triggered() -> None:
    """Test merge vertical chunkwise for memory threshold."""
    # Create dummy canvas and count arrays with 3 vertical chunks
    data = np.ones((30, 10), dtype=np.uint8)
    canvas = da.from_array(data, chunks=(10, 10))
    count = da.from_array(data, chunks=(10, 10))

    # Output locations to simulate overlaps
    output_locs_y_ = np.array([[0, 10], [10, 20], [20, 30]])

    # Temporary Zarr group
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir)

        # Mock psutil to simulate low memory
        with mock.patch(
            "tiatoolbox.models.engine.semantic_segmentor.psutil.virtual_memory"
        ) as mock_vm:
            mock_vm.return_value.free = 1  # Very low free memory

            result = merge_vertical_chunkwise(
                canvas=canvas,
                count=count,
                output_locs_y_=output_locs_y_,
                zarr_group=None,
                save_path=save_path,
                memory_threshold=0.01,  # Very low threshold to trigger the condition
            )

        # Assertions
        assert isinstance(result, da.Array)
        assert hasattr(result, "name")
        assert result.name.startswith("from-zarr")
        assert np.all(result.compute() == data)

        zarr_group = zarr.open(tmpdir, mode="r")
        assert np.all(zarr_group["probabilities"][:] == data)


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
    # Return Probabilities is True
    # Testing with WSIReader
    output = segmentor.run(
        images=[WSIReader.open(sample_svs)],
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=tmp_path / "task_length_cache",
        batch_size=2,
        output_type="zarr",
        memory_threshold=1,
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
