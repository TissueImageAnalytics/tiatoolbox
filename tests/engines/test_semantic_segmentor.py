"""Test SemanticSegmentor."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import torch
import zarr
from click.testing import CliRunner

from tiatoolbox import cli
from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.models.engine import semantic_segmentor
from tiatoolbox.models.engine.semantic_segmentor import (
    SemanticSegmentor,
    merge_vertical_chunkwise,
    prepare_full_batch,
)
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore import WSIReader

if TYPE_CHECKING:
    from collections.abc import Callable

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def test_semantic_segmentor_init() -> None:
    """Tests SemanticSegmentor initialization."""
    segmentor = SemanticSegmentor(model="fcn-tissue_mask", device=device)

    assert isinstance(segmentor, SemanticSegmentor)
    assert isinstance(segmentor.model, torch.nn.Module)


def test_semantic_segmentor_patches(
    remote_sample: Callable, track_tmp_path: Path
) -> None:
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
        save_dir=track_tmp_path / "output0",
    )

    assert output == track_tmp_path / "output0" / "output.zarr"

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
        save_dir=track_tmp_path / "output1",
    )

    assert output == track_tmp_path / "output1" / "output.zarr"

    output = zarr.open(output, mode="r")
    assert 0.62 < np.mean(output["predictions"][:]) < 0.66
    assert "probabilities" not in output.keys()  # noqa: SIM118

    output = segmentor.run(
        images=inputs,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
        save_dir=track_tmp_path / "output2",
        output_type="zarr",
    )

    assert output == track_tmp_path / "output2" / "output.zarr"

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


def test_save_annotation_store(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Test for saving output as annotation store."""
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask", batch_size=32, verbose=False, device=device
    )

    # Test str input
    sample_image = remote_sample("thumbnail-1k-1k")

    inputs = [str(sample_image)]

    output = segmentor.run(
        images=inputs,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
        save_dir=track_tmp_path / "output1",
        output_type="annotationstore",
        verbose=True,
    )

    assert output[0] == track_tmp_path / "output1" / (sample_image.stem + ".db")
    assert len(output) == 1
    _test_store_output_patch(output[0])


def test_save_annotation_store_nparray(
    remote_sample: Callable, track_tmp_path: Path, caplog: pytest.LogCaptureFixture
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
        save_dir=track_tmp_path / "output1",
        output_type="annotationstore",
    )

    assert output[0] == track_tmp_path / "output1" / "0.db"
    assert output[1] == track_tmp_path / "output1" / "1.db"

    assert (track_tmp_path / "output1" / "output.zarr").exists()

    zarr_group = zarr.open(str(track_tmp_path / "output1" / "output.zarr"), mode="r")
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
        save_dir=track_tmp_path / "output2",
        output_type="annotationstore",
    )

    assert output[0] == track_tmp_path / "output2" / "0.db"
    assert output[1] == track_tmp_path / "output2" / "1.db"
    assert not (track_tmp_path / "output2" / "output.zarr").exists()

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


def test_raise_value_error_return_labels_wsi(
    sample_svs: Path,
    track_tmp_path: Path,
) -> None:
    """Test for raises value error for return_labels in wsi mode."""
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask",
        batch_size=64,
        verbose=False,
        num_workers=1,
    )
    with pytest.raises(
        ValueError,
        match=r".*return_labels` is not supported when `patch_mode` is False",
    ):
        _ = segmentor.run(
            images=[sample_svs],
            return_probabilities=False,
            return_labels=True,
            device=device,
            patch_mode=False,
            save_dir=track_tmp_path / "wsi_out_check",
            batch_size=2,
            output_type="zarr",
        )


def test_wsi_segmentor_zarr(
    remote_sample: Callable,
    sample_svs: Path,
    track_tmp_path: Path,
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
        save_dir=track_tmp_path / "wsi_out_check",
        batch_size=2,
        output_type="zarr",
        memory_threshold=1,
    )

    output_ = zarr.open(output[sample_svs], mode="r")
    assert 0.17 < np.mean(output_["predictions"][:]) < 0.21
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
        save_dir=track_tmp_path / "task_length_cache",
        batch_size=2,
        output_type="zarr",
        memory_threshold=1,
    )

    output_ = zarr.open(output[sample_svs], mode="r")
    assert 0.17 < np.mean(output_["predictions"][:]) < 0.21
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
        save_dir=track_tmp_path / "wsi_out_check_prob",
        output_type="zarr",
    )

    output_ = zarr.open(output[sample_svs], mode="r")
    assert 0.17 < np.mean(output_["predictions"][:]) < 0.21
    assert 0.48 < np.mean(output_["probabilities"][:]) < 0.52

    output_ = zarr.open(output[wsi1_2k_2k_svs], mode="r")
    assert 0.24 < np.mean(output_["predictions"][:]) < 0.25
    assert 0.48 < np.mean(output_["probabilities"][:]) < 0.52


def test_wsi_segmentor_annotationstore(
    sample_svs: Path, track_tmp_path: Path, caplog: pytest.CaptureFixture
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
        save_dir=track_tmp_path / "wsi_out_check",
        verbose=True,
        output_type="annotationstore",
    )

    assert output[sample_svs] == track_tmp_path / "wsi_out_check" / (
        sample_svs.stem + ".db"
    )

    # Return Probabilities
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask",
        batch_size=32,
        verbose=False,
    )
    # Return Probabilities is True
    output = segmentor.run(
        images=[sample_svs],
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=track_tmp_path / "wsi_prob_out_check",
        verbose=True,
        output_type="annotationstore",
    )

    assert output[sample_svs] == track_tmp_path / "wsi_prob_out_check" / (
        sample_svs.stem + ".db"
    )
    assert output[sample_svs].with_suffix(".zarr").exists()

    zarr_group = zarr.open(output[sample_svs].with_suffix(".zarr"), mode="r")
    assert "probabilities" in zarr_group
    assert "Probability maps cannot be saved as AnnotationStore." in caplog.text


def test_prepare_full_batch_low_memory(track_tmp_path: Path) -> None:
    """Test prepare_full_batch with low memory condition (disk-based zarr)."""
    # Create mock data
    batch_size = 10
    patch_h, patch_w, channels = 256, 256, 3
    rand = np.random.default_rng(12345)
    batch_output = rand.random((batch_size, patch_h, patch_w, channels)).astype(
        np.float32
    )

    # Create batch locations (x_start, y_start, x_end, y_end)
    batch_locs = np.array([[i * 100, 0, i * 100 + 100, 100] for i in range(batch_size)])

    # Create full output locations (all possible patch locations)
    num_full_locs = 50
    full_output_locs = np.array(
        [[i * 100, 0, i * 100 + 100, 100] for i in range(num_full_locs)]
    )

    output_locs = np.empty((0, 4), dtype=batch_locs.dtype)

    # Mock psutil.virtual_memory to simulate low memory
    mock_vm = mock.MagicMock()
    # Set very small available memory to force zarr usage
    mock_vm.available = 1024 * 1024  # 1 MB
    mock_vm.free = 1024 * 1024 * 1024  # 1 GB for checking purposes

    with mock.patch("psutil.virtual_memory", return_value=mock_vm):
        full_batch_output, updated_full_locs, updated_output_locs = prepare_full_batch(
            batch_output=batch_output,
            batch_locs=batch_locs,
            full_output_locs=full_output_locs,
            output_locs=output_locs,
            canvas_np=None,
            save_path=track_tmp_path / "test_zarr",
            memory_threshold=80,
            is_last=False,
        )

    # Verify that a zarr array was created (disk-based)
    assert isinstance(full_batch_output, zarr.Array)
    assert full_batch_output.shape[0] == batch_size
    assert full_batch_output.shape[1:] == (patch_h, patch_w, channels)

    # Check that batch_output was correctly placed
    for i in range(batch_size):
        np.testing.assert_array_almost_equal(
            full_batch_output[i], batch_output[i], decimal=5
        )

    # Verify output locations were updated correctly
    assert len(updated_output_locs) == batch_size
    assert len(updated_full_locs) == num_full_locs - batch_size

    # Clean up temp directories
    if (track_tmp_path / "test_zarr").exists():
        shutil.rmtree(track_tmp_path / "test_zarr")


def test_prepare_full_batch_sufficient_memory(track_tmp_path: Path) -> None:
    """Test prepare_full_batch with sufficient memory (in-memory numpy)."""
    # Create mock data
    batch_size = 1
    patch_h, patch_w, channels = 256, 256, 3
    rand = np.random.default_rng(12345)
    batch_output = rand.random((batch_size, patch_h, patch_w, channels)).astype(
        np.float32
    )

    # Create batch locations
    batch_locs = np.array([[i * 100, 0, i * 100 + 100, 100] for i in range(batch_size)])

    # Create full output locations
    num_full_locs = 50
    full_output_locs = np.array(
        [[i * 100, 0, i * 100 + 100, 100] for i in range(num_full_locs)]
    )

    output_locs = np.empty((0, 4), dtype=batch_locs.dtype)

    # Mock psutil.virtual_memory to simulate sufficient memory
    mock_vm = mock.MagicMock()
    # Set large available memory to force numpy usage
    mock_vm.available = 1024 * 1024 * 1024 * 100  # 100 GB
    mock_vm.free = 1024 * 1024 * 1024 * 100  # 100 GB

    with mock.patch("psutil.virtual_memory", return_value=mock_vm):
        full_batch_output, updated_full_locs, updated_output_locs = prepare_full_batch(
            batch_output=batch_output,
            batch_locs=batch_locs,
            full_output_locs=full_output_locs,
            output_locs=output_locs,
            canvas_np=None,
            save_path=track_tmp_path / "test_numpy",
            memory_threshold=80,
            is_last=False,
        )

    # Verify that a numpy array was created (in-memory)
    assert isinstance(full_batch_output, np.ndarray)
    assert full_batch_output.shape[0] == batch_size
    assert full_batch_output.shape[1:] == (patch_h, patch_w, channels)

    # Check that batch_output was correctly placed
    for i in range(batch_size):
        np.testing.assert_array_almost_equal(
            full_batch_output[i], batch_output[i], decimal=5
        )

    # Verify output locations were updated correctly
    assert len(updated_output_locs) == batch_size
    assert len(updated_full_locs) == num_full_locs - batch_size


def test_prepare_full_batch_with_existing_canvas(track_tmp_path: Path) -> None:
    """Test prepare_full_batch considering existing canvas memory footprint."""
    # Create mock data for first batch
    batch_size = 5
    patch_h, patch_w, channels = 128, 128, 2
    rand = np.random.default_rng(12345)
    batch_output = rand.random((batch_size, patch_h, patch_w, channels)).astype(
        np.float32
    )

    batch_locs = np.array([[i * 100, 0, i * 100 + 100, 100] for i in range(batch_size)])

    num_full_locs = 20
    full_output_locs = np.array(
        [[i * 100, 0, i * 100 + 100, 100] for i in range(num_full_locs)]
    )

    output_locs = np.empty((0, 4), dtype=batch_locs.dtype)

    # Create an existing canvas that would contribute to memory usage
    existing_canvas = rand.random((100, patch_h, patch_w, channels)).astype(np.float32)

    # Mock psutil to simulate moderate memory
    mock_vm = mock.MagicMock()
    # Set memory such that new array + existing canvas exceeds threshold
    mock_vm.available = (
        existing_canvas.nbytes + batch_output.nbytes
    )  # Just barely enough
    mock_vm.free = 1024 * 1024 * 1024  # 1 GB

    with mock.patch("psutil.virtual_memory", return_value=mock_vm):
        full_batch_output, _, _ = prepare_full_batch(
            batch_output=batch_output,
            batch_locs=batch_locs,
            full_output_locs=full_output_locs,
            output_locs=output_locs,
            canvas_np=existing_canvas,  # Pass existing canvas
            save_path=track_tmp_path / "test_with_canvas",
            memory_threshold=80,
            is_last=False,
        )

    # With existing canvas, should trigger zarr usage due to memory constraints
    assert isinstance(full_batch_output, zarr.Array)

    # Clean up
    if (track_tmp_path / "test_with_canvas").exists():
        shutil.rmtree(track_tmp_path / "test_with_canvas")


def test_prepare_full_batch_last_batch_padding(track_tmp_path: Path) -> None:
    """Test prepare_full_batch with is_last=True to verify padding behavior."""
    batch_size = 5
    patch_h, patch_w, channels = 64, 64, 1
    rand = np.random.default_rng(12345)
    batch_output = rand.random((batch_size, patch_h, patch_w, channels)).astype(
        np.float32
    )

    batch_locs = np.array([[i * 100, 0, i * 100 + 100, 100] for i in range(batch_size)])

    # Have more full_output_locs than batch_size to trigger padding
    num_full_locs = 10
    full_output_locs = np.array(
        [[i * 100, 0, i * 100 + 100, 100] for i in range(num_full_locs)]
    )

    output_locs = np.empty((0, 4), dtype=batch_locs.dtype)

    # Use sufficient memory for numpy array
    mock_vm = mock.MagicMock()
    mock_vm.available = 1024 * 1024 * 1024 * 10  # 10 GB
    mock_vm.free = 1024 * 1024 * 1024 * 10

    with mock.patch("psutil.virtual_memory", return_value=mock_vm):
        full_batch_output, updated_full_locs, updated_output_locs = prepare_full_batch(
            batch_output=batch_output,
            batch_locs=batch_locs,
            full_output_locs=full_output_locs,
            output_locs=output_locs,
            canvas_np=None,
            save_path=track_tmp_path / "test_last",
            memory_threshold=80,
            is_last=True,  # Last batch
        )

    # Verify padding was applied
    assert isinstance(full_batch_output, np.ndarray)
    expected_size = num_full_locs + batch_size  # batch_size + remaining
    assert full_batch_output.shape[0] == expected_size

    # Verify that remaining locations are zero-padded
    for i in range(batch_size, expected_size):
        assert np.all(full_batch_output[i] == 0)

    # All locations should be consumed
    assert len(updated_full_locs) == 0
    assert len(updated_output_locs) == num_full_locs


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_cli_model_single_file(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Test semantic segmentor CLI single file."""
    wsi4_512_512_svs = remote_sample("wsi4_512_512_svs")
    runner = CliRunner()
    models_wsi_result = runner.invoke(
        cli.main,
        [
            "semantic-segmentor",
            "--img-input",
            str(wsi4_512_512_svs),
            "--patch-mode",
            "False",
            "--output-path",
            str(track_tmp_path / "output"),
        ],
    )

    assert models_wsi_result.exit_code == 0
    assert (track_tmp_path / "output" / (wsi4_512_512_svs.stem + ".db")).exists()
