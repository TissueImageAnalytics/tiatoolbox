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
from huggingface_hub import hf_hub_download

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

    annotation_types = set()

    for item in annotations_properties:
        for json_str in item:
            probs = json.loads(json_str)
            if "type" in probs:
                annotation_types.add(probs.pop("type"))
    # When class_dict is none, types are assigned as 0, 1, ...
    assert 0 in annotation_types
    assert 1 in annotation_types

    assert annotations_properties is not None


def _test_qupath_output_patch(output: Path) -> None:
    """Helper function to test QuPath JSON output for a patch."""
    with Path.open(output) as f:
        data = json.load(f)

    assert "features" in data
    features = data["features"]
    assert len(features) > 0

    geometry_types = []
    class_values = set()

    for feat in features:
        # geometry type
        geom = feat.get("geometry", {})
        geometry_types.append(geom.get("type"))

        # class index (you stored this as class_value)
        class_val = feat.get("class_value")
        if class_val is not None:
            class_values.add(class_val)

    # Check geometry type
    assert "Polygon" in geometry_types

    # When class_dict is None, types are assigned as 0, 1, ...
    assert 0 in class_values
    assert 1 in class_values

    # Basic sanity check
    assert features is not None


def test_semantic_segmentor_tiles(track_tmp_path: Path) -> None:
    """Tests SemanticSegmentor on image tiles with no mpp metadata."""
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask", batch_size=32, verbose=False, device=device
    )

    sample_image = Path(
        hf_hub_download(
            repo_id="TIACentre/TIAToolBox_Remote_Samples",
            subfolder="sample_imgs",
            filename="breast_tissue.jpg",
            repo_type="dataset",
            local_dir=track_tmp_path,
        )
    )

    inputs = [sample_image]

    output = segmentor.run(
        images=inputs,
        device=device,
        patch_mode=False,
        auto_get_mask=False,
        save_dir=track_tmp_path / "output",
        input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        patch_input_shape=(1024, 1024),
    )
    print(output)
    output = zarr.open(output[sample_image], mode="r")

    assert output["predictions"].shape == (2048, 3584)

    sample_image.unlink()


def test_save_annotation_store(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Test for saving output as annotation store."""
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask", batch_size=32, verbose=False, device=device
    )

    # Test str input
    sample_image = remote_sample("thumbnail-1k-1k")

    inputs = [Path(sample_image)]

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


def test_save_qupath_json(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Test for saving output as annotation store."""
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask", batch_size=32, verbose=False, device=device
    )

    # Test str input
    sample_image = remote_sample("thumbnail-1k-1k")

    inputs = [Path(sample_image)]

    output = segmentor.run(
        images=inputs,
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=True,
        save_dir=track_tmp_path / "output1",
        output_type="qupath",
        verbose=True,
    )

    assert output[0] == track_tmp_path / "output1" / (sample_image.stem + ".json")
    assert len(output) == 1
    _test_qupath_output_patch(output[0])


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

    assert "Probability maps cannot be saved as AnnotationStore" in caplog.text

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


def test_merge_batch_to_canvas_with_dask_arrays() -> None:
    """Test merge_batch_to_canvas with Dask arrays to trigger conversion."""
    # Create numpy blocks and locations first
    blocks_np = np.array([np.ones((2, 2, 1)), np.ones((2, 2, 1)) * 2])
    output_locations_np = np.array([[0, 0, 2, 2], [2, 0, 4, 2]])

    # Convert to Dask arrays to trigger the isinstance checks
    blocks_dask = da.from_array(blocks_np, chunks=blocks_np.shape)
    output_locations_dask = da.from_array(
        output_locations_np, chunks=output_locations_np.shape
    )

    merged_shape = (2, 4, 1)

    # Call with Dask arrays - should convert to numpy internally
    canvas, count = semantic_segmentor.merge_batch_to_canvas(
        blocks_dask, output_locations_dask, merged_shape
    )

    # Verify results match expected output
    assert np.array_equal(canvas[:, :2, :], np.ones((2, 2, 1)))
    assert np.array_equal(canvas[:, 2:, :], np.ones((2, 2, 1)) * 2)
    assert np.array_equal(count, np.ones((2, 4, 1)))

    # Verify the function returns numpy arrays
    assert isinstance(canvas, np.ndarray)
    assert isinstance(count, np.ndarray)


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
    assert "Probability maps cannot be saved as AnnotationStore or JSON." in caplog.text


def test_wsi_segmentor_qupath_json(sample_svs: Path, track_tmp_path: Path) -> None:
    """Test SemanticSegmentor for WSIs with QuPath JSON output."""
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
        verbose=False,
        output_type="QuPath",
    )

    assert output[sample_svs] == track_tmp_path / "wsi_out_check" / (
        sample_svs.stem + ".json"
    )


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


def test_prepare_full_batch_last_batch_padding_zarr(track_tmp_path: Path) -> None:
    """Test prepare_full_batch with is_last=True and low memory (zarr resize path)."""
    batch_size = 5
    patch_h, patch_w, channels = 512, 512, 3  # Larger patches to exceed memory
    rand = np.random.default_rng(12345)
    batch_output = rand.random((batch_size, patch_h, patch_w, channels)).astype(
        np.float32
    )

    batch_locs = np.array([[i * 100, 0, i * 100 + 100, 100] for i in range(batch_size)])

    # Have more full_output_locs than batch_size to trigger padding
    num_full_locs = 15  # More locations to increase array size
    full_output_locs = np.array(
        [[i * 100, 0, i * 100 + 100, 100] for i in range(num_full_locs)]
    )

    output_locs = np.empty((0, 4), dtype=batch_locs.dtype)

    # Mock psutil to simulate low memory (trigger zarr path)
    mock_vm = mock.MagicMock()
    mock_vm.available = 1024 * 1024  # 1 MB - force zarr
    mock_vm.free = 1024 * 1024 * 1024  # 1 GB for checking

    with mock.patch("psutil.virtual_memory", return_value=mock_vm):
        full_batch_output, updated_full_locs, updated_output_locs = prepare_full_batch(
            batch_output=batch_output,
            batch_locs=batch_locs,
            full_output_locs=full_output_locs,
            output_locs=output_locs,
            canvas_np=None,
            save_path=track_tmp_path / "test_last_zarr",
            memory_threshold=80,
            is_last=True,  # Last batch with low memory
        )

    # Verify that a zarr array was created (disk-based)
    assert isinstance(full_batch_output, zarr.Array)

    # Verify padding was applied (zarr resize happened)
    expected_size = 15
    assert full_batch_output.shape[0] == expected_size

    # Check that batch_output was correctly placed
    for i in range(batch_size):
        np.testing.assert_array_almost_equal(
            full_batch_output[i], batch_output[i], decimal=5
        )

    # Verify that remaining locations are zero-padded
    for i in range(batch_size, expected_size):
        assert np.all(full_batch_output[i] == 0)

    # All locations should be consumed
    assert len(updated_full_locs) == 0
    assert len(updated_output_locs) == num_full_locs


def test_infer_wsi_cleanup_full_batch_tmp(track_tmp_path: Path) -> None:
    """Test that infer_wsi cleans up full_batch_tmp directory after processing."""
    # Create a minimal segmentor
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask", batch_size=2, verbose=False, device=device
    )

    # Create save path
    save_path = track_tmp_path / "test_cleanup" / "output.zarr"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the full_batch_tmp directory that would be created during processing
    full_batch_tmp_dir = save_path.with_name("full_batch_tmp")
    full_batch_tmp_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy file inside to verify it gets removed
    dummy_file = full_batch_tmp_dir / "dummy.txt"
    dummy_file.write_text("test")

    # Verify directory exists before running
    assert full_batch_tmp_dir.exists()
    assert dummy_file.exists()

    # Mock the dataloader to return minimal data
    mock_dataloader = mock.MagicMock()
    mock_batch = {
        "image": torch.randn(1, 3, 256, 256),
        "output_locs": torch.tensor([[0, 0, 256, 256]]),
        "coords": np.array([[0, 0, 256, 256]]),  # Required by _get_coordinates
    }
    mock_dataloader.__iter__ = mock.MagicMock(return_value=iter([mock_batch]))
    mock_dataloader.__len__ = mock.MagicMock(return_value=1)
    mock_dataloader.dataset = mock.MagicMock()
    mock_dataloader.dataset.outputs = np.array([[0, 0, 256, 256]])
    mock_dataloader.dataset.full_outputs = np.array([[0, 0, 256, 256]])

    rand = np.random.default_rng(12345)
    # Mock the model's infer_batch method
    with mock.patch.object(
        segmentor.model,
        "infer_batch",
        return_value=rand.random((1, 256, 256, 2)).astype(np.float32),
    ):
        # Run infer_wsi
        _ = segmentor.infer_wsi(
            dataloader=mock_dataloader,
            save_path=save_path,
            memory_threshold=80,
        )

    # Verify that full_batch_tmp directory was cleaned up
    assert not full_batch_tmp_dir.exists(), "full_batch_tmp directory should be removed"
    assert not dummy_file.exists(), "Files in full_batch_tmp should be removed"


def test_save_predictions_cleanup_zarr(track_tmp_path: Path) -> None:
    """Test that save_predictions cleans up .zarr file.

    When saving as annotationstore without probabilities,
    zarr directory should be removed.

    """
    # Create a minimal segmentor
    segmentor = SemanticSegmentor(
        model="fcn-tissue_mask", batch_size=2, verbose=False, device=device
    )
    segmentor.patch_mode = True
    segmentor.images = [Path("dummy.png")]

    # Create save path
    save_path = track_tmp_path / "output" / "result.db"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a .zarr file that should be cleaned up
    zarr_path = save_path.with_suffix(".zarr")
    zarr_path.mkdir(parents=True, exist_ok=True)

    # Create a dummy file inside the zarr directory
    dummy_file = zarr_path / ".zarray"
    dummy_file.write_text('{"test": "data"}')

    # Verify zarr directory exists before running
    assert zarr_path.exists()
    assert dummy_file.exists()

    rand = np.random.default_rng(12345)
    # Create minimal processed predictions
    processed_predictions = {
        "predictions": [rand.integers(0, 2, size=(256, 256), dtype=np.uint8)]
    }

    # Mock the parent class save_predictions to return predictions as dict
    with mock.patch.object(
        semantic_segmentor.PatchPredictor,
        "save_predictions",
        return_value=processed_predictions,
    ):
        # Call save_predictions with annotationstore output type
        # and return_probabilities=False
        result = segmentor.save_predictions(
            processed_predictions=processed_predictions,
            output_type="annotationstore",
            save_path=save_path,
            return_probabilities=False,  # This should trigger zarr cleanup
            class_dict={0: "background", 1: "tissue"},
            scale_factor=(1.0, 1.0),
        )

    # Verify that .zarr directory was cleaned up
    assert not zarr_path.exists(), ".zarr directory should be removed"
    assert not dummy_file.exists(), "Files in .zarr directory should be removed"

    # Verify that .db file was created
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].suffix == ".db"


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
