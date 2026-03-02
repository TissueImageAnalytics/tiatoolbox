"""Tests for NucleusDetector."""

import json
from collections.abc import Callable
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import zarr
from click.testing import CliRunner

from tiatoolbox import cli
from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models.engines.nucleus_detector import (
    NucleusDetector,
    _write_detection_arrays_to_store,
    save_detection_arrays_to_qupath_json,
    save_detection_arrays_to_store,
)
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.misc import imwrite
from tiatoolbox.wsicore.wsireader import WSIReader

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def test_centroid_maps_to_detection_arrays() -> None:
    """Convert centroid maps to detection arrays."""
    detection_maps = np.zeros((4, 4, 2), dtype=np.float32)
    detection_maps[1, 1, 0] = 1.0
    detection_maps[2, 3, 1] = 0.5
    detection_maps = da.from_array(detection_maps, chunks=(2, 2, 2))

    detections = NucleusDetector._centroid_maps_to_detection_arrays(detection_maps)

    xs = detections["x"]
    ys = detections["y"]
    classes = detections["classes"]
    probs = detections["probabilities"]

    np.testing.assert_array_equal(xs, np.array([1, 3], dtype=np.uint32))
    np.testing.assert_array_equal(ys, np.array([1, 2], dtype=np.uint32))
    np.testing.assert_array_equal(classes, np.array([0, 1], dtype=np.uint32))
    np.testing.assert_array_equal(probs, np.array([1.0, 0.5], dtype=np.float32))


def test_write_detection_arrays_to_store() -> None:
    """Test writing detection arrays to annotation store."""
    detection_arrays = {
        "x": np.array([1, 3], dtype=np.uint32),
        "y": np.array([1, 2], dtype=np.uint32),
        "classes": np.array([0, 1], dtype=np.uint32),
        "probabilities": np.array([1.0, 0.5], dtype=np.float32),
    }

    store = save_detection_arrays_to_store(detection_arrays)
    assert len(store.values()) == 2

    detection_arrays = {
        "x": np.array([1], dtype=np.uint32),
        "y": np.array([1, 2], dtype=np.uint32),
        "classes": np.array([0], dtype=np.uint32),
        "probabilities": np.array([1.0, 0.5], dtype=np.float32),
    }
    with pytest.raises(
        ValueError,
        match=r"Detection record lengths are misaligned.",
    ):
        _ = save_detection_arrays_to_store(detection_arrays)


def test_write_detection_arrays_to_qupath() -> None:
    """Test writing detection arrays to QuPath JSON."""
    detection_arrays = {
        "x": np.array([1, 3], dtype=np.uint32),
        "y": np.array([1, 2], dtype=np.uint32),
        "classes": np.array([0, 1], dtype=np.uint32),
        "probabilities": np.array([1.0, 0.5], dtype=np.float32),
    }

    json_ = save_detection_arrays_to_qupath_json(detection_arrays)
    assert len(json_.values()) == 2

    detection_arrays = {
        "x": np.array([1], dtype=np.uint32),
        "y": np.array([1, 2], dtype=np.uint32),
        "classes": np.array([0], dtype=np.uint32),
        "probabilities": np.array([1.0, 0.5], dtype=np.float32),
    }
    with pytest.raises(
        ValueError,
        match=r"Detection record lengths are misaligned.",
    ):
        _ = save_detection_arrays_to_store(detection_arrays)


def test_write_detection_records_to_store_no_class_dict() -> None:
    """Test writing detection records to annotation store."""
    detection_records = (np.array([1]), np.array([2]), np.array([0]), np.array([1.0]))

    dummy_store = SQLiteStore()
    total = _write_detection_arrays_to_store(
        detection_records, store=dummy_store, scale_factor=(1.0, 1.0), class_dict=None
    )
    assert len(dummy_store.values()) == 1
    assert total == 1
    annotation = next(iter(dummy_store.values()))
    assert annotation.properties["type"] == 0
    dummy_store.close()


def test_nucleus_detector_patch_annotation_store_output(
    remote_sample: Callable, track_tmp_path: Path, rm_dir: Callable
) -> None:
    """Test for nucleus detection engine in patch mode."""
    mini_wsi_svs = Path(remote_sample("wsi4_512_512_svs"))

    wsi_reader = WSIReader.open(mini_wsi_svs)
    patch_1 = wsi_reader.read_bounds(
        (120, 120, 151, 151),
        resolution=0.25,
        units="mpp",
        coord_space="resolution",
    )
    patch_2 = np.zeros((31, 31, 3), dtype=np.uint8)

    pretrained_model = "sccnn-conic"

    save_dir = track_tmp_path / "output"

    nucleus_detector = NucleusDetector(model=pretrained_model)
    _ = nucleus_detector.run(
        images=[patch_1, patch_2],
        patch_mode=True,
        device=device,
        output_type="annotationstore",
        memory_threshold=50,
        save_dir=save_dir,
        overwrite=True,
        class_dict=None,
    )

    store_1 = SQLiteStore.open(save_dir / "0.db")
    assert len(store_1.values()) == 1
    store_1.close()

    store_2 = SQLiteStore.open(save_dir / "1.db")
    assert len(store_2.values()) == 0
    store_2.close()

    image_dir = track_tmp_path / "inputs"
    image_dir.mkdir()
    imwrite(image_dir / "patch_0.png", patch_1)
    imwrite(image_dir / "patch_1.png", patch_2)
    _ = nucleus_detector.run(
        patch_mode=True,
        device=device,
        output_type="annotationstore",
        memory_threshold=50,
        images=[image_dir / "patch_0.png", image_dir / "patch_1.png"],
        save_dir=save_dir,
        overwrite=True,
    )

    store_1 = SQLiteStore.open(save_dir / "patch_0.db")
    assert len(store_1.values()) == 1
    store_1.close()

    store_2 = SQLiteStore.open(save_dir / "patch_1.db")
    assert len(store_2.values()) == 0
    store_2.close()

    _ = nucleus_detector.run(
        patch_mode=True,
        device=device,
        output_type="qupath",
        memory_threshold=50,
        images=[image_dir / "patch_0.png", image_dir / "patch_1.png"],
        save_dir=save_dir,
        overwrite=True,
    )

    with Path.open(save_dir / "patch_0.json", "r") as f:
        data_1 = json.load(f)
    features_1 = data_1.get("features", [])
    assert len(features_1) == 1

    with Path.open(save_dir / "patch_1.json", "r") as f:
        data_2 = json.load(f)
    features_2 = data_2.get("features", [])

    assert len(features_2) == 0

    rm_dir(save_dir)


def test_nucleus_detector_patches_dict_output(
    remote_sample: Callable,
) -> None:
    """Test for nucleus detection engine in patch mode."""
    mini_wsi_svs = Path(remote_sample("wsi4_512_512_svs"))

    wsi_reader = WSIReader.open(mini_wsi_svs)
    patch_1 = wsi_reader.read_bounds(
        (120, 120, 151, 151),
        resolution=0.25,
        units="mpp",
        coord_space="resolution",
    )
    patch_2 = np.zeros_like(patch_1)

    model = "sccnn-conic"

    nucleus_detector = NucleusDetector(model=model)

    output_dict = nucleus_detector.run(
        patch_mode=True,
        device=device,
        output_type="dict",
        memory_threshold=50,
        images=np.stack([patch_1, patch_2], axis=0),
        save_dir=None,
        class_dict=None,
        return_probabilities=True,
    )
    assert len(output_dict["x"]) == 2
    assert len(output_dict["y"]) == 2
    assert len(output_dict["classes"]) == 2
    assert len(output_dict["probabilities"]) == 2
    assert len(output_dict["x"][0]) == 1
    assert len(output_dict["x"][1]) == 0
    assert len(output_dict["y"][0]) == 1
    assert len(output_dict["y"][1]) == 0
    assert len(output_dict["classes"][0]) == 1
    assert len(output_dict["classes"][1]) == 0
    assert len(output_dict["probabilities"][0]) == 1
    assert len(output_dict["probabilities"][1]) == 0


def test_nucleus_detector_patches_zarr_output(
    remote_sample: Callable, track_tmp_path: Path, rm_dir: Callable
) -> None:
    """Test for nucleus detection engine in patch mode."""
    mini_wsi_svs = Path(remote_sample("wsi4_512_512_svs"))
    wsi_reader = WSIReader.open(mini_wsi_svs)
    patch_1 = wsi_reader.read_bounds(
        (120, 120, 151, 151),
        resolution=0.25,
        units="mpp",
        coord_space="resolution",
    )
    patch_2 = np.zeros_like(patch_1)

    pretrained_model = "sccnn-conic"

    nucleus_detector = NucleusDetector(model=pretrained_model)

    save_dir = track_tmp_path

    output_path = nucleus_detector.run(
        patch_mode=True,
        device=device,
        output_type="zarr",
        memory_threshold=50,
        images=np.stack([patch_1, patch_2], axis=0),
        save_dir=save_dir,
        class_dict=None,
        overwrite=True,
        return_probabilities=True,
    )

    output_zarr = zarr.open(output_path, mode="r")

    assert output_zarr["x"][0].size == 1
    assert output_zarr["x"][1].size == 0
    assert output_zarr["y"][0].size == 1
    assert output_zarr["y"][1].size == 0
    assert output_zarr["classes"][0].size == 1
    assert output_zarr["classes"][1].size == 0
    assert output_zarr["probabilities"][0].size == 1
    assert output_zarr["probabilities"][1].size == 0

    rm_dir(save_dir)


def test_nucleus_detector_wsi(
    remote_sample: Callable, track_tmp_path: Path, rm_dir: Callable
) -> None:
    """Test for nucleus detection engine."""
    mini_wsi_svs = Path(remote_sample("wsi4_512_512_svs"))

    pretrained_model = "sccnn-conic"

    save_dir = track_tmp_path

    nucleus_detector = NucleusDetector(model=pretrained_model)
    nucleus_detector.drop_keys = []
    _ = nucleus_detector.run(
        patch_mode=False,
        device=device,
        output_type="annotationstore",
        memory_threshold=50,
        images=[mini_wsi_svs],
        save_dir=save_dir,
        overwrite=True,
        batch_size=8,
        class_dict={0: "test_nucleus"},
        min_distance=5,
        tile_shape=(2048, 2048),
    )

    store = SQLiteStore.open(save_dir / "wsi4_512_512.db")
    assert 245 <= len(store.values()) <= 255
    annotation = next(iter(store.values()))
    assert annotation.properties["type"] == "test_nucleus"
    store.close()

    # QuPath
    nucleus_detector.drop_keys = []
    _ = nucleus_detector.run(
        patch_mode=False,
        device=device,
        output_type="qupath",
        memory_threshold=50,
        images=[mini_wsi_svs],
        save_dir=save_dir,
        overwrite=True,
        batch_size=8,
        class_dict={0: "test_nucleus"},
        min_distance=5,
        postproc_tile_shape=(2048, 2048),
    )

    with Path.open(save_dir / "wsi4_512_512.json", "r") as f:
        qupath_json = json.load(f)
    features: list[dict] = qupath_json.get("features", [])
    assert 245 <= len(features) <= 255
    first = features[0]
    # Classification name
    assert first["properties"]["classification"]["name"] == "test_nucleus"

    # Check cached centroid maps are removed
    temp_zarr_files = save_dir / "wsi4_512_512.zarr"
    assert not temp_zarr_files.exists()

    nucleus_detector.drop_keys = ["probabilities"]
    result_path = nucleus_detector.run(
        patch_mode=False,
        device=device,
        output_type="zarr",
        memory_threshold=50,
        images=[mini_wsi_svs],
        save_dir=save_dir,
        overwrite=True,
        batch_size=8,
    )
    print("Result path:", result_path)

    zarr_path = result_path[mini_wsi_svs]
    print("Zarr path:", zarr_path)
    zarr_group = zarr.open(zarr_path, mode="r")

    # Check cached centroid maps are removed
    assert "centroid_maps" not in zarr_group

    xs = zarr_group["x"][:]
    ys = zarr_group["y"][:]
    classes = zarr_group["classes"][:]
    probs = zarr_group.get("probabilities", None)
    assert probs is None
    assert 245 <= len(xs) <= 255
    assert 245 <= len(ys) <= 255
    assert 245 <= len(classes) <= 255

    rm_dir(save_dir)


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_cli_model_single_file(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Test nucleus detector CLI single file."""
    runner = CliRunner()
    mini_wsi_svs = Path(remote_sample("wsi4_512_512_svs"))
    models_wsi_result = runner.invoke(
        cli.main,
        [
            "nucleus-detector",
            "--img-input",
            str(mini_wsi_svs),
            "--patch-mode",
            "False",
            "--output-path",
            str(track_tmp_path / "output"),
        ],
    )

    assert models_wsi_result.exit_code == 0, models_wsi_result.output
    assert (track_tmp_path / "output" / ("wsi4_512_512" + ".db")).exists()
    mini_wsi_svs.unlink()
