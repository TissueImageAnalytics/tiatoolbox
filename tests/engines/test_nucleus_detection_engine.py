"""Tests for NucleusDetector."""

import pathlib
import shutil
from collections.abc import Callable

import dask.array as da
import numpy as np
import pytest
import zarr

from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.misc import imwrite
from tiatoolbox.wsicore.wsireader import WSIReader

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def _rm_dir(path: pathlib.Path) -> None:
    """Helper func to remove directory."""
    if pathlib.Path(path).exists():
        shutil.rmtree(path, ignore_errors=True)


def test_nucleus_detector_wsi(remote_sample: Callable, tmp_path: pathlib.Path) -> None:
    """Test for nucleus detection engine."""
    mini_wsi_svs = pathlib.Path(remote_sample("wsi1_2k_2k_svs"))

    pretrained_model = "sccnn-conic"

    save_dir = tmp_path

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
    )

    store = SQLiteStore.open(save_dir / "wsi1_2k_2k.db")
    assert len(store.values()) == 2617
    store.close()

    result_path = nucleus_detector.run(
        patch_mode=False,
        device=device,
        output_type="zarr",
        memory_threshold=50,
        images=[mini_wsi_svs],
        save_dir=save_dir,
        overwrite=True,
    )

    zarr_path = result_path[mini_wsi_svs]
    zarr_group = zarr.open(zarr_path, mode="r")
    xs = zarr_group["x"][:]
    ys = zarr_group["y"][:]
    types = zarr_group["types"][:]
    probs = zarr_group["probs"][:]
    assert len(xs) == 2617
    assert len(ys) == 2617
    assert len(types) == 2617
    assert len(probs) == 2617

    nucleus_detector.drop_keys = ["probs"]
    result_path = nucleus_detector.run(
        patch_mode=False,
        device=device,
        output_type="zarr",
        memory_threshold=50,
        images=[mini_wsi_svs],
        save_dir=save_dir,
        overwrite=True,
    )

    zarr_path = result_path[mini_wsi_svs]
    zarr_group = zarr.open(zarr_path, mode="r")
    xs = zarr_group["x"][:]
    ys = zarr_group["y"][:]
    types = zarr_group["types"][:]
    probs = zarr_group.get("probs", None)
    assert probs is None
    assert len(xs) == 2617
    assert len(ys) == 2617
    assert len(types) == 2617

    _rm_dir(save_dir)


def test_nucleus_detector_patch_annotation_store_output(
    remote_sample: Callable, tmp_path: pathlib.Path
) -> None:
    """Test for nucleus detection engine in patch mode."""
    mini_wsi_svs = pathlib.Path(remote_sample("wsi1_2k_2k_svs"))

    wsi_reader = WSIReader.open(mini_wsi_svs)
    patch_1 = wsi_reader.read_bounds(
        (30, 30, 61, 61),
        resolution=0.25,
        units="mpp",
        coord_space="resolution",
    )
    patch_2 = np.zeros((31, 31, 3), dtype=np.uint8)

    pretrained_model = "sccnn-conic"

    save_dir = tmp_path

    nucleus_detector = NucleusDetector(model=pretrained_model)
    _ = nucleus_detector.run(
        patch_mode=True,
        device=device,
        output_type="annotationstore",
        memory_threshold=50,
        images=[patch_1, patch_2],
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

    imwrite(save_dir / "patch_0.png", patch_1)
    imwrite(save_dir / "patch_1.png", patch_2)
    _ = nucleus_detector.run(
        patch_mode=True,
        device=device,
        output_type="annotationstore",
        memory_threshold=50,
        images=[save_dir / "patch_0.png", save_dir / "patch_1.png"],
        save_dir=save_dir,
        overwrite=True,
    )

    store_1 = SQLiteStore.open(save_dir / "patch_0.db")
    assert len(store_1.values()) == 1
    store_1.close()

    store_2 = SQLiteStore.open(save_dir / "patch_1.db")
    assert len(store_2.values()) == 0
    store_2.close()

    _rm_dir(save_dir)


def test_nucleus_detector_patches_dict_output(
    remote_sample: Callable,
) -> None:
    """Test for nucleus detection engine in patch mode."""
    mini_wsi_svs = pathlib.Path(remote_sample("wsi1_2k_2k_svs"))

    wsi_reader = WSIReader.open(mini_wsi_svs)
    patch_1 = wsi_reader.read_bounds(
        (30, 30, 61, 61),
        resolution=0.25,
        units="mpp",
        coord_space="resolution",
    )
    patch_2 = np.zeros((31, 31, 3), dtype=np.uint8)

    pretrained_model = "sccnn-conic"

    nucleus_detector = NucleusDetector(model=pretrained_model)

    output_dict = nucleus_detector.run(
        patch_mode=True,
        device=device,
        output_type="dict",
        memory_threshold=50,
        images=[patch_1, patch_2],
        save_dir=None,
        class_dict=None,
    )
    output_dict = output_dict["predictions"]
    assert len(output_dict["x"]) == 2
    assert len(output_dict["y"]) == 2
    assert len(output_dict["types"]) == 2
    assert len(output_dict["probs"]) == 2
    assert len(output_dict["x"][0]) == 1
    assert len(output_dict["x"][1]) == 0
    assert len(output_dict["y"][0]) == 1
    assert len(output_dict["y"][1]) == 0
    assert len(output_dict["types"][0]) == 1
    assert len(output_dict["types"][1]) == 0
    assert len(output_dict["probs"][0]) == 1
    assert len(output_dict["probs"][1]) == 0


def test_nucleus_detector_patches_zarr_output(
    remote_sample: Callable, tmp_path: pathlib.Path
) -> None:
    """Test for nucleus detection engine in patch mode."""
    mini_wsi_svs = pathlib.Path(remote_sample("wsi1_2k_2k_svs"))
    wsi_reader = WSIReader.open(mini_wsi_svs)
    patch_1 = wsi_reader.read_bounds(
        (30, 30, 61, 61),
        resolution=0.25,
        units="mpp",
        coord_space="resolution",
    )
    patch_2 = np.zeros((31, 31, 3), dtype=np.uint8)

    pretrained_model = "sccnn-conic"

    nucleus_detector = NucleusDetector(model=pretrained_model)

    save_dir = tmp_path

    output_path = nucleus_detector.run(
        patch_mode=True,
        device=device,
        output_type="zarr",
        memory_threshold=50,
        images=[patch_1, patch_2],
        save_dir=save_dir,
        class_dict=None,
        overwrite=True,
    )

    zarr_group = zarr.open(output_path, mode="r")
    output_dict = {
        "x": zarr_group["x"][:],
        "y": zarr_group["y"][:],
        "types": zarr_group["types"][:],
        "probs": zarr_group["probs"][:],
        "patch_offsets": zarr_group["patch_offsets"][:],
    }

    assert len(output_dict["x"]) == 1
    assert len(output_dict["y"]) == 1
    assert len(output_dict["types"]) == 1
    assert len(output_dict["probs"]) == 1
    assert len(output_dict["patch_offsets"]) == 3

    patch_1_start, patch_1_end = (
        output_dict["patch_offsets"][0],
        output_dict["patch_offsets"][1],
    )
    patch_2_start, patch_2_end = (
        output_dict["patch_offsets"][1],
        output_dict["patch_offsets"][2],
    )
    assert len(output_dict["x"][patch_1_start:patch_1_end]) == 1
    assert len(output_dict["x"][patch_2_start:patch_2_end]) == 0

    assert len(output_dict["y"][patch_1_start:patch_1_end]) == 1
    assert len(output_dict["y"][patch_2_start:patch_2_end]) == 0

    assert len(output_dict["types"][patch_1_start:patch_1_end]) == 1
    assert len(output_dict["types"][patch_2_start:patch_2_end]) == 0

    assert len(output_dict["probs"][patch_1_start:patch_1_end]) == 1
    assert len(output_dict["probs"][patch_2_start:patch_2_end]) == 0

    _rm_dir(save_dir)


def test_centroid_maps_to_detection_arrays() -> None:
    """Convert centroid maps to detection arrays."""
    detection_maps = np.zeros((4, 4, 2), dtype=np.float32)
    detection_maps[1, 1, 0] = 1.0
    detection_maps[2, 3, 1] = 0.5
    detection_maps = da.from_array(detection_maps, chunks=(2, 2, 2))

    detections = NucleusDetector._centroid_maps_to_detection_arrays(detection_maps)

    xs = detections["x"]
    ys = detections["y"]
    types = detections["types"]
    probs = detections["probs"]

    np.testing.assert_array_equal(xs, np.array([1, 3], dtype=np.uint32))
    np.testing.assert_array_equal(ys, np.array([1, 2], dtype=np.uint32))
    np.testing.assert_array_equal(types, np.array([0, 1], dtype=np.uint32))
    np.testing.assert_array_equal(probs, np.array([1.0, 0.5], dtype=np.float32))


def test_write_detection_arrays_to_store() -> None:
    """Test writing detection arrays to annotation store."""
    detection_arrays = {
        "x": np.array([1, 3], dtype=np.uint32),
        "y": np.array([1, 2], dtype=np.uint32),
        "types": np.array([0, 1], dtype=np.uint32),
        "probs": np.array([1.0, 0.5], dtype=np.float32),
    }

    store = NucleusDetector.write_detection_arrays_to_store(detection_arrays)
    assert len(store.values()) == 2

    detection_arrays = {
        "x": np.array([1], dtype=np.uint32),
        "y": np.array([1, 2], dtype=np.uint32),
        "types": np.array([0], dtype=np.uint32),
        "probs": np.array([1.0, 0.5], dtype=np.float32),
    }
    with pytest.raises(
        ValueError,
        match=r"Detection record lengths are misaligned.",
    ):
        _ = NucleusDetector.write_detection_arrays_to_store(detection_arrays)


def test_write_detection_records_to_store_no_class_dict() -> None:
    """Test writing detection records to annotation store."""
    detection_records = (np.array([1]), np.array([2]), np.array([0]), np.array([1.0]))

    dummy_store = SQLiteStore()
    total = NucleusDetector._write_detection_records_to_store(
        detection_records, store=dummy_store, scale_factor=(1.0, 1.0), class_dict=None
    )
    assert len(dummy_store.values()) == 1
    assert total == 1
    annotation = next(iter(dummy_store.values()))
    assert annotation.properties["type"] == 0
    dummy_store.close()
