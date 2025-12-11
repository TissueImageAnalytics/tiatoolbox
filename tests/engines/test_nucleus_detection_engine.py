"""Tests for NucleusDetector."""

import pathlib
import shutil
from collections.abc import Callable

import dask.array as da
import numpy as np
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


def check_output(path: pathlib.Path) -> None:
    """Check NucleusDetector output."""


def test_nucleus_detector_wsi(remote_sample: Callable, tmp_path: pathlib.Path) -> None:
    """Test for nucleus detection engine."""
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))

    pretrained_model = "mapde-conic"

    save_dir = tmp_path

    nucleus_detector = NucleusDetector(model=pretrained_model)
    _ = nucleus_detector.run(
        patch_mode=False,
        device=device,
        output_type="annotationstore",
        memory_threshold=50,
        images=[mini_wsi_svs],
        save_dir=save_dir,
        overwrite=True,
    )

    store = SQLiteStore.open(save_dir / "wsi4_512_512.db")
    assert len(store.values()) == 281
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
    assert len(xs) == 281
    assert len(ys) == 281
    assert len(types) == 281
    assert len(probs) == 281

    _rm_dir(save_dir)


def test_nucleus_detector_patch_annotation_store_output(
    remote_sample: Callable, tmp_path: pathlib.Path
) -> None:
    """Test for nucleus detection engine in patch mode."""
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))

    wsi_reader = WSIReader.open(mini_wsi_svs)
    patch_1 = wsi_reader.read_rect((0, 0), (252, 252), resolution=0.5, units="mpp")
    patch_2 = wsi_reader.read_rect((252, 252), (252, 252), resolution=0.5, units="mpp")
    patch_3 = np.zeros((252, 252, 3), dtype=np.uint8)

    pretrained_model = "mapde-conic"

    save_dir = tmp_path

    nucleus_detector = NucleusDetector(model=pretrained_model)
    _ = nucleus_detector.run(
        patch_mode=True,
        device=device,
        output_type="annotationstore",
        memory_threshold=50,
        images=[patch_1, patch_2, patch_3],
        save_dir=save_dir,
        overwrite=True,
        class_dict=None,
    )

    store_1 = SQLiteStore.open(save_dir / "0.db")
    assert len(store_1.values()) == 270
    store_1.close()

    store_2 = SQLiteStore.open(save_dir / "1.db")
    assert len(store_2.values()) == 52
    store_2.close()

    store_3 = SQLiteStore.open(save_dir / "2.db")
    assert len(store_3.values()) == 0
    store_3.close()

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
    assert len(store_1.values()) == 270
    store_1.close()

    store_2 = SQLiteStore.open(save_dir / "patch_1.db")
    assert len(store_2.values()) == 52
    store_2.close()

    _rm_dir(save_dir)


def test_nucleus_detector_patches_dict_output(
    remote_sample: Callable,
) -> None:
    """Test for nucleus detection engine in patch mode."""
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))

    wsi_reader = WSIReader.open(mini_wsi_svs)
    patch_1 = wsi_reader.read_rect((0, 0), (252, 252), resolution=0.5, units="mpp")
    patch_2 = wsi_reader.read_rect((252, 252), (252, 252), resolution=0.5, units="mpp")
    patch_3 = np.zeros((252, 252, 3), dtype=np.uint8)

    pretrained_model = "mapde-conic"

    nucleus_detector = NucleusDetector(model=pretrained_model)

    output_dict = nucleus_detector.run(
        patch_mode=True,
        device=device,
        output_type="dict",
        memory_threshold=50,
        images=[patch_1, patch_2, patch_3],
        save_dir=None,
        class_dict=None,
    )
    assert len(output_dict["x"]) == 3
    assert len(output_dict["y"]) == 3
    assert len(output_dict["types"]) == 3
    assert len(output_dict["probs"]) == 3
    assert len(output_dict["x"][0]) == 270
    assert len(output_dict["x"][1]) == 52
    assert len(output_dict["x"][2]) == 0
    assert len(output_dict["y"][0]) == 270
    assert len(output_dict["y"][1]) == 52
    assert len(output_dict["y"][2]) == 0
    assert len(output_dict["types"][0]) == 270
    assert len(output_dict["types"][1]) == 52
    assert len(output_dict["types"][2]) == 0
    assert len(output_dict["probs"][0]) == 270
    assert len(output_dict["probs"][1]) == 52
    assert len(output_dict["probs"][2]) == 0


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
