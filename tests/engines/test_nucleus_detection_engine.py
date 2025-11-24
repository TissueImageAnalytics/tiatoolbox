"""Tests for NucleusDetector."""

import pathlib
import shutil
from collections.abc import Callable

import pandas as pd
import pytest

from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils.misc import imwrite
import numpy as np
import dask.array as da

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def _rm_dir(path: pathlib.Path) -> None:
    """Helper func to remove directory."""
    if pathlib.Path(path).exists():
        shutil.rmtree(path, ignore_errors=True)


def check_output(path: pathlib.Path) -> None:
    """Check NucleusDetector output."""


def test_nucleus_detection_nms_empty_dataframe() -> None:
    """nucleus_detection_nms should return a copy for empty inputs."""
    df = pd.DataFrame(columns=["x", "y", "type", "prob"])

    result = NucleusDetector.nucleus_detection_nms(df, radius=3)

    assert result.empty
    assert result is not df
    assert list(result.columns) == ["x", "y", "type", "prob"]


def test_nucleus_detection_nms_invalid_radius() -> None:
    """Radius must be strictly positive."""
    df = pd.DataFrame({"x": [0], "y": [0], "type": [1], "prob": [0.9]})

    with pytest.raises(ValueError, match="radius must be > 0"):
        NucleusDetector.nucleus_detection_nms(df, radius=0)


def test_nucleus_detection_nms_invalid_overlap_threshold() -> None:
    """overlap_threshold must lie in (0, 1]."""
    df = pd.DataFrame({"x": [0], "y": [0], "type": [1], "prob": [0.9]})

    message = r"overlap_threshold must be in \(0\.0, 1\.0\], got 0"
    with pytest.raises(ValueError, match=message):
        NucleusDetector.nucleus_detection_nms(df, radius=1, overlap_threshold=0)


def test_nucleus_detection_nms_suppresses_overlapping_detections() -> None:
    """Lower-probability overlapping detections are removed."""
    df = pd.DataFrame(
        {
            "x": [2, 0, 20],
            "y": [1, 0, 20],
            "type": [1, 1, 2],
            "prob": [0.6, 0.9, 0.7],
        }
    )

    result = NucleusDetector.nucleus_detection_nms(df, radius=5)

    expected = pd.DataFrame(
        {"x": [0, 20], "y": [0, 20], "type": [1, 2], "prob": [0.9, 0.7]}
    )
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_nucleus_detection_nms_suppresses_across_types() -> None:
    """Overlapping detections of different types are also suppressed."""
    df = pd.DataFrame(
        {
            "x": [0, 0, 20],
            "y": [0, 0, 0],
            "type": [1, 2, 1],
            "prob": [0.6, 0.95, 0.4],
        }
    )

    result = NucleusDetector.nucleus_detection_nms(df, radius=5)

    expected = pd.DataFrame(
        {"x": [0, 20], "y": [0, 0], "type": [2, 1], "prob": [0.95, 0.4]}
    )
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_nucleus_detection_nms_retains_non_overlapping_candidates() -> None:
    """Detections with IoU below the threshold are preserved."""
    df = pd.DataFrame(
        {
            "x": [0, 10],
            "y": [0, 0],
            "type": [1, 1],
            "prob": [0.8, 0.5],
        }
    )

    result = NucleusDetector.nucleus_detection_nms(df, radius=5, overlap_threshold=0.5)

    expected = pd.DataFrame(
        {"x": [0, 10], "y": [0, 0], "type": [1, 1], "prob": [0.8, 0.5]}
    )
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


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

    _rm_dir(save_dir)


def test_nucleus_detector_patch(
    remote_sample: Callable, tmp_path: pathlib.Path
) -> None:
    """Test for nucleus detection engine in patch mode."""
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))

    wsi_reader = WSIReader.open(mini_wsi_svs)
    patch_1 = wsi_reader.read_rect((0, 0), (252, 252), resolution=0.5, units="mpp")
    patch_2 = wsi_reader.read_rect((252, 252), (252, 252), resolution=0.5, units="mpp")

    pretrained_model = "mapde-conic"

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
    )

    store_1 = SQLiteStore.open(save_dir / "0.db")
    assert len(store_1.values()) == 270
    store_1.close()

    store_2 = SQLiteStore.open(save_dir / "1.db")
    assert len(store_2.values()) == 52
    store_2.close()

    imwrite(save_dir / "patch_0.png", patch_1)
    imwrite(save_dir / "patch_1.png", patch_2)
    _ = nucleus_detector.run(
        patch_mode=True,
        device=device,
        output_type="zarr",
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


def test_nucleus_detector_write_centroid_maps(tmp_path: pathlib.Path)->None:
    """Test for _write_centroid_maps function."""

    detection_maps = np.zeros((20, 20, 1), dtype=np.uint8)
    detection_maps = da.from_array(detection_maps, chunks=(20, 20, 1))

    store = NucleusDetector.write_centroid_maps_to_store(
        detection_maps=detection_maps,
    )
    assert len(store.values()) == 0
    store.close()

    detection_maps = np.zeros((20, 20, 1), dtype=np.uint8)
    detection_maps[10, 10, 0] = 1
    detection_maps = da.from_array(detection_maps, chunks=(20, 20, 1))
    _ = NucleusDetector.write_centroid_maps_to_store(
        detection_maps=detection_maps,
        save_path=tmp_path / "test.db",
        class_dict={0: "nucleus"},
    )
    store = SQLiteStore.open(tmp_path / "test.db")
    assert len(store.values()) == 1
    annotation = next(iter(store.values()))
    print(annotation)
    assert annotation.properties["type"] == "nucleus"
    assert annotation.geometry.centroid.x == 10.0
    assert annotation.geometry.centroid.y == 10.0
    store.close()