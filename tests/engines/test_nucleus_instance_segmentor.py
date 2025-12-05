"""Test tiatoolbox.models.engine.nucleus_instance_segmentor."""

from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
import torch
import zarr

from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models import NucleusInstanceSegmentor
from tiatoolbox.wsicore import WSIReader

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_functionality_patch_mode(  # noqa: PLR0915
    remote_sample: Callable, track_tmp_path: Path
) -> None:
    """Patch mode functionality test for nuclei instance segmentor."""
    mini_wsi_svs = Path(remote_sample("wsi4_1k_1k_svs"))
    mini_wsi = WSIReader.open(mini_wsi_svs)
    size = (256, 256)
    resolution = 0.25
    units: Final = "mpp"
    patch1 = mini_wsi.read_rect(
        location=(0, 0),
        size=size,
        resolution=resolution,
        units=units,
    )
    patch2 = mini_wsi.read_rect(
        location=(512, 512),
        size=size,
        resolution=resolution,
        units=units,
    )

    # Test dummy input, should result in no output segmentation
    patch3 = np.zeros_like(patch1)

    patches = np.stack(arrays=[patch1, patch2, patch3], axis=0)

    inst_segmentor = NucleusInstanceSegmentor(
        batch_size=1,
        num_workers=0,
        model="hovernet_fast-pannuke",
    )
    output = inst_segmentor.run(
        images=patches,
        patch_mode=True,
        device=device,
        output_type="dict",
    )

    assert np.max(output["predictions"][0][:]) == 41
    assert np.max(output["predictions"][1][:]) == 17
    assert np.max(output["predictions"][2][:]) == 0

    assert len(output["box"][0]) == 41
    assert len(output["box"][1]) == 17
    assert len(output["box"][2]) == 0

    assert len(output["centroid"][0]) == 41
    assert len(output["centroid"][1]) == 17
    assert len(output["centroid"][2]) == 0

    assert len(output["contour"][0]) == 41
    assert len(output["contour"][1]) == 17
    assert len(output["contour"][2]) == 0

    assert len(output["prob"][0]) == 41
    assert len(output["prob"][1]) == 17
    assert len(output["prob"][2]) == 0

    assert len(output["type"][0]) == 41
    assert len(output["type"][1]) == 17
    assert len(output["type"][2]) == 0

    output_ = output

    output = inst_segmentor.run(
        images=patches,
        patch_mode=True,
        device=device,
        output_type="zarr",
        save_dir=track_tmp_path / "patch_output_zarr",
    )

    output = zarr.open(output, mode="r")

    assert np.max(output["predictions"][0][:]) == 41
    assert np.max(output["predictions"][1][:]) == 17

    assert all(
        np.array_equal(a, b)
        for a, b in zip(output["box"][0], output_["box"][0], strict=False)
    )
    assert all(
        np.array_equal(a, b)
        for a, b in zip(output["box"][1], output_["box"][1], strict=False)
    )
    assert len(output["box"][2]) == 0

    assert all(
        np.array_equal(a, b)
        for a, b in zip(output["centroid"][0], output_["centroid"][0], strict=False)
    )
    assert all(
        np.array_equal(a, b)
        for a, b in zip(output["centroid"][1], output_["centroid"][1], strict=False)
    )

    assert all(
        np.array_equal(a, b)
        for a, b in zip(output["contour"][0], output_["contour"][0], strict=False)
    )
    assert all(
        np.array_equal(a, b)
        for a, b in zip(output["contour"][1], output_["contour"][1], strict=False)
    )

    assert all(
        np.array_equal(a, b)
        for a, b in zip(output["prob"][0], output_["prob"][0], strict=False)
    )
    assert all(
        np.array_equal(a, b)
        for a, b in zip(output["prob"][1], output_["prob"][1], strict=False)
    )

    assert all(
        np.array_equal(a, b)
        for a, b in zip(output["type"][0], output_["type"][0], strict=False)
    )
    assert all(
        np.array_equal(a, b)
        for a, b in zip(output["type"][1], output_["type"][1], strict=False)
    )

    inst_segmentor = NucleusInstanceSegmentor(
        batch_size=1,
        num_workers=0,
        model="hovernet_fast-pannuke",
    )
    output = inst_segmentor.run(
        images=patches,
        patch_mode=True,
        device=device,
        output_type="annotationstore",
        save_dir=track_tmp_path / "patch_output_annotationstore",
    )

    assert output[0] == track_tmp_path / "patch_output_annotationstore" / "0.db"
    assert len(output) == 3
    store_ = SQLiteStore.open(output[0])
    annotations_ = store_.values()
    annotations_geometry_type = [
        str(annotation_.geometry_type) for annotation_ in annotations_
    ]
    assert "Polygon" in annotations_geometry_type

    annotations_list = list(annotations_)
    ann_properties = [ann.properties for ann in annotations_list]

    result = {}
    for d in ann_properties:
        for key, value in d.items():
            result.setdefault(key, []).append(value)

    polygons = [ann.geometry for ann in annotations_list]
    result["contour"] = [list(poly.exterior.coords) for poly in polygons]

    assert all(
        np.array_equal(a, b)
        for a, b in zip(result["box"], output_["box"][0], strict=False)
    )

    assert all(
        np.array_equal(a, b)
        for a, b in zip(result["centroid"], output_["centroid"][0], strict=False)
    )

    assert all(
        np.array_equal(a, b)
        for a, b in zip(result["prob"], output_["prob"][0], strict=False)
    )

    assert all(
        np.array_equal(a, b)
        for a, b in zip(result["type"], output_["type"][0], strict=False)
    )

    assert all(
        np.array_equal(
            np.array(a[:-1], dtype=int),  # discard last point
            np.array(b, dtype=int),
        )
        for a, b in zip(result["contour"], output_["contour"][0], strict=False)
    )
