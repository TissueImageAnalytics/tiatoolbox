"""Test tiatoolbox.models.engine.nucleus_instance_segmentor."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import zarr

from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models import NucleusInstanceSegmentor
from tiatoolbox.wsicore import WSIReader

device = "cuda:0" if torch.cuda.is_available() else "cpu"
OutputType = dict[str, Any] | Any


def assert_output_lengths(output: OutputType, expected_counts: Sequence[int]) -> None:
    """Assert lengths of output dict fields against expected counts."""
    for field in ["box", "centroid", "contour", "prob", "type"]:
        for i, expected in enumerate(expected_counts):
            assert len(output[field][i]) == expected, f"{field}[{i}] mismatch"


def assert_output_equal(
    output_a: OutputType,
    output_b: OutputType,
    fields: Sequence[str],
    indices_a: Sequence[int],
    indices_b: Sequence[int],
) -> None:
    """Assert equality of arrays across outputs for given fields/indices."""
    for field in fields:
        for i_a, i_b in zip(indices_a, indices_b, strict=False):
            left = output_a[field][i_a]
            right = output_b[field][i_b]
            assert all(
                np.array_equal(a, b) for a, b in zip(left, right, strict=False)
            ), f"{field}[{i_a}] vs {field}[{i_b}] mismatch"


def assert_predictions_and_boxes(
    output: OutputType, expected_counts: Sequence[int], *, is_zarr: bool = False
) -> None:
    """Assert predictions maxima and box lengths against expected counts."""
    # predictions maxima
    for idx, expected in enumerate(expected_counts):
        if is_zarr and idx == 2:
            # zarr output doesn't store predictions for patch 2
            continue
        assert np.max(output["predictions"][idx][:]) == expected, (
            f"predictions[{idx}] mismatch"
        )

    # box lengths
    for idx, expected in enumerate(expected_counts):
        if is_zarr and idx < 2:
            # for zarr, compare boxes only for patches 0 and 1
            continue
        assert len(output["box"][idx]) == expected, f"box[{idx}] mismatch"


def test_functionality_patch_mode(
    remote_sample: Callable, track_tmp_path: Path
) -> None:
    """Patch mode functionality test for nuclei instance segmentor."""
    mini_wsi_svs = Path(remote_sample("wsi4_1k_1k_svs"))
    mini_wsi = WSIReader.open(mini_wsi_svs)
    size = (256, 256)
    resolution = 0.25
    units: Final = "mpp"

    patch1 = mini_wsi.read_rect(
        location=(0, 0), size=size, resolution=resolution, units=units
    )
    patch2 = mini_wsi.read_rect(
        location=(512, 512), size=size, resolution=resolution, units=units
    )
    patch3 = np.zeros_like(patch1)
    patches = np.stack([patch1, patch2, patch3], axis=0)

    inst_segmentor = NucleusInstanceSegmentor(
        batch_size=1, num_workers=0, model="hovernet_fast-pannuke"
    )
    output_dict = inst_segmentor.run(
        images=patches, patch_mode=True, device=device, output_type="dict"
    )

    expected_counts = [41, 17, 0]

    assert_predictions_and_boxes(output_dict, expected_counts, is_zarr=False)
    assert_output_lengths(output_dict, expected_counts)

    # Zarr output comparison
    output_zarr = inst_segmentor.run(
        images=patches,
        patch_mode=True,
        device=device,
        output_type="zarr",
        save_dir=track_tmp_path / "patch_output_zarr",
    )
    output_zarr = zarr.open(output_zarr, mode="r")
    assert_predictions_and_boxes(output_zarr, expected_counts, is_zarr=True)

    assert_output_equal(
        output_zarr,
        output_dict,
        fields=["box", "centroid", "contour", "prob", "type"],
        indices_a=[0, 1, 2],
        indices_b=[0, 1, 2],
    )

    # AnnotationStore output comparison
    output_ann = inst_segmentor.run(
        images=patches,
        patch_mode=True,
        device=device,
        output_type="annotationstore",
        save_dir=track_tmp_path / "patch_output_annotationstore",
    )
    assert len(output_ann) == 3
    assert output_ann[0] == track_tmp_path / "patch_output_annotationstore" / "0.db"

    for patch_idx, db_path in enumerate(output_ann):
        assert (
            db_path
            == track_tmp_path / "patch_output_annotationstore" / f"{patch_idx}.db"
        )
        store_ = SQLiteStore.open(db_path)
        annotations_ = store_.values()
        annotations_geometry_type = [
            str(annotation_.geometry_type) for annotation_ in annotations_
        ]
        annotations_list = list(annotations_)
        if expected_counts[patch_idx] > 0:
            assert "Polygon" in annotations_geometry_type

            # Build result dict from annotation properties
            result = {}
            for ann in annotations_list:
                for key, value in ann.properties.items():
                    result.setdefault(key, []).append(value)
            result["contour"] = [
                list(poly.exterior.coords)
                for poly in (a.geometry for a in annotations_list)
            ]

            # wrap it to make it compatible to assert_output_lengths
            result_ = {
                field: [result[field]]
                for field in ["box", "centroid", "contour", "prob", "type"]
            }

            # Lengths and equality checks for this patch
            assert_output_lengths(result_, [expected_counts[patch_idx]])
            assert_output_equal(
                result_,
                output_dict,
                fields=["box", "centroid", "prob", "type"],
                indices_a=[0],
                indices_b=[patch_idx],
            )

            # Contour check (discard last point)
            assert all(
                np.array_equal(np.array(a[:-1], dtype=int), np.array(b, dtype=int))
                for a, b in zip(
                    result["contour"], output_dict["contour"][patch_idx], strict=False
                )
            )
        else:
            assert annotations_geometry_type == []
            assert annotations_list == []
