"""Test MultiTaskSegmentor."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pytest
import torch
import zarr

from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils import imwrite
from tiatoolbox.wsicore import WSIReader

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

OutputType = dict[str, Any] | Any
device = "cuda" if toolbox_env.has_gpu() else "cpu"


def assert_output_lengths(
    output: OutputType, expected_counts: Sequence[int], fields: list[str]
) -> None:
    """Assert lengths of output dict fields against expected counts."""
    for field in fields:
        for i, expected in enumerate(expected_counts):
            assert len(output[field][i]) == expected, f"{field}[{i}] mismatch"


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


def assert_annotation_store_patch_output(
    output_ann: list[Path],
    task_name: str | None,
    track_tmp_path: Path,
    expected_counts: Sequence[int],
    output_dict: OutputType,
    fields: list[str],
) -> None:
    """Helper function to test AnnotationStore output."""
    for patch_idx, db_path in enumerate(output_ann):
        store_file_name = (
            f"{patch_idx}.db" if task_name is None else f"{patch_idx}_{task_name}.db"
        )
        assert (
            db_path == track_tmp_path / "patch_output_annotationstore" / store_file_name
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
            result["contours"] = [
                list(poly.exterior.coords)
                for poly in (a.geometry for a in annotations_list)
            ]

            # wrap it to make it compatible to assert_output_lengths
            result_ = {field: [result[field]] for field in fields}

            # Lengths and equality checks for this patch
            assert_output_lengths(
                result_,
                expected_counts=[expected_counts[patch_idx]],
                fields=fields,
            )
            fields_ = fields.copy()
            fields_.remove("contours")
            assert_output_equal(
                result_,
                output_dict,
                fields=fields_,
                indices_a=[0],
                indices_b=[patch_idx],
            )

            # Contour check (discard last point)
            matches = [
                np.array_equal(np.array(a[:-1], dtype=int), np.array(b, dtype=int))
                for a, b in zip(
                    result["contours"], output_dict["contours"][patch_idx], strict=False
                )
            ]
            # Due to make valid poly there might be translation in a few points
            # in AnnotationStore
            assert sum(matches) / len(matches) >= 0.95
        else:
            assert annotations_geometry_type == []
            assert annotations_list == []


def test_mtsegmentor_init() -> None:
    """Tests MultiTaskSegmentor initialization."""
    segmentor = MultiTaskSegmentor(model="hovernetplus-oed", device=device)

    assert isinstance(segmentor, MultiTaskSegmentor)
    assert isinstance(segmentor.model, torch.nn.Module)


def test_raise_value_error_return_labels_wsi(
    sample_svs: Path,
    track_tmp_path: Path,
) -> None:
    """Tests MultiTaskSegmentor return_labels error."""
    mtsegmentor = MultiTaskSegmentor(model="hovernetplus-oed", device=device)

    with pytest.raises(
        ValueError,
        match=r".*return_labels` is not supported for MultiTaskSegmentor.",
    ):
        _ = mtsegmentor.run(
            images=[sample_svs],
            return_probabilities=False,
            return_labels=True,
            device=device,
            patch_mode=False,
            save_dir=track_tmp_path / "wsi_out_check",
            batch_size=2,
            output_type="zarr",
        )


def test_mtsegmentor_patches(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Tests MultiTaskSegmentor on image patches."""
    mtsegmentor = MultiTaskSegmentor(
        model="hovernetplus-oed", batch_size=32, verbose=False, device=device
    )

    mini_wsi_svs = Path(remote_sample("wsi4_1k_1k_svs"))
    mini_wsi = WSIReader.open(mini_wsi_svs)
    size = (256, 256)
    resolution = 0.50
    units: Final = "mpp"

    patch1 = mini_wsi.read_rect(
        location=(0, 0), size=size, resolution=resolution, units=units
    )
    patch2 = mini_wsi.read_rect(
        location=(512, 512), size=size, resolution=resolution, units=units
    )
    patch3 = np.zeros_like(patch1)
    patches = np.stack([patch1, patch2, patch3], axis=0)

    assert not mtsegmentor.patch_mode

    output_dict = mtsegmentor.run(
        images=patches,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
    )

    expected_counts_nuclei = [95, 33, 0]
    assert_output_lengths(
        output_dict["nuclei_segmentation"],
        expected_counts_nuclei,
        fields=["box", "centroid", "contours", "prob", "type"],
    )
    assert_predictions_and_boxes(
        output_dict["nuclei_segmentation"], expected_counts_nuclei, is_zarr=False
    )
    expected_counts_layer = [1, 1, 0]
    assert_output_lengths(
        output_dict["layer_segmentation"],
        expected_counts_layer,
        fields=["contours", "type"],
    )
    assert_predictions_and_boxes(
        output_dict["layer_segmentation"], expected_counts_layer, is_zarr=False
    )

    # Zarr output comparison
    output_zarr = mtsegmentor.run(
        images=patches,
        patch_mode=True,
        device=device,
        output_type="zarr",
        save_dir=track_tmp_path / "patch_output_zarr",
    )
    output_zarr = zarr.open(output_zarr, mode="r")

    assert_output_lengths(
        output_zarr["nuclei_segmentation"],
        expected_counts_nuclei,
        fields=["box", "centroid", "contours", "prob", "type"],
    )
    assert_output_lengths(
        output_zarr["layer_segmentation"],
        expected_counts_layer,
        fields=["contours", "type"],
    )

    assert_output_equal(
        output_zarr["nuclei_segmentation"],
        output_dict["nuclei_segmentation"],
        fields=["box", "centroid", "contours", "prob", "type"],
        indices_a=[0, 1, 2],
        indices_b=[0, 1, 2],
    )
    assert_output_equal(
        output_zarr["layer_segmentation"],
        output_dict["layer_segmentation"],
        fields=["contours", "type"],
        indices_a=[0, 1, 2],
        indices_b=[0, 1, 2],
    )

    # AnnotationStore output comparison
    output_ann = mtsegmentor.run(
        images=patches,
        patch_mode=True,
        device=device,
        output_type="annotationstore",
        save_dir=track_tmp_path / "patch_output_annotationstore",
    )

    assert len(output_ann) == 6

    for task_name in mtsegmentor.tasks:
        fields_nuclei = ["box", "centroid", "contours", "prob", "type"]
        fields_layer = ["contours", "type"]
        fields = fields_nuclei if task_name == "nuclei_segmentation" else fields_layer
        output_ann_ = [p for p in output_ann if p.name.endswith(f"{task_name}.db")]
        expected_counts = (
            expected_counts_nuclei
            if task_name == "nuclei_segmentation"
            else expected_counts_layer
        )
        assert_annotation_store_patch_output(
            output_ann=output_ann_,
            output_dict=output_dict[task_name],
            track_tmp_path=track_tmp_path,
            fields=fields,
            expected_counts=expected_counts,
            task_name=task_name,
        )


def test_single_output_mtsegmentor(
    remote_sample: Callable,
    track_tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Tests MultiTaskSegmentor on single task output."""
    mtsegmentor = MultiTaskSegmentor(
        model="hovernet_fast-pannuke", batch_size=32, verbose=False, device=device
    )
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

    patch1_path = track_tmp_path / "patch1.png"
    patch2_path = track_tmp_path / "patch2.png"
    patch3_path = track_tmp_path / "patch3.png"

    imwrite(patch1_path, patch1)
    imwrite(patch2_path, patch2)
    imwrite(patch3_path, patch3)

    inputs = [Path(patch1_path), str(patch2_path), str(patch3_path)]
    patches = np.stack([patch1, patch2, patch3], axis=0)

    assert not mtsegmentor.patch_mode

    output_dict = mtsegmentor.run(
        images=inputs,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
    )

    expected_counts_nuclei = [41, 17, 0]
    assert_output_lengths(
        output_dict,
        expected_counts_nuclei,
        fields=["box", "centroid", "contours", "prob", "type"],
    )
    assert_predictions_and_boxes(output_dict, expected_counts_nuclei, is_zarr=False)

    # Zarr output comparison
    output_zarr = mtsegmentor.run(
        images=patches,
        patch_mode=True,
        device=device,
        output_type="zarr",
        save_dir=track_tmp_path / "patch_output_zarr",
    )
    output_zarr = zarr.open(output_zarr, mode="r")

    assert_output_lengths(
        output_zarr,
        expected_counts_nuclei,
        fields=["box", "centroid", "contours", "prob", "type"],
    )

    assert_output_equal(
        output_zarr,
        output_dict,
        fields=["box", "centroid", "contours", "prob", "type"],
        indices_a=[0, 1, 2],
        indices_b=[0, 1, 2],
    )

    # AnnotationStore output comparison

    # Reinitialize to check for probabilities in output.
    mtsegmentor.drop_keys = []
    output_ann = mtsegmentor.run(
        images=patches,
        patch_mode=True,
        device=device,
        output_type="annotationstore",
        save_dir=track_tmp_path / "patch_output_annotationstore",
        return_probabilities=True,
    )

    assert len(output_ann) == 3

    assert_annotation_store_patch_output(
        output_ann=output_ann,
        output_dict=output_dict,
        track_tmp_path=track_tmp_path,
        fields=["box", "centroid", "contours", "prob", "type"],
        expected_counts=expected_counts_nuclei,
        task_name=None,
    )

    zarr_file = track_tmp_path / "patch_output_annotationstore" / "output.zarr"

    assert zarr_file.exists()

    zarr_group = zarr.open(
        str(zarr_file),
        mode="r",
    )

    assert "probabilities" in zarr_group

    fields = ["box", "centroid", "contours", "prob", "type", "predictions"]
    for field in fields:
        assert field not in zarr_group

    assert "Probability maps cannot be saved as AnnotationStore" in caplog.text
