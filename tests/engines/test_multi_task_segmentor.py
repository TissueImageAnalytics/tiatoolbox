"""Test MultiTaskSegmentor."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final

import dask.array as da
import numpy as np
import psutil
import pytest
import torch
import zarr

from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.models.engine.multi_task_segmentor import (
    MultiTaskSegmentor,
    _clear_zarr,
    _save_multitask_vertical_to_cache,
)
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils import imwrite
from tiatoolbox.wsicore import WSIReader

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

OutputType = dict[str, Any] | Any
device = "cuda" if toolbox_env.has_gpu() else "cpu"


def test_mtsegmentor_init() -> None:
    """Tests MultiTaskSegmentor initialization."""
    segmentor = MultiTaskSegmentor(model="hovernetplus-oed", device=device)

    assert isinstance(segmentor, MultiTaskSegmentor)
    assert isinstance(segmentor.model, torch.nn.Module)


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
            inputs=patches,
            output_ann=output_ann_,
            output_dict=output_dict[task_name],
            track_tmp_path=track_tmp_path,
            fields=fields,
            expected_counts=expected_counts,
            task_name=task_name,
            class_dict=mtsegmentor.model.class_dict,
        )


def test_single_task_mtsegmentor(
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

    inputs = [Path(patch1_path), Path(patch2_path), Path(patch3_path)]

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
        images=inputs,
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
        images=inputs,
        patch_mode=True,
        device=device,
        output_type="annotationstore",
        save_dir=track_tmp_path / "patch_output_annotationstore",
        return_probabilities=True,
    )

    assert len(output_ann) == 3

    assert_annotation_store_patch_output(
        inputs=inputs,
        output_ann=output_ann,
        output_dict=output_dict,
        track_tmp_path=track_tmp_path,
        fields=["box", "centroid", "contours", "prob", "type"],
        expected_counts=expected_counts_nuclei,
        task_name=None,
        class_dict=mtsegmentor.model.class_dict["nuclei_segmentation"],
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


def test_wsi_mtsegmentor_zarr(
    remote_sample: Callable,
    track_tmp_path: Path,
) -> None:
    """Test MultiTaskSegmentor for WSIs with zarr output."""
    wsi4_1k_1k_svs = remote_sample("wsi4_1k_1k_svs")
    mtsegmentor = MultiTaskSegmentor(
        model="hovernetplus-oed",
        batch_size=64,
        verbose=False,
        num_workers=1,
    )
    ioconfig = mtsegmentor.ioconfig
    # Return Probabilities is False
    output_full = mtsegmentor.run(
        images=[wsi4_1k_1k_svs],
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=track_tmp_path / "wsi_out_full",
        batch_size=2,
        output_type="zarr",
        ioconfig=ioconfig,
    )

    output_ = zarr.open(output_full[wsi4_1k_1k_svs], mode="r")
    assert 37 < np.mean(output_["nuclei_segmentation"]["predictions"][:]) < 41
    assert 0.87 < np.mean(output_["layer_segmentation"]["predictions"][:]) < 0.91
    assert "probabilities" not in output_
    assert "canvas" not in output_["nuclei_segmentation"]
    assert "count" not in output_["nuclei_segmentation"]
    assert "canvas" not in output_["layer_segmentation"]
    assert "count" not in output_["layer_segmentation"]

    # Redefine tile size to force tile-based processing.
    ioconfig.tile_shape = (512, 512)

    # Return Probabilities is False
    output_tile = mtsegmentor.run(
        images=[wsi4_1k_1k_svs],
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=track_tmp_path / "wsi_out_tile_based",
        batch_size=2,
        output_type="zarr",
        memory_threshold=1,  # Memory threshold forces tile_mode
        ioconfig=ioconfig,
        # HoVerNet does not return predictions once
        # contours have been calculated in original implementation.
        # It's also not straight forward to keep track of instances
        # Prediction masks can be tracked and saved as for layer segmentation in
        # HoVerNet Plus.
        return_predictions=(False, True),
    )

    _ = zarr.open(output_tile[wsi4_1k_1k_svs], mode="r")
    wsi4_1k_1k_svs.unlink()


def test_multi_input_wsi_mtsegmentor_zarr(
    remote_sample: Callable,
    track_tmp_path: Path,
) -> None:
    """Test MultiTaskSegmentor for multiple WSIs with zarr output."""
    wsi4_512_512_svs = Path(remote_sample("wsi4_512_512_svs"))
    wsi4_512_512_svs_2 = wsi4_512_512_svs.parent / (
        wsi4_512_512_svs.stem + "_2" + wsi4_512_512_svs.suffix
    )
    wsi4_512_512_svs_2 = Path(
        shutil.copy(str(wsi4_512_512_svs), str(wsi4_512_512_svs_2))
    )

    # Return Probabilities is True
    # Add multi-input test
    # Use single task output from hovernet
    mtsegmentor = MultiTaskSegmentor(
        model="hovernet_fast-pannuke",
        batch_size=64,
        verbose=False,
        num_workers=1,
    )
    output = mtsegmentor.run(
        images=[wsi4_512_512_svs_2, wsi4_512_512_svs],
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=track_tmp_path / "return_probabilities_check",
        batch_size=2,
        output_type="zarr",
        stride_shape=(160, 160),
        verbose=True,
    )

    output_ = zarr.open(output[wsi4_512_512_svs], mode="r")
    assert 23 < np.mean(output_["predictions"][:]) < 27
    assert "probabilities" in output_
    assert "canvas" not in output_
    assert "count" not in output_

    output_ = zarr.open(output[wsi4_512_512_svs_2], mode="r")
    assert 23 < np.mean(output_["predictions"][:]) < 27
    assert "probabilities" in output_
    assert "canvas" not in output_
    assert "count" not in output_


def test_wsi_segmentor_annotationstore(
    remote_sample: Callable, track_tmp_path: Path
) -> None:
    """Test MultiTaskSegmentor for WSIs with AnnotationStore output."""
    wsi4_512_512_svs = remote_sample("wsi4_512_512_svs")
    mtsegmentor = MultiTaskSegmentor(
        model="hovernet_fast-pannuke",
        batch_size=32,
        verbose=False,
    )

    class_dict = mtsegmentor.model.class_dict

    # Return Probabilities is False
    output = mtsegmentor.run(
        images=[wsi4_512_512_svs],
        return_probabilities=False,
        device=device,
        patch_mode=False,
        save_dir=track_tmp_path / "wsi_out_check",
        verbose=True,
        output_type="annotationstore",
        class_dict=class_dict,
        memory_threshold=0,
    )

    for output_ in output[wsi4_512_512_svs]:
        assert output_.suffix != ".zarr"

    store_file_name = f"{wsi4_512_512_svs.stem}.db"
    store_file_path = track_tmp_path / "wsi_out_check" / store_file_name
    assert store_file_path.exists()
    assert store_file_path == output[wsi4_512_512_svs][0]


def test_wsi_segmentor_annotationstore_probabilities(
    remote_sample: Callable, track_tmp_path: Path, caplog: pytest.CaptureFixture
) -> None:
    """Test MultiTaskSegmentor with AnnotationStore and probabilities output."""
    wsi4_512_512_svs = remote_sample("wsi4_512_512_svs")
    # Return Probabilities is True
    mtsegmentor = MultiTaskSegmentor(
        model="hovernetplus-oed",
        batch_size=32,
        verbose=False,
    )

    output = mtsegmentor.run(
        images=[wsi4_512_512_svs],
        return_probabilities=True,
        device=device,
        patch_mode=False,
        save_dir=track_tmp_path / "wsi_prob_out_check",
        verbose=True,
        output_type="annotationstore",
    )

    assert "Probability maps cannot be saved as AnnotationStore." in caplog.text
    zarr_group = zarr.open(output[wsi4_512_512_svs][0], mode="r")
    assert "probabilities" in zarr_group

    for task_name in mtsegmentor.tasks:
        store_file_name = f"{wsi4_512_512_svs.stem}_{task_name}.db"
        store_file_path = track_tmp_path / "wsi_prob_out_check" / store_file_name
        assert store_file_path.exists()
        assert store_file_path in output[wsi4_512_512_svs]
        assert task_name not in zarr_group


def test_raise_value_error_return_labels_wsi(
    remote_sample: Callable,
    track_tmp_path: Path,
) -> None:
    """Tests MultiTaskSegmentor return_labels error."""
    wsi4_512_512_svs = remote_sample("wsi4_512_512_svs")
    mtsegmentor = MultiTaskSegmentor(model="hovernetplus-oed", device=device)

    with pytest.raises(
        ValueError,
        match=r".*return_labels` is not supported for MultiTaskSegmentor.",
    ):
        _ = mtsegmentor.run(
            images=[wsi4_512_512_svs],
            return_probabilities=False,
            return_labels=True,
            device=device,
            patch_mode=False,
            save_dir=track_tmp_path / "wsi_out_check",
            batch_size=2,
            output_type="zarr",
        )


def test_clear_zarr() -> None:
    """Test _clear_zarr working appropriately.

    This test only covers scenarios which are not feasible to run on GitHub Actions.

    """
    store = zarr.MemoryStore()
    root = zarr.group(store=store)

    # Create a dummy zarr array for probabilities_zarr
    probabilities_zarr = root.create_dataset("probabilities", data=np.zeros((5, 3, 3)))

    idx = 2
    chunk_shape = (1,)
    probabilities_shape = (3, 3)

    result = _clear_zarr(
        probabilities_zarr=probabilities_zarr,
        probabilities_da=None,
        zarr_group=root,
        idx=idx,
        chunk_shape=chunk_shape,
        probabilities_shape=probabilities_shape,
    )

    # Ensure the keys still exist but the specific index was removed
    assert "canvas" not in root
    assert "count" not in root
    assert isinstance(result, da.Array)

    result_ = _clear_zarr(
        probabilities_zarr=None,
        probabilities_da=result,
        zarr_group=root,
        idx=idx,
        chunk_shape=chunk_shape,
        probabilities_shape=probabilities_shape,
    )

    assert np.all(result_.compute() == result.compute())


def test_vertical_save_branch_without_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test saving to cache if memory threshold is breached for vertical merge."""
    idx = 0

    class FakeVM:
        """Fake psutil.virtual_memory() with extremely low free memory."""

        free = 1  # force used_percent > memory_threshold

    monkeypatch.setattr(psutil, "virtual_memory", FakeVM)

    # --- Real dask array ---
    da_arr = da.from_array(np.array([[1, 2, 3]]), chunks=(1, 3))
    probabilities_da = [da_arr]

    # --- probabilities_zarr slot is None to trigger the branch ---
    probabilities_zarr = [None]

    # --- Real numpy array for shape/dtype ---
    probabilities = np.zeros((1, 3))

    class DummyTqdm:
        """Dummy tqdm with a write() method."""

        messages: ClassVar[list[str]] = []

        @classmethod
        def write(cls: DummyTqdm, msg: str) -> None:
            """Append a message to the messages list."""
            cls.messages.append(msg)

    # --- Call function ---
    new_zarr, new_da = _save_multitask_vertical_to_cache(
        probabilities_zarr=probabilities_zarr,
        probabilities_da=probabilities_da,
        probabilities=probabilities,
        idx=idx,
        tqdm_=DummyTqdm,
        save_path=tmp_path / "cache.zarr",
        chunk_shape=(1,),
        memory_threshold=0,  # ensure branch triggers
    )

    # --- Assertions ---
    # tqdm.write was called
    assert len(DummyTqdm.messages) == 1
    assert "Saving intermediate results to disk" in DummyTqdm.messages[0]

    # probabilities_da must be set to None
    assert new_da[idx] is None

    # new_zarr must be a real zarr array
    assert isinstance(new_zarr[idx], zarr.Array)

    # Data was written correctly
    assert np.array_equal(new_zarr[idx][:], np.array([[1, 2, 3]]))


# HELPER functions
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
    inputs: list | np.ndarray,
    output_ann: list[Path],
    task_name: str | None,
    track_tmp_path: Path,
    expected_counts: Sequence[int],
    output_dict: OutputType,
    fields: list[str],
    class_dict: dict,
) -> None:
    """Helper function to test AnnotationStore output."""
    for patch_idx, db_path in enumerate(output_ann):
        if isinstance(inputs[patch_idx], Path):
            store_file_name = (
                f"{inputs[patch_idx].stem}.db"
                if task_name is None
                else f"{inputs[patch_idx].stem}_{task_name}.db"
            )
        else:
            store_file_name = (
                f"{patch_idx}.db"
                if task_name is None
                else f"{patch_idx}_{task_name}.db"
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

            class_dict_ = class_dict[task_name] if task_name else class_dict
            type_ = [class_dict_[c_id] for c_id in output_dict["type"][patch_idx]]
            output_dict["type"][patch_idx] = type_
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
