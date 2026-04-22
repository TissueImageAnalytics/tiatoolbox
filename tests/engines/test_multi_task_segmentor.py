"""Test MultiTaskSegmentor."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Final
from unittest.mock import MagicMock

import dask.array as da
import numpy as np
import psutil
import pytest
import torch
import zarr
from click.testing import CliRunner
from shapely import Point, STRtree
from tqdm.auto import tqdm
from zarr.storage import LocalStore

from tiatoolbox import cli
from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.models import IOSegmentorConfig
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.engine.multi_task_segmentor import (
    DaskDelayedJSONStore,
    MultiTaskSegmentor,
    _clear_zarr,
    _get_sel_indices_margin_lines,
    _post_save_json_store,
    _process_instance_predictions,
    _save_multitask_vertical_to_cache,
    merge_multitask_vertical_chunkwise,
)
from tiatoolbox.utils import download_data, imwrite
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.wsicore import WSIReader

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

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
        return_predictions=(True, True),
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
    processed_predictions = convert_to_dask(output_dict)
    output_zarr = mtsegmentor.save_predictions(
        processed_predictions=processed_predictions.copy(),
        output_type="zarr",
        save_path=track_tmp_path / "patch_output_zarr" / "output.zarr",
        return_probabilities=False,
        return_predictions=(True, True),
    )

    output_zarr_ = zarr.open(output_zarr, mode="r")

    assert_output_lengths(
        output_zarr_["nuclei_segmentation"],
        expected_counts_nuclei,
        fields=["box", "centroid", "contours", "prob", "type"],
    )
    assert_output_lengths(
        output_zarr_["layer_segmentation"],
        expected_counts_layer,
        fields=["contours", "type"],
    )

    assert_output_equal(
        output_zarr_["nuclei_segmentation"],
        output_dict["nuclei_segmentation"],
        fields=["box", "centroid", "contours", "prob", "type"],
        indices_a=[0, 1, 2],
        indices_b=[0, 1, 2],
    )
    assert_output_equal(
        output_zarr_["layer_segmentation"],
        output_dict["layer_segmentation"],
        fields=["contours", "type"],
        indices_a=[0, 1, 2],
        indices_b=[0, 1, 2],
    )

    # AnnotationStore output comparison
    output_ann = mtsegmentor.save_predictions(
        processed_predictions=processed_predictions.copy(),
        output_type="annotationstore",
        save_path=track_tmp_path
        / "patch_output_annotationstore"
        / (output_zarr.stem + "_ann.db"),
        return_probabilities=False,
        return_predictions=(True, True),
    )

    assert len(output_ann) == 6

    fields_nuclei = ["box", "centroid", "contours", "prob", "type"]
    fields_layer = ["contours", "type"]

    for task_name in mtsegmentor.tasks:
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
            class_dict=mtsegmentor._get_model_attr("class_dict"),
        )

    # QuPath JSON does not have fields
    fields_nuclei = ["contours", "prob", "type"]
    # QuPath output comparison
    output_json = mtsegmentor.save_predictions(
        processed_predictions=processed_predictions.copy(),
        output_type="qupath",
        save_path=track_tmp_path
        / "patch_output_qupath"
        / (output_zarr.stem + "_qupath.db"),
        return_probabilities=False,
        return_predictions=(True, True),
    )

    assert len(output_json) == 6

    for task_name in mtsegmentor.tasks:
        fields = fields_nuclei if task_name == "nuclei_segmentation" else fields_layer
        output_json_ = [p for p in output_json if p.name.endswith(f"{task_name}.json")]
        expected_counts = (
            expected_counts_nuclei
            if task_name == "nuclei_segmentation"
            else expected_counts_layer
        )
        assert_qupath_json_patch_output(
            inputs=patches,
            output_json=output_json_,
            output_dict=output_dict[task_name],
            track_tmp_path=track_tmp_path,
            fields=fields,
            expected_counts=expected_counts,
            task_name=task_name,
        )


def test_mtsegmentor_tiles_no_metadata(track_tmp_path: Path) -> None:
    """Tests MultiTaskSegmentor on a tile with no metadata."""
    img_file_name = track_tmp_path / "tcga_hnscc.png"
    download_data(
        "https://huggingface.co/datasets/TIACentre/TIAToolBox_Remote_Samples/resolve/main/sample_imgs/tcga_hnscc.png",
        img_file_name,
    )
    # Tile prediction
    multi_segmentor = MultiTaskSegmentor(
        model="hovernetplus-oed",
        num_workers=0,
        batch_size=4,
    )

    tile_output = multi_segmentor.run(
        [img_file_name],
        save_dir=track_tmp_path / "sample_tile_results",
        patch_mode=False,
        device=device,
        auto_get_mask=False,
        wsireader_kwargs={"mpp": 0.25},  # use this mpp to run test faster
        return_predictions=(True, True),
    )

    assert tile_output[img_file_name].exists()
    output_zarr = zarr.open(tile_output[img_file_name], mode="r")
    assert "nuclei_segmentation" in output_zarr
    assert "layer_segmentation" in output_zarr
    assert "predictions" in output_zarr["layer_segmentation"]
    assert "predictions" in output_zarr["nuclei_segmentation"]
    fields_layer = ["contours", "type"]
    assert (field in output_zarr["layer_segmentation"] for field in fields_layer)
    fields_nuclei = ["box", "centroid", "contours", "prob", "type"]
    assert (field in output_zarr["nuclei_segmentation"] for field in fields_nuclei)
    assert len(output_zarr["layer_segmentation"]["contours"][:]) == 12
    assert len(output_zarr["nuclei_segmentation"]["contours"][:]) == 1299


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

    imwrite(track_tmp_path / "patch1.png", patch1)
    imwrite(track_tmp_path / "patch2.png", patch2)
    imwrite(track_tmp_path / "patch3.png", patch3)

    inputs = [
        track_tmp_path / "patch1.png",
        track_tmp_path / "patch2.png",
        track_tmp_path / "patch3.png",
    ]

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

    assert next(iter(mtsegmentor.tasks)) == "nuclei_segmentation"
    assert len(mtsegmentor.tasks) == 1

    # Zarr output comparison
    processed_predictions = convert_to_dask_single_task(
        output_dict=output_dict,
        task_name="nuclei_segmentation",
    )

    _ = zarr.open(str(track_tmp_path / "patch_output_zarr" / "output.zarr"), mode="w")

    output_zarr = zarr.open(
        mtsegmentor.save_predictions(
            processed_predictions=processed_predictions.copy(),
            output_type="zarr",
            save_path=track_tmp_path / "patch_output_zarr" / "output.zarr",
            return_probabilities=False,
            return_predictions=(True, True),
        ),
        mode="r",
    )

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
    mtsegmentor.drop_keys = []

    # Triggers Return Coordinates for patch inference
    output_ann = mtsegmentor.run(
        images=inputs,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
        save_dir=track_tmp_path / "patch_output_annotationstore",
        return_predictions=(True,),
        output_type="annotationstore",
    )

    assert len(output_ann) == 3

    class_dict_ = mtsegmentor._get_model_attr("class_dict")
    assert_annotation_store_patch_output(
        inputs=inputs,
        output_ann=output_ann,
        output_dict=output_dict,
        track_tmp_path=track_tmp_path,
        fields=["box", "centroid", "contours", "prob", "type"],
        expected_counts=expected_counts_nuclei,
        task_name=None,
        class_dict=class_dict_["nuclei_segmentation"],
    )

    assert (track_tmp_path / "patch_output_annotationstore" / "output.zarr").exists()

    zarr_group = zarr.open(
        str(track_tmp_path / "patch_output_annotationstore" / "output.zarr"),
        mode="r",
    )

    assert "probabilities" in zarr_group
    assert "predictions" in zarr_group

    fields = ["box", "centroid", "contours", "prob", "type"]
    for field in fields:
        assert field not in zarr_group

    assert "Probability maps cannot be saved as AnnotationStore or JSON" in caplog.text

    # QuPath output comparison
    mtsegmentor.drop_keys = []
    output_json = mtsegmentor.run(
        images=inputs,
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=True,
        save_dir=track_tmp_path / "patch_output_qupath",
        return_predictions=(False,),
        output_type="qupath",
    )

    assert len(output_json) == 3

    assert_qupath_json_patch_output(
        inputs=inputs,
        output_json=output_json,
        output_dict=output_dict,
        track_tmp_path=track_tmp_path,
        fields=["box", "centroid", "contours", "prob", "type"],
        expected_counts=expected_counts_nuclei,
        task_name=None,
    )

    assert (track_tmp_path / "patch_output_qupath" / "output.zarr").exists()

    zarr_group = zarr.open(
        str(track_tmp_path / "patch_output_qupath" / "output.zarr"),
        mode="r",
    )

    assert "probabilities" in zarr_group

    fields = ["box", "centroid", "contours", "prob", "type", "predictions"]
    for field in fields:
        assert field not in zarr_group

    assert "Probability maps cannot be saved as AnnotationStore or JSON" in caplog.text


def test_wsi_mtsegmentor_correct_nonsquare_shape(
    remote_sample: Callable,
    track_tmp_path: Path,
) -> None:
    """Test MultiTaskSegmentor output shape for non-square WSIs with zarr output."""
    # Using this image to check for non-square output.
    svs_1_small = remote_sample("svs-1-small")

    # masking the image for shorter runtime
    svs_1_small = WSIReader.open(svs_1_small)
    mask = np.zeros(
        svs_1_small.slide_dimensions(resolution=1.25, units="power")[::-1],
        dtype=np.uint8,
    )
    mask[150:160, 50:75] = 1

    mtsegmentor = MultiTaskSegmentor(
        model="hovernetplus-oed",
        batch_size=64,
        verbose=False,
        num_workers=1,
    )
    ioconfig = mtsegmentor.ioconfig
    # Return Probabilities is True
    output_full = mtsegmentor.run(
        # Use rectangular (not square) to test output shape
        images=[svs_1_small],
        masks=[mask],
        return_probabilities=True,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=track_tmp_path / "wsi_out_full",
        batch_size=2,
        output_type="zarr",
        ioconfig=ioconfig,
        return_predictions=(True, True),  # True for both tasks.
    )

    output_full_ = zarr.open(output_full[svs_1_small.input_path], mode="r")
    assert 0.24 < np.mean(output_full_["nuclei_segmentation"]["predictions"][:]) < 0.26
    assert 0.03 < np.mean(output_full_["layer_segmentation"]["predictions"][:]) < 0.04
    assert "probabilities" in output_full_
    assert "canvas" not in output_full_["nuclei_segmentation"]
    assert "count" not in output_full_["nuclei_segmentation"]
    assert "canvas" not in output_full_["layer_segmentation"]
    assert "count" not in output_full_["layer_segmentation"]

    # Verify output shape
    expected_shape = svs_1_small.slide_dimensions(
        **mtsegmentor.ioconfig.highest_input_resolution
    )[::-1]
    assert np.all(
        output_full_["nuclei_segmentation"]["predictions"][:].shape == expected_shape
    )
    assert np.all(
        output_full_["layer_segmentation"]["predictions"][:].shape == expected_shape
    )

    # Redefine tile size to force tile-based processing.
    # 350 x 350 forces tile mode 3 (overlap)
    ioconfig.tile_shape = (500, 500)
    mtsegmentor.drop_keys = []

    # Return Probabilities is False
    output_tile = mtsegmentor.run(
        # Use rectangular (not square) to test output shape
        images=[svs_1_small],
        masks=[mask],
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=track_tmp_path / "wsi_out_tile",
        batch_size=2,
        output_type="zarr",
        ioconfig=ioconfig,
        return_predictions=(True, True),  # True for both tasks.
    )

    output_tile_ = zarr.open(output_tile[svs_1_small.input_path], mode="r")
    assert 0.23 < np.mean(output_tile_["nuclei_segmentation"]["predictions"][:]) < 0.25
    assert 0.03 < np.mean(output_tile_["layer_segmentation"]["predictions"][:]) < 0.04
    assert "probabilities" not in output_tile_
    assert "canvas" not in output_tile_["nuclei_segmentation"]
    assert "count" not in output_tile_["nuclei_segmentation"]
    assert "canvas" not in output_tile_["layer_segmentation"]
    assert "count" not in output_tile_["layer_segmentation"]

    # Verify output shape
    expected_shape = svs_1_small.slide_dimensions(
        **mtsegmentor.ioconfig.highest_input_resolution
    )[::-1]
    assert np.all(
        output_tile_["nuclei_segmentation"]["predictions"][:].shape == expected_shape
    )
    assert np.all(
        output_tile_["layer_segmentation"]["predictions"][:].shape == expected_shape
    )


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

    # Force calculation without tile-based processing.
    ioconfig.tile_shape = (1200, 1200)
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
        return_predictions=(True, True),  # True for both tasks.
    )

    output_full_ = zarr.open(output_full[wsi4_1k_1k_svs], mode="r")
    assert 64 < np.mean(output_full_["nuclei_segmentation"]["predictions"][:]) < 68
    assert 0.88 < np.mean(output_full_["layer_segmentation"]["predictions"][:]) < 0.92
    assert "probabilities" not in output_full_
    assert "canvas" not in output_full_["nuclei_segmentation"]
    assert "count" not in output_full_["nuclei_segmentation"]
    assert "canvas" not in output_full_["layer_segmentation"]
    assert "count" not in output_full_["layer_segmentation"]
    assert np.all(
        output_full_["nuclei_segmentation"]["predictions"][:].shape == (504, 504)
    )
    assert np.all(
        output_full_["layer_segmentation"]["predictions"][:].shape == (504, 504)
    )

    # Redefine tile size to force tile-based processing.
    # 350 x 350 forces tile mode 3 (overlap)
    ioconfig.tile_shape = (350, 350)
    mtsegmentor.drop_keys = []

    # Return predictions is False
    output_tile = mtsegmentor.run(
        images=[wsi4_1k_1k_svs],
        return_probabilities=False,
        return_labels=False,
        device=device,
        patch_mode=False,
        save_dir=track_tmp_path / "wsi_out_tile_based",
        batch_size=2,
        output_type="zarr",
        memory_threshold=0,  # Memory threshold forces tile_mode
        ioconfig=ioconfig,
        # HoVerNet does not return predictions once
        # contours have been calculated in original implementation.
        # It's also not straight forward to keep track of instances
        # Prediction masks can be tracked and saved as for layer segmentation in
        # HoVerNet Plus.
        return_predictions=(False, True),
        verbose=False,
    )

    output_tile_ = zarr.open(output_tile[wsi4_1k_1k_svs], mode="r")
    assert "predictions" not in output_tile_["nuclei_segmentation"]
    assert 0.87 < np.mean(output_tile_["layer_segmentation"]["predictions"][:]) < 0.91
    predictions_tile = output_tile_["layer_segmentation"]["predictions"][:]
    predictions_full = output_full_["layer_segmentation"]["predictions"][:]
    overlap_pct = np.mean(predictions_full == predictions_tile) * 100
    assert overlap_pct > 99
    assert len(output_full_["layer_segmentation"]["contours"][:]) == len(
        output_tile_["layer_segmentation"]["contours"][:]
    )
    assert (
        len(output_tile_["nuclei_segmentation"]["contours"][:])
        / len(output_full_["nuclei_segmentation"]["contours"][:])
        > 0.9
    )


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
        return_predictions=(True,),
    )

    output_ = zarr.open(output[wsi4_512_512_svs], mode="r")
    assert 37 < np.mean(output_["predictions"][:]) < 41
    assert "probabilities" in output_
    assert "canvas" not in output_
    assert "count" not in output_

    output_ = zarr.open(output[wsi4_512_512_svs_2], mode="r")
    assert 37 < np.mean(output_["predictions"][:]) < 41
    assert "probabilities" in output_
    assert "canvas" not in output_
    assert "count" not in output_


def test_wsi_segmentor_annotationstore(
    remote_sample: Callable, track_tmp_path: Path
) -> None:
    """Test MultiTaskSegmentor for WSIs with AnnotationStore output."""
    wsi4_512_512_svs = remote_sample("wsi4_512_512_svs")
    # testing different configuration for hovernet.
    # kumar only has two probability maps
    model_name = "hovernet_original-kumar"
    mtsegmentor = MultiTaskSegmentor(
        model=model_name,
        batch_size=32,
        verbose=False,
    )

    class_dict = mtsegmentor._get_model_attr("class_dict")

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


def test_wsi_segmentor_qupath(remote_sample: Callable, track_tmp_path: Path) -> None:
    """Test MultiTaskSegmentor for WSIs with AnnotationStore output."""
    wsi4_512_512_svs = remote_sample("wsi4_512_512_svs")
    # testing different configuration for hovernet.
    # kumar only has two probability maps
    # Need to test Null values in JSON output.
    model_name = "hovernet_original-kumar"
    mtsegmentor = MultiTaskSegmentor(
        model=model_name,
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
        output_type="qupath",
        class_dict=class_dict,
        memory_threshold=0,
    )

    for output_ in output[wsi4_512_512_svs]:
        assert output_.suffix != ".zarr"

    json_file_name = f"{wsi4_512_512_svs.stem}.json"
    json_file_name = track_tmp_path / "wsi_out_check" / json_file_name
    assert json_file_name.exists()
    assert json_file_name == output[wsi4_512_512_svs][0]

    # Weights not used after this test
    weights_path = Path(fetch_pretrained_weights(model_name=model_name))
    weights_path.unlink()


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

    assert "Probability maps cannot be saved as AnnotationStore or JSON." in caplog.text
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

    # inst_dict must contain boxes
    inst_dict = {
        1: {"box": np.array([81, 0, 96, 9])},
        2: {"box": np.array([138, 0, 151, 8])},
    }

    invalid_tile_mode = 99  # not in [0,1,2,3]
    ioconfig = mtsegmentor.ioconfig
    ioconfig.margin = 128
    with pytest.raises(ValueError, match=r".*Unknown tile mode.*"):
        _get_sel_indices_margin_lines(
            ioconfig=ioconfig,
            tile_shape=(492, 492),
            tile_flag=(0, 1, 0, 1),
            tile_mode=invalid_tile_mode,
            tile_tl=(0, 0),
            inst_dict=inst_dict,
        )


def test_clear_zarr() -> None:
    """Test _clear_zarr working appropriately.

    This test only covers scenarios which are not feasible to run on GitHub Actions.

    """
    store = zarr.storage.MemoryStore()
    root = zarr.group(store=store)

    # Create a dummy zarr array for probabilities_zarr
    probabilities_zarr = root.create_dataset(
        "probabilities",
        data=np.zeros((5, 3, 3)),
        shape=(5, 3, 3),
    )

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

        available = 0  # force used_percent > memory_threshold

    monkeypatch.setattr(psutil, "virtual_memory", FakeVM)

    # --- Real dask array ---
    da_arr = da.from_array(np.array([[1, 2, 3]]), chunks=(1, 3))
    probabilities_da = [da_arr]

    # --- probabilities_zarr slot is None to trigger the branch ---
    probabilities_zarr = [None]

    # --- Real numpy array for shape/dtype ---
    probabilities = np.zeros((1, 3))

    tqdm_loop = tqdm(
        range(1),
    )

    # --- Call function ---
    new_zarr, new_da = _save_multitask_vertical_to_cache(
        probabilities_zarr=probabilities_zarr,
        probabilities_da=probabilities_da,
        probabilities=probabilities,
        idx=idx,
        tqdm_loop=tqdm_loop,
        save_path=tmp_path / "cache.zarr",
        chunk_shape=(1,),
        memory_threshold=0,  # ensure branch triggers
    )

    # probabilities_da must be set to None
    assert new_da[idx] is None

    # new_zarr must be a real zarr array
    assert isinstance(new_zarr[idx], zarr.Array)

    # Data was written correctly
    assert np.array_equal(new_zarr[idx][:], np.array([[1, 2, 3]]))


def test_qupath_feature_class_dict_lookup_fails() -> None:
    """Test qupath_feature_class_dict lookup fails."""
    qupath_json = DaskDelayedJSONStore.__new__(DaskDelayedJSONStore)
    qupath_json._contours = [np.array([[0, 0], [1, 0], [1, 1]])]
    qupath_json._processed_predictions = {"type": np.array([5], dtype=object)}

    class_dict = {0: "A", 1: "B"}  # does NOT contain 5
    class_colors = {0: [255, 0, 0], 1: [0, 255, 0]}  # also does NOT contain 5

    feat = qupath_json._build_single_qupath_feature(
        i=0,
        class_dict=class_dict,
        origin=(0, 0),
        scale_factor=(1, 1),
        class_colors=class_colors,
    )

    # type should fall back to raw value (5)
    assert feat["properties"]["type"] == 5
    # classification block should NOT appear
    assert "classification" not in feat["properties"]


def test_qupath_feature_classification_block_skipped() -> None:
    """Test qupath_feature_classification_block_skipped fails."""
    qupath_json = DaskDelayedJSONStore.__new__(DaskDelayedJSONStore)
    qupath_json._contours = [np.array([[0, 0], [1, 0], [1, 1]])]
    qupath_json._processed_predictions = {"type": np.array([1], dtype=object)}

    class_dict = {1: "Tumor"}
    class_colors = {0: [255, 0, 0]}  # does NOT contain 1

    feat = qupath_json._build_single_qupath_feature(
        i=0,
        class_dict=class_dict,
        origin=(0, 0),
        scale_factor=(1, 1),
        class_colors=class_colors,
    )

    assert feat["properties"]["type"] == "Tumor"
    assert "classification" not in feat["properties"]


def test_compute_qupath_json_valid_ids_not_empty(track_tmp_path: Path) -> None:
    """Test compute_qupath_json valid ids not empty."""
    store = DaskDelayedJSONStore.__new__(DaskDelayedJSONStore)

    # One simple contour
    store._contours = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])]

    # Mixed type array → valid_ids = [1, 2]
    store._processed_predictions = {"type": np.array([1, None, 2], dtype=object)}

    out_path = track_tmp_path / "out.json"
    result_path = store.compute_qupath_json(
        class_dict=None,
        save_path=out_path,
        verbose=False,
    )

    # Load JSON
    data = json.loads(Path(result_path).read_text())
    props = data["features"][0]["properties"]

    # 1. class_dict should have been inferred as {0:0, 1:1, 2:2}
    assert props["type"] in (1, 2)

    # 2. type must NOT be null
    assert props["type"] is not None

    # 3. classification block should exist only if class_value in class_colours
    assert "null" not in json.dumps(data)


def test_compute_qupath_json_string_class_names(track_tmp_path: Path) -> None:
    """Test compute_qupath_json string class names not empty and str."""
    store = DaskDelayedJSONStore.__new__(DaskDelayedJSONStore)

    # One simple contour
    store._contours = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])]

    # String class names → triggers the "already class names" branch
    store._processed_predictions = {
        "type": np.array(["Tumor", None, "Stroma"], dtype=object)
    }

    # Run compute_qupath_json with class_dict=None
    out_path = track_tmp_path / "out.json"
    result_path = store.compute_qupath_json(
        class_dict=None,
        save_path=out_path,
        verbose=False,
    )

    # Load JSON
    data = json.loads(Path(result_path).read_text())
    props = data["features"][0]["properties"]

    # --- Assertions ---

    # 1. type must be one of the string class names
    assert props["type"] in ("Tumor", "Stroma")

    # 2. type must NOT be null
    assert props["type"] is not None

    # 3. class_dict should have been inferred as identity mapping
    #    "Stroma": "Stroma", "Tumor": "Tumor"
    #    So classification block should exist only if class_colours
    #    contains the key, but we don't enforce that here — just
    #    ensure no nulls
    assert "null" not in json.dumps(data)


def test_get_tile_info_small_image_triggers_early_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests _get_tile_info.

    Ensures that when image_shape <= tile_shape, the function returns
    [[boxes, flag]] with flag = zeros.

    """
    # --- Arrange ---
    image_shape = [100, 100]

    # Configure tile_shape so that tile_shape >= image_shape
    ioconfig = IOSegmentorConfig(
        tile_shape=[200, 200],
        patch_output_shape=[200, 200],
        margin=None,
        input_resolutions=[{"units": "mpp", "resolution": 0.25}],
        patch_input_shape=(200, 200),
    )

    # Fake return from PatchExtractor.get_coordinates
    fake_boxes = np.array([[0, 0, 100, 100]])
    monkeypatch.setattr(
        "tiatoolbox.tools.patchextraction.PatchExtractor.get_coordinates",
        lambda **_: (None, fake_boxes),
    )

    # Create a dummy dataset with required attributes
    dummy_dataset = SimpleNamespace(mask_reader=None)

    # Dummy dataloader-like container
    dummy_dataloader = SimpleNamespace(dataset=dummy_dataset)

    # Create a real instance and inject required fields
    m = MultiTaskSegmentor(model="hovernet_fast-pannuke")
    m._ioconfig = ioconfig
    m.mask_padding = (0, 0, 0, 0)
    m.dataloader = dummy_dataloader

    # --- Act ---
    result = m._get_tile_info(image_shape=image_shape, wsi_proc_shape=image_shape)

    # --- Assert ---
    assert isinstance(result, list)
    assert len(result) == 1

    boxes, flag = result[0]

    assert np.array_equal(boxes, fake_boxes)
    assert flag.shape == (1, 4)
    assert np.all(flag == 0)


class FakeSeg(MultiTaskSegmentor):
    """Minimal subclass that allows us to override internals cleanly."""

    def __init__(self: FakeSeg) -> None:
        """Initialize the FakeSeg."""
        # Pretend we have one task
        super().__init__(model="hovernetplus-oed")
        self.tasks = {"instance"}
        self.return_predictions_dict = {"instance": True}

    def _get_model_attr(self: FakeSeg, name: str = "a") -> dict:
        """Pretend the model has a class_dict."""
        return {"instance": {name: 1}}

    # These will be patched in the test
    _save_predictions_as_dict_zarr = MagicMock()
    _save_predictions_as_json_store = MagicMock()


def test_save_predictions_includes_coordinates(track_tmp_path: Path) -> None:
    """Test save predictions includes coordinates.

    Ensures that when 'coordinates' is present in processed_predictions,
    the method merges it into dict_for_store before calling
    _save_predictions_as_json_store.

    """
    seg = FakeSeg()

    # processed_predictions returned by _save_predictions_as_dict_zarr
    processed_predictions = {
        "instance": {"predictions": [1, 2, 3]},
        "coordinates": [10, 20, 30],
    }

    # Make the dict-zarr saver return our processed_predictions
    seg._save_predictions_as_dict_zarr.return_value = processed_predictions

    # Make the json-store saver return a fake path list
    seg._save_predictions_as_json_store.return_value = [track_tmp_path / "out.db"]

    save_path = track_tmp_path / "result"

    # --- Act ---
    seg.save_predictions(
        processed_predictions=processed_predictions,
        output_type="annotationstore",
        save_path=save_path,
        return_probabilities=False,
        return_predictions=(False,),  # ensures output_type_ becomes "dict"
    )

    # --- Assert ---
    assert seg._save_predictions_as_json_store.called

    # Extract the dict passed to the JSON store saver
    call_args = seg._save_predictions_as_json_store.call_args.kwargs
    dict_for_store = call_args["processed_predictions"]

    # Must contain both the task predictions and the coordinates
    assert dict_for_store["predictions"] == [1, 2, 3]
    assert dict_for_store["coordinates"] == [10, 20, 30]


def test_merge_stops_when_should_stop(
    monkeypatch: pytest.MonkeyPatch, track_tmp_path: Path
) -> None:
    """Test should stop in multitask vertical merge."""
    canvas = [da.from_array(np.ones((1, 4, 4, 1)), chunks=(1, 4, 4, 1))]
    count = [da.from_array(np.ones((1, 4, 4, 1)), chunks=(1, 4, 4, 1))]

    # output_locs_y_ → only one row, so overlaps=[0]
    output_locs_y_ = np.array([[0, 4]])

    # Zarr group
    store = zarr.open_group(str(track_tmp_path / "test.zarr"), mode="w")

    # --- Force should_stop=True on first iteration ---
    output_shape = (0, 4)  # height=0 → remaining_height <= 0 → should_stop=True

    # --- Mock functions that should NOT be called ---
    called_store = False

    def fake_store_probabilities(
        *_: Any,  # noqa: ANN401
        **__: Any,  # noqa: ANN401
    ) -> tuple[zarr.Array | None, da.Array | None]:
        nonlocal called_store
        called_store = True
        return None, None

    monkeypatch.setattr(
        "tiatoolbox.models.engine.semantic_segmentor.store_probabilities",
        fake_store_probabilities,
    )

    monkeypatch.setattr(
        "tiatoolbox.models.engine.multi_task_segmentor._save_multitask_vertical_to_cache",
        lambda **__: (None, None),
    )

    monkeypatch.setattr(
        "tiatoolbox.models.engine.multi_task_segmentor._clear_zarr",
        lambda **__: da.zeros((0, 4, 1)),
    )

    # --- Act ---
    result = merge_multitask_vertical_chunkwise(
        canvas=canvas,
        count=count,
        output_locs_y_=output_locs_y_,
        zarr_group=store,
        save_path=track_tmp_path,
        output_shape=output_shape,
        verbose=False,
    )

    # store_probabilities should NOT be called because break happens first
    assert called_store is False

    # Returned array should be empty height
    assert result[0].shape[0] == 0


# HELPER functions
def assert_output_lengths(
    output: OutputType, expected_counts: Sequence[int], fields: list[str]
) -> None:
    """Assert lengths of output dict fields against expected counts."""
    for field in fields:
        for i, expected in enumerate(expected_counts):
            idx = str(i) if isinstance(output[field], zarr.Group) else i
            assert len(np.asarray(output[field][idx], dtype=object)) == expected, (
                f"{field}[{idx}] mismatch"
            )


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


def test_process_instance_predictions_empty_inst_dict() -> None:
    """Test _process_instance_predictions, when no nuclei detected in input images."""
    inst_dict = {}  # triggers the branch
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        patch_input_shape=(2048, 2048),
        patch_output_shape=(1024, 1024),
        stride_shape=(512, 512),
    )
    tile_shape = (256, 256)
    tile_flag = (0, 0, 0, 0)
    tile_mode = 0
    tile_tl = (0, 0)
    ref_inst_dict = {}
    ref_inst_rtree = STRtree([Point(0, 0)])  # dummy tree, never used

    result = _process_instance_predictions(
        inst_dict=inst_dict,
        ioconfig=ioconfig,
        tile_shape=tile_shape,
        tile_flag=tile_flag,
        tile_mode=tile_mode,
        tile_tl=tile_tl,
        ref_inst_dict=ref_inst_dict,
        ref_inst_rtree=ref_inst_rtree,
    )

    assert result == ({}, [])


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
            i_a_ = str(i_a) if isinstance(output_a[field], zarr.Group) else i_a
            i_b_ = str(i_b) if isinstance(output_b[field], zarr.Group) else i_b
            left = np.asarray(output_a[field][i_a_])
            right = np.asarray(output_b[field][i_b_])
            assert all(
                np.array_equal(a, b) for a, b in zip(left, right, strict=False)
            ), f"{field}[{i_a_}] vs {field}[{i_b_}] mismatch"


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
        store_file_name = _get_store_file_name(
            inputs=inputs,
            task_name=task_name,
            patch_idx=patch_idx,
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
            contours = output_dict["contours"][patch_idx]
            pad_value = np.iinfo(contours.dtype).min
            contours = np.array(
                [row[~(np.asarray(row) == pad_value).all(axis=1)] for row in contours],
                dtype=object,
            )
            matches = [
                np.array_equal(np.array(a[:-1], dtype=int), np.array(b, dtype=int))
                for a, b in zip(result["contours"], contours, strict=False)
            ]
            # Due to make valid poly there might be translation in a few points
            # in AnnotationStore
            assert sum(matches) / len(matches) >= 0.95
        else:
            assert annotations_geometry_type == []
            assert annotations_list == []


def _get_store_file_name(
    inputs: list | np.ndarray,
    task_name: str | None,
    patch_idx: int,
) -> str:
    """Helper function to get store filename."""
    if isinstance(inputs[patch_idx], Path):
        return (
            f"{inputs[patch_idx].stem}.db"
            if task_name is None
            else f"{inputs[patch_idx].stem}_{task_name}.db"
        )

    return f"{patch_idx}.db" if task_name is None else f"{patch_idx}_{task_name}.db"


def assert_qupath_json_patch_output(  # skipcq: PY-R1000
    inputs: list | np.ndarray,
    output_json: list[Path],
    task_name: str | None,
    track_tmp_path: Path,
    expected_counts: Sequence[int],
    output_dict: dict,
    fields: list[str],
) -> None:
    """Helper function to test QuPath JSON output."""
    for patch_idx, json_path in enumerate(output_json):
        # --- 1. Verify filename matches expected pattern ---
        if isinstance(inputs[patch_idx], Path):
            file_name = (
                f"{inputs[patch_idx].stem}.json"
                if task_name is None
                else f"{inputs[patch_idx].stem}_{task_name}.json"
            )
        else:
            file_name = (
                f"{patch_idx}.json"
                if task_name is None
                else f"{patch_idx}_{task_name}.json"
            )

        assert json_path == track_tmp_path / "patch_output_qupath" / file_name

        # --- 2. Load JSON ---
        with Path.open(json_path, "r") as f:
            qupath_json = json.load(f)

        features = qupath_json.get("features", [])
        assert isinstance(features, list)

        # --- 3. Zero-object case ---
        if expected_counts[patch_idx] == 0:
            assert len(features) == 0
            continue

        # --- 4. Non-zero case ---
        assert len(features) == expected_counts[patch_idx]

        # Extract results from JSON
        result = {field: [] for field in fields}

        for feat in features:
            props = feat.get("properties", {})

            # non-geometric fields (box, centroid, prob, type, etc.)
            for field in fields:
                if field == "contours":
                    continue
                if field in props:
                    result[field].append(props[field])

            # contours from geometry
            if "contours" in fields:
                geom = feat["geometry"]
                coords = geom["coordinates"][0]  # exterior ring
                coords = [(int(x), int(y)) for x, y in coords]
                result["contours"].append(coords)

        # Wrap for compatibility with assert_output_lengths
        result_wrapped = {field: [result[field]] for field in fields}

        # --- 5. Length check ---
        assert_output_lengths(
            result_wrapped,
            expected_counts=[expected_counts[patch_idx]],
            fields=fields,
        )

        # --- 6. Equality check for non-contour fields ---
        fields_no_contours = fields.copy()
        if "contours" in fields_no_contours:
            fields_no_contours.remove("contours")

        assert_output_equal(
            result_wrapped,
            output_dict,
            fields=fields_no_contours,
            indices_a=[0],
            indices_b=[patch_idx],
        )

        # --- 7. Contour comparison ---
        contours = output_dict["contours"][patch_idx]
        pad_value = np.iinfo(contours.dtype).min
        contours = np.array(
            [row[~(np.asarray(row) == pad_value).all(axis=1)] for row in contours],
            dtype=object,
        )
        if "contours" in fields:
            matches = []
            for a, b in zip(
                result["contours"],
                contours,
                strict=False,
            ):
                # Discard last point (closed polygon)
                a_arr = np.array(a[:-1], dtype=int)
                b_arr = np.array(b, dtype=int)
                matches.append(np.array_equal(a_arr, b_arr))

            # Allow small geometric differences
            assert sum(matches) / len(matches) >= 0.95


def convert_to_dask(output_dict: dict | list[dict]) -> dict | list[dict]:
    """Helper function to convert dict with np arrays into a dict with dask arrays."""
    if isinstance(output_dict, dict):
        return {k: convert_to_dask(v) for k, v in output_dict.items()}
    if isinstance(output_dict, list):
        if all(isinstance(x, str) for x in output_dict):
            arr = np.array(output_dict, dtype=object)
            return da.from_array(arr, chunks=(len(arr),))
        return [convert_to_dask(x) for x in output_dict]
    if isinstance(output_dict, np.ndarray):
        if output_dict.dtype == object:
            # Force chunking for object arrays
            return da.from_array(output_dict, chunks=(1,) * output_dict.ndim)
        return da.from_array(output_dict)
    return output_dict


def convert_to_dask_single_task(
    output_dict: dict | list[dict], task_name: str
) -> dict | list[dict]:
    """Helper to convert a dict into a dict with dask arrays for single task."""
    processed_predictions = {task_name: {}}
    for k, v in output_dict.items():
        if k == "probabilities":
            processed_predictions[k] = [da.from_array(v_) for v_ in v]
            continue
        if isinstance(v, np.ndarray):
            processed_predictions[task_name][k] = da.from_array(v)
        if isinstance(v, list):
            processed_predictions[task_name][k] = []
            for v_ in v:
                chunks = (len(v_),) if v_.dtype == object else "auto"
                processed_predictions[task_name][k].append(
                    da.from_array(v_, chunks=chunks)
                )

    return processed_predictions


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
            "multitask-segmentor",
            "--img-input",
            str(wsi4_512_512_svs),
            "--patch-mode",
            "False",
            "--output-path",
            str(track_tmp_path / "output"),
            "--return-predictions",
            "False, True",
        ],
    )

    assert models_wsi_result.exit_code == 0
    assert (
        track_tmp_path / "output" / f"{wsi4_512_512_svs.stem}_layer_segmentation.db"
    ).exists()
    assert (
        track_tmp_path / "output" / f"{wsi4_512_512_svs.stem}_nuclei_segmentation.db"
    ).exists()
    zarr_group = zarr.open(
        str(track_tmp_path / "output" / f"{wsi4_512_512_svs.stem}.zarr"), mode="r"
    )
    assert "probabilities" in zarr_group
    assert "nuclei_segmentation" not in zarr_group
    assert "layer_segmentation" in zarr_group
    assert "predictions" in zarr_group["layer_segmentation"]


def test_rearrange_raw_predictions_skips_private_subkeys() -> None:
    """Tests private keys not in output dict."""
    # Create a fake task name
    tasks = {"taskA"}

    # Create raw_predictions structured so that:
    # - values is a list of dicts
    # - each dict contains a subkey starting with "_"
    raw_predictions = {
        "taskA": {
            "some_key": [
                {"_private": 1, "public": 10},
                {"_private": 2, "public": 20},
            ]
        }
    }

    # Call the staticmethod
    out = MultiTaskSegmentor._rearrange_raw_predictions_to_per_task_dict(
        tasks, raw_predictions
    )

    # The "_private" key should be skipped entirely
    assert "_private" not in out["taskA"]

    # The "public" key should be added
    assert out["taskA"]["public"] == [10, 20]

    # The original key should be deleted
    assert "some_key" not in out["taskA"]


def test_post_save_json_store_deletes_empty_store(
    track_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test zarr store deletion post save JSON."""
    # Create an empty Zarr v3 store
    store_root = track_tmp_path / "empty_store.zarr"
    store = LocalStore(str(store_root))
    root = zarr.open(store, mode="w")  # empty zarr.Group

    assert list(root.keys()) == []

    # ---- Proxy object that LOOKS like a zarr.Group ----
    class GroupProxy:
        def __init__(self: GroupProxy, group: zarr.Group, path: Path | str) -> None:
            self._group = group
            self.path = path
            self.store = group.store

        # Make isinstance(proxy, zarr.Group) return True
        @property
        def __class__(self: GroupProxy) -> type[zarr.Group]:
            return zarr.Group

        # Delegate attribute access
        def __getattr__(
            self: GroupProxy, item: str
        ) -> zarr.Group | zarr.Array | str | int | float | Iterable[str]:
            return getattr(self._group, item)

        # Delegate mapping behavior
        def keys(self: GroupProxy) -> Iterable[str]:
            return self._group.keys()

        def __getitem__(self: GroupProxy, item: str) -> zarr.Group | zarr.Array:
            return self._group[item]

    processed_predictions = GroupProxy(root, "dummy")

    # Patch shutil.rmtree so we can detect the call
    called = {"flag": False}

    def fake_rmtree(path: Path | str, *, ignore_errors: bool) -> None:  # noqa: ARG001
        called["flag"] = True

    monkeypatch.setattr(shutil, "rmtree", fake_rmtree)

    # Call the function
    _post_save_json_store(
        keys_to_compute=[],
        processed_predictions=processed_predictions,
        save_path=None,
    )

    # Assert deletion branch executed
    assert called["flag"] is True
