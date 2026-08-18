"""Test MultiTaskSegmentor."""

from __future__ import annotations

import importlib
import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Final
from unittest.mock import MagicMock, Mock, patch

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
from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.engine import multi_task_segmentor
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
_RUNNING_ON_CI = toolbox_env.running_on_ci()


def test_mtsegmentor_init() -> None:
    """Tests MultiTaskSegmentor initialization."""
    segmentor = MultiTaskSegmentor(model="hovernetplus-oed", device=device)

    assert isinstance(segmentor, MultiTaskSegmentor)
    assert isinstance(segmentor.model, torch.nn.Module)


def test_run_raises_when_return_labels_true() -> None:
    """MultiTaskSegmentor does not support return_labels."""
    segmentor = MultiTaskSegmentor.__new__(MultiTaskSegmentor)

    with pytest.raises(
        ValueError,
        match=r"`return_labels` is not supported for MultiTaskSegmentor\.",
    ):
        segmentor.run(
            images=np.zeros((1, 256, 256, 3), dtype=np.uint8),
            return_labels=True,
        )


def test_post_process_patches() -> None:
    """Test patch-level post-processing."""
    segmentor = Mock()

    def mock_postproc_func(
        probs: list[object],
        offset: tuple[int, int],
    ) -> dict[str, object]:
        """Mock post-processing function for testing."""
        _ = offset
        return {"task_type": "seg", "data": probs}

    segmentor._get_model_attr.return_value = mock_postproc_func
    segmentor.build_post_process_raw_predictions.return_value = {
        "result": "ok",
    }

    raw_predictions = {
        "probabilities": [
            ["a", "b"],
            ["c", "d"],
        ],
    }

    result = MultiTaskSegmentor.post_process_patches(
        segmentor,
        raw_predictions,
        return_predictions=[True],
    )

    assert result == {"result": "ok"}

    segmentor.build_post_process_raw_predictions.assert_called_once()


def test_post_process_wsi_tile_mode(track_tmp_path: Path) -> None:
    """Test WSI post-processing using tile mode."""
    segmentor = Mock()

    segmentor._ioconfig.tile_shape = (10, 10)
    segmentor.num_workers = 0
    segmentor.mask_padding = (1, 2, 3, 4)
    segmentor.return_predictions_dict = {}

    probabilities = [np.zeros((20, 20))]

    segmentor._process_tile_mode.return_value = [
        {
            "task_type": "seg",
            "mask": np.ones((5, 5)),
            "stats": {"dice": 0.9},
        }
    ]

    raw_predictions = {
        "probabilities": probabilities,
    }

    result = MultiTaskSegmentor.post_process_wsi(
        segmentor,
        raw_predictions,
        track_tmp_path / "output",
        return_predictions=[True],
        return_probabilities=False,
        num_workers=4,
    )

    segmentor._process_tile_mode.assert_called_once()
    segmentor._process_full_wsi.assert_not_called()

    assert segmentor.return_predictions_dict["seg"] is True
    assert isinstance(result["seg"]["mask"], da.Array)
    assert result["seg"]["dice"] == 0.9
    assert segmentor.tasks == {"seg"}


def test_post_process_wsi_full_wsi(track_tmp_path: Path) -> None:
    """Test WSI post-processing without tile mode."""
    segmentor = Mock()

    segmentor._ioconfig.tile_shape = (100, 100)
    segmentor.mask_padding = (0, 0, 0, 0)
    segmentor.return_predictions_dict = {}
    segmentor.num_workers = 4

    probabilities = [np.zeros((50, 50))]

    segmentor._process_full_wsi.return_value = [
        {
            "task_type": "seg",
            "mask": da.ones((5, 5)),
        }
    ]

    raw_predictions = {
        "probabilities": probabilities,
    }

    result = MultiTaskSegmentor.post_process_wsi(
        segmentor,
        raw_predictions,
        track_tmp_path / "output",
    )

    segmentor._process_full_wsi.assert_called_once()
    segmentor._process_tile_mode.assert_not_called()

    assert segmentor.return_predictions_dict["seg"] is False
    assert isinstance(result["seg"]["mask"], da.Array)
    assert segmentor.tasks == {"seg"}


@patch("tiatoolbox.models.engine.multi_task_segmentor.build_da_pad_width")
def test_post_process_wsi_return_probabilities(
    mock_build_da_pad_width: Mock,
    track_tmp_path: Path,
) -> None:
    """Test probability padding."""
    mock_build_da_pad_width.return_value = ((0, 0), (1, 1))

    segmentor = Mock()
    segmentor._ioconfig.tile_shape = (100, 100)
    segmentor.mask_padding = (1, 1, 1, 1)
    segmentor.return_predictions_dict = {}

    prob = da.ones((5, 5))

    segmentor._process_full_wsi.return_value = [{"task_type": "seg"}]

    raw_predictions = {"probabilities": [prob]}

    MultiTaskSegmentor.post_process_wsi(
        segmentor,
        raw_predictions,
        track_tmp_path / "output",
        return_probabilities=True,
    )

    mock_build_da_pad_width.assert_called_once()


def test_process_full_wsi_mixed_return_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test _process_full_wsi with and without returned predictions."""
    segmentor = MultiTaskSegmentor.__new__(MultiTaskSegmentor)

    segmentor.mask_padding = (1, 2, 3, 4)

    predictions_0 = da.from_array(
        np.ones((4, 4), dtype=np.uint8),
        chunks=(4, 4),
    )

    predictions_1 = da.from_array(
        np.ones((4, 4), dtype=np.uint8),
        chunks=(4, 4),
    )

    def _fake_postproc_func(
        probabilities: list[da.Array | np.ndarray],
        offset: tuple[int, int],
    ) -> list:
        """Return two task outputs."""
        _ = probabilities

        assert offset == (1, 2)

        return [
            {
                "task_type": "task_a",
                "predictions": predictions_0,
            },
            {
                "task_type": "task_b",
                "predictions": predictions_1,
            },
        ]

    def _fake_get_model_attr(
        attr_name: str,
    ) -> Callable:
        """Return fake post-processing function."""
        _ = attr_name

        return _fake_postproc_func

    monkeypatch.setattr(
        segmentor,
        "_get_model_attr",
        _fake_get_model_attr,
    )

    result = segmentor._process_full_wsi(
        probabilities=[
            da.from_array(
                np.zeros((4, 4), dtype=np.float32),
                chunks=(4, 4),
            ),
        ],
        return_predictions=(False, True),
    )

    #
    # False branch:
    # if not return_predictions_:
    #
    assert "predictions" not in result[0]

    #
    # True branch:
    # else -> build_da_pad_width + da.pad
    #
    assert "predictions" in result[1]

    assert result[1]["predictions"].shape == (
        4 + 2 + 4,  # top + bottom padding
        4 + 1 + 3,  # left + right padding
    )


def test_process_full_wsi_default_return_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test default return_predictions behaviour."""
    segmentor = MultiTaskSegmentor.__new__(MultiTaskSegmentor)

    segmentor.mask_padding = (0, 0, 0, 0)

    def _fake_postproc_func(
        probabilities: list[da.Array | np.ndarray],
        offset: tuple[int, int],
    ) -> list:
        """Return one task output."""
        _ = probabilities, offset

        return [
            {
                "task_type": "task_a",
                "predictions": da.from_array(
                    np.ones((2, 2)),
                    chunks=(2, 2),
                ),
            },
        ]

    def _fake_get_model_attr(
        attr_name: str,
    ) -> Callable:
        """Return fake post-processing function."""
        _ = attr_name
        return _fake_postproc_func

    monkeypatch.setattr(
        segmentor,
        "_get_model_attr",
        _fake_get_model_attr,
    )

    result = segmentor._process_full_wsi(
        probabilities=[],
        return_predictions=None,
    )

    assert "predictions" not in result[0]


def test_process_tile_mode_removes_instances(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test removal of overlapping instances during tile merging."""
    segmentor = MultiTaskSegmentor.__new__(MultiTaskSegmentor)

    segmentor.verbose = False
    segmentor.num_workers = 0

    segmentor.mask_bounds = np.array([0, 0, 256, 256])
    segmentor.mask_padding = (0, 0)

    segmentor._ioconfig = SimpleNamespace(
        highest_input_resolution={},
        tile_shape=(256, 256),
        to_baseline=lambda: SimpleNamespace(margin=0),
    )

    def _fake_slide_dimensions() -> tuple[int, int]:
        """Return slide dimensions."""
        return (256, 256)

    segmentor.dataloader = SimpleNamespace(
        dataset=SimpleNamespace(
            reader=SimpleNamespace(
                slide_dimensions=_fake_slide_dimensions,
            ),
        ),
    )

    existing_info_dict = {
        "remove_me": {"type": 1},
    }

    fake_wsi_info_dict = (
        {
            "task_type": "nuclei",
            "predictions": None,
            "info_dict": existing_info_dict,
        },
    )

    monkeypatch.setattr(
        segmentor,
        "_get_tile_info",
        lambda image_shape, wsi_proc_shape: [  # noqa: ARG005
            (
                np.array([0, 0, 128, 128]),
                (0, 0, 0, 0),
                3,
            ),
        ],
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "_build_tile_tasks",
        lambda tile_info_sets, verbose: tile_info_sets,  # noqa: ARG005
    )

    monkeypatch.setattr(
        segmentor,
        "_compute_tile",
        lambda tile_meta: tile_meta,
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "tqdm_dask_progress_bar",
        lambda **kwargs: [  # noqa: ARG005
            (
                {
                    "task_type": "nuclei",
                    "predictions": np.array([[1]], dtype=np.int32),
                    "info_dict": {},
                    "seg_type": "instance",
                },
            ),
        ],
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "_create_wsi_info_dict",
        lambda **kwargs: fake_wsi_info_dict,  # noqa: ARG005
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "_update_tile_based_predictions_array",
        lambda **kwargs: (fake_wsi_info_dict, None),  # noqa: ARG005
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "_get_inst_info_dicts",
        lambda **kwargs: [{}],  # noqa: ARG005
    )

    def _fake_compute_info_dict_for_merge(
        **kwargs: object,
    ) -> tuple[dict, list[str]]:
        """Return an instance and one UUID to remove."""
        _ = kwargs

        return (
            {
                "new_uuid": {
                    "type": 1,
                },
            },
            ["remove_me"],
        )

    monkeypatch.setattr(
        multi_task_segmentor,
        "_compute_info_dict_for_merge",
        _fake_compute_info_dict_for_merge,
    )

    monkeypatch.setattr(
        segmentor,
        "_inst_dict_for_dask_processing",
        lambda wsi_info_dict: wsi_info_dict,
    )

    probabilities = [
        np.zeros((1, 1, 1), dtype=np.float32),
    ]

    result = segmentor._process_tile_mode(
        probabilities=probabilities,
        save_path=track_tmp_path / "tmp.zarr",
        return_predictions=(False,),
    )

    assert "remove_me" not in result[0]["info_dict"]
    assert "new_uuid" in result[0]["info_dict"]


def test_build_post_process_raw_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test building raw predictions from post-processed outputs."""
    segmentor = MultiTaskSegmentor.__new__(MultiTaskSegmentor)
    segmentor.return_predictions_dict = {}
    rearrange_called: dict[str, object] = {}

    def _fake_rearrange_raw_predictions_to_per_task_dict(
        tasks: set[str],
        raw_predictions: dict,
    ) -> dict:
        """Return predictions unchanged and capture arguments."""
        rearrange_called["tasks"] = tasks
        rearrange_called["raw_predictions"] = raw_predictions

        return raw_predictions

    monkeypatch.setattr(
        segmentor,
        "_rearrange_raw_predictions_to_per_task_dict",
        _fake_rearrange_raw_predictions_to_per_task_dict,
    )

    raw_predictions: dict[str, dict] = {}

    post_process_predictions = [
        (
            {
                "task_type": "nuclei",
                "predictions": np.array([1]),
                "info_dict": {"a": 1},
            },
            {
                "task_type": "layers",
                "predictions": np.array([2]),
                "info_dict": {"b": 2},
            },
        ),
        (
            {
                "task_type": "nuclei",
                "predictions": np.array([3]),
                "info_dict": {"c": 3},
            },
            {
                "task_type": "layers",
                "predictions": np.array([4]),
                "info_dict": {"d": 4},
            },
        ),
    ]

    result = segmentor.build_post_process_raw_predictions(
        post_process_predictions=post_process_predictions,
        raw_predictions=raw_predictions,
        return_predictions=(True, False),
    )

    assert result is raw_predictions

    assert segmentor.return_predictions_dict == {
        "nuclei": True,
        "layers": False,
    }

    assert segmentor.tasks == {
        "nuclei",
        "layers",
    }

    assert rearrange_called["tasks"] == {
        "nuclei",
        "layers",
    }

    assert raw_predictions["nuclei"]["predictions"] == [
        np.array([1]),
        np.array([3]),
    ]

    assert raw_predictions["layers"]["predictions"] == [
        np.array([2]),
        np.array([4]),
    ]

    assert raw_predictions["nuclei"]["info_dict"] == [
        {"a": 1},
        {"c": 3},
    ]

    assert raw_predictions["layers"]["info_dict"] == [
        {"b": 2},
        {"d": 4},
    ]


def test_rearrange_raw_predictions_to_per_task_dict_array_values() -> None:
    """Test stacking array values into a Dask array."""
    raw_predictions = {
        "task_a": {
            "predictions": [
                np.array([[1, 2]]),
                np.array([[3, 4]]),
            ],
        },
    }

    result = MultiTaskSegmentor._rearrange_raw_predictions_to_per_task_dict(
        tasks={"task_a"},
        raw_predictions=raw_predictions,
    )

    assert isinstance(
        result["task_a"]["predictions"],
        da.Array,
    )

    np.testing.assert_array_equal(
        result["task_a"]["predictions"].compute(),
        np.array(
            [
                [[1, 2]],
                [[3, 4]],
            ],
        ),
    )


def test_save_predictions_as_dict_multiple_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test dict output does not flatten multi-task predictions."""
    segmentor = MultiTaskSegmentor.__new__(MultiTaskSegmentor)

    segmentor.tasks = {
        "task_a",
        "task_b",
    }

    captured: dict[str, object] = {}

    captured: dict[str, object] = {}

    def _fake_save_predictions(
        self: object,
        processed_predictions: dict,
        output_type: str,
        save_path: Path | None = None,
        **kwargs: object,
    ) -> dict:
        """Capture parent save_predictions call."""
        _ = self, output_type, save_path, kwargs
        captured["processed_predictions"] = processed_predictions

        return processed_predictions

    monkeypatch.setattr(
        SemanticSegmentor,
        "save_predictions",
        _fake_save_predictions,
    )

    processed_predictions = {
        "task_a": {"value": 1},
        "task_b": {"value": 2},
    }

    result = segmentor._save_predictions_as_dict_zarr(
        processed_predictions=processed_predictions,
        output_type="dict",
    )

    assert result == processed_predictions
    assert "task_a" in captured["processed_predictions"]
    assert "task_b" in captured["processed_predictions"]


def test_save_predictions_as_dict_zarr_multitask(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test zarr saving for multiple tasks without probabilities."""
    segmentor = MultiTaskSegmentor.__new__(MultiTaskSegmentor)

    segmentor.tasks = {
        "task_a",
        "task_b",
    }

    segmentor.drop_keys = set()

    calls: list[dict[str, object]] = []

    def _fake_save_predictions_as_zarr(
        processed_predictions: dict,
        save_path: Path,
        keys_to_compute: list[str],
        task_name: str | None,
    ) -> Path:
        """Record save requests."""
        calls.append(
            {
                "processed_predictions": processed_predictions,
                "keys_to_compute": keys_to_compute,
                "task_name": task_name,
            },
        )

        return save_path

    monkeypatch.setattr(
        segmentor,
        "save_predictions_as_zarr",
        _fake_save_predictions_as_zarr,
    )

    save_path = track_tmp_path / "output.zarr"

    result = segmentor._save_predictions_as_dict_zarr(
        processed_predictions={
            "task_a": {
                "contours": np.array([1]),
            },
            "task_b": {
                "contours": np.array([2]),
            },
        },
        output_type="zarr",
        save_path=save_path,
        return_probabilities=False,
    )

    assert result == save_path

    assert len(calls) == 2

    assert {call["task_name"] for call in calls} == {
        "task_a",
        "task_b",
    }


def test_save_predictions_as_json_store_keeps_predictions_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test predictions are not deleted when return_predictions is True."""
    segmentor = MultiTaskSegmentor.__new__(MultiTaskSegmentor)

    segmentor.patch_mode = False
    segmentor.verbose = False
    segmentor.num_workers = 1

    segmentor.return_predictions_dict = {
        "nuclei": True,
    }

    def _fake_get_model_attr(attr_name: str) -> dict:
        """Return fake class dictionary."""
        _ = attr_name
        return {}

    def _fake_save_annotation_json_store(
        **kwargs: object,
    ) -> Path:
        """Avoid file writing."""
        _ = kwargs
        return track_tmp_path / "output.db"

    def _fake_post_save_json_store(
        **kwargs: object,
    ) -> None:
        """Avoid cleanup."""
        _ = kwargs

    monkeypatch.setattr(
        segmentor,
        "_get_model_attr",
        _fake_get_model_attr,
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "_save_annotation_json_store",
        _fake_save_annotation_json_store,
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "_post_save_json_store",
        _fake_post_save_json_store,
    )

    processed_predictions = {
        "predictions": np.array([1]),
        "contours": np.array([2]),
    }

    segmentor._save_predictions_as_json_store(
        processed_predictions=processed_predictions,
        task_name="nuclei",
        save_path=track_tmp_path / "output.db",
    )

    assert "predictions" in processed_predictions


def test_save_predictions_as_dict_zarr_probabilities(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test probability zarr output."""
    segmentor = MultiTaskSegmentor.__new__(MultiTaskSegmentor)

    segmentor.tasks = {"task_a"}
    segmentor.drop_keys = set()

    probability_calls: list[dict[str, object]] = []

    def _fake_save_predictions_as_zarr(
        processed_predictions: dict,
        save_path: Path,
        keys_to_compute: list[str],
        task_name: str | None,
    ) -> Path:
        """Record save requests."""
        probability_calls.append(
            {
                "processed_predictions": processed_predictions,
                "keys_to_compute": keys_to_compute,
                "task_name": task_name,
            },
        )

        return save_path

    monkeypatch.setattr(
        segmentor,
        "save_predictions_as_zarr",
        _fake_save_predictions_as_zarr,
    )

    segmentor._save_predictions_as_dict_zarr(
        processed_predictions={
            "probabilities": np.array([1]),
            "task_a": {
                "contours": np.array([1]),
            },
        },
        output_type="zarr",
        save_path=track_tmp_path / "output.zarr",
        return_probabilities=True,
    )

    assert probability_calls[0]["keys_to_compute"] == [
        "probabilities",
    ]

    assert probability_calls[0]["task_name"] is None


def test_save_predictions_as_dict_zarr_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test coordinate propagation to task output."""
    segmentor = MultiTaskSegmentor.__new__(MultiTaskSegmentor)

    segmentor.tasks = {"task_a"}
    segmentor.drop_keys = set()

    captured: dict[str, object] = {}

    def _fake_save_predictions_as_zarr(
        processed_predictions: dict,
        save_path: Path,
        keys_to_compute: list[str],
        task_name: str | None,
    ) -> Path:
        """Capture arguments."""
        _ = save_path, task_name

        captured["processed_predictions"] = processed_predictions
        captured["keys_to_compute"] = keys_to_compute

        return track_tmp_path / "output.zarr"

    monkeypatch.setattr(
        segmentor,
        "save_predictions_as_zarr",
        _fake_save_predictions_as_zarr,
    )

    segmentor._save_predictions_as_dict_zarr(
        processed_predictions={
            "coordinates": np.array([[0, 0]]),
            "task_a": {
                "contours": np.array([1]),
            },
        },
        output_type="zarr",
        save_path=track_tmp_path / "output.zarr",
    )

    assert "coordinates" in captured["processed_predictions"]

    assert "coordinates" in captured["keys_to_compute"]


def test_save_predictions_as_json_store_patch_mode(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test patch-mode AnnotationStore saving."""
    segmentor = MultiTaskSegmentor.__new__(MultiTaskSegmentor)

    segmentor.patch_mode = True
    segmentor.verbose = False
    segmentor.num_workers = 0

    segmentor.images = [
        track_tmp_path / "img1.png",
    ]

    segmentor.return_predictions_dict = {
        "nuclei": False,
    }

    save_calls: list[dict[str, object]] = []
    post_save_called = False

    def _fake_save_annotation_json_store(
        curr_image: Path | None,
        predictions: dict,
        task_name: str | None,
        idx: int,
        save_path: Path,
        output_type: str,
        class_dict: dict,
        scale_factor: tuple[float, float],
        num_workers: int,
        *,
        verbose: bool,
    ) -> Path:
        """Capture save arguments."""
        _ = (
            task_name,
            output_type,
            class_dict,
            scale_factor,
            num_workers,
            verbose,
        )

        save_calls.append(
            {
                "curr_image": curr_image,
                "predictions": predictions,
                "idx": idx,
            },
        )

        return save_path.with_suffix(".db")

    def _fake_post_save_json_store(
        keys_to_compute: list[str],
        processed_predictions: dict,
        save_path: Path | None,
        **kwargs: object,
    ) -> None:
        """Record cleanup call."""
        nonlocal post_save_called

        _ = (
            keys_to_compute,
            processed_predictions,
            save_path,
            kwargs,
        )

        post_save_called = True

    def _fake_get_model_attr(
        attr_name: str,
    ) -> dict[int, str]:
        """Return fake model attributes."""
        _ = attr_name

        return {1: "Tumour"}

    monkeypatch.setattr(
        multi_task_segmentor,
        "_save_annotation_json_store",
        _fake_save_annotation_json_store,
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "_post_save_json_store",
        _fake_post_save_json_store,
    )

    monkeypatch.setattr(
        segmentor,
        "_get_model_attr",
        _fake_get_model_attr,
    )

    result = segmentor._save_predictions_as_json_store(
        processed_predictions={
            "canvas": np.array([1]),
            "count": np.array([1]),
            "probabilities": np.array([0.5]),
            "predictions": np.array([1]),
            "contours": np.array([1]),
        },
        task_name="nuclei",
        save_path=track_tmp_path / "output.db",
        output_type="annotationstore",
    )

    assert len(result) == 1

    assert post_save_called is True

    predictions = save_calls[0]["predictions"]

    assert "predictions" not in predictions
    assert "probabilities" not in predictions
    assert "canvas" not in predictions
    assert "count" not in predictions

    assert "contours" in predictions


def test_save_predictions_annotationstore_without_probabilities(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test AnnotationStore saving without probability outputs."""
    segmentor = MultiTaskSegmentor.__new__(MultiTaskSegmentor)

    segmentor.tasks = {"nuclei_segmentation"}

    segmentor.return_predictions_dict = {
        "nuclei_segmentation": True,
    }

    fake_zarr_group = {
        "nuclei_segmentation": {
            "contours": np.array([1]),
        },
    }

    def _fake_get_model_attr(name: str) -> dict:
        """Fake get_model_attr."""
        _ = name
        return {
            "nuclei_segmentation": {
                1: "Tumour",
            },
        }

    def _fake_save_predictions_as_dict_zarr(
        processed_predictions: dict,
        output_type: str,
        save_path: Path,
        **kwargs: object,
    ) -> Path:
        """Fake save_predictions_as_dict_zarr."""
        _ = processed_predictions, output_type, kwargs
        return save_path

    def _fake_save_predictions_as_json_store(
        processed_predictions: dict,
        task_name: str | None,
        save_path: Path,
        output_type: str,
        **kwargs: object,
    ) -> list:
        """Fake save_predictions_as_json_store."""
        _ = (
            processed_predictions,
            task_name,
            output_type,
            kwargs,
        )

        return [save_path.with_suffix(".db")]

    monkeypatch.setattr(
        segmentor,
        "_get_model_attr",
        _fake_get_model_attr,
    )

    monkeypatch.setattr(
        segmentor,
        "_save_predictions_as_dict_zarr",
        _fake_save_predictions_as_dict_zarr,
    )

    monkeypatch.setattr(
        segmentor,
        "_save_predictions_as_json_store",
        _fake_save_predictions_as_json_store,
    )

    def _fake_is_zarr(path: Path) -> bool:
        """Return False for all paths."""
        _ = path

        return False

    monkeypatch.setattr(
        multi_task_segmentor,
        "is_zarr",
        _fake_is_zarr,
    )

    def _fake_zarr_open(
        store: str | Path,
        mode: str = "r+",
    ) -> dict:
        """Return fake zarr group."""
        _ = store, mode

        return fake_zarr_group

    monkeypatch.setattr(
        zarr,
        "open",
        _fake_zarr_open,
    )

    result = segmentor.save_predictions(
        processed_predictions={},
        output_type="annotationstore",
        save_path=track_tmp_path / "output.db",
        return_probabilities=False,
        return_predictions=(False,),
    )

    assert result == [
        track_tmp_path / "output.db",
    ]


def test_save_predictions_annotationstore_path_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test annotationstore saving for multitask outputs."""
    segmentor = MultiTaskSegmentor.__new__(MultiTaskSegmentor)

    segmentor.tasks = {"semantic", "instance"}

    segmentor.return_predictions_dict = {
        "semantic": False,
        "instance": True,
    }

    segmentor.model = SimpleNamespace(
        class_dict={
            "semantic": {"0": "bg"},
            "instance": {"0": "cell"},
        },
    )

    zarr_path = tmp_path / "predictions.zarr"

    def _fake_save_predictions_as_dict_zarr(
        *_args: object,
        **_kwargs: object,
    ) -> Path:
        return zarr_path

    monkeypatch.setattr(
        segmentor,
        "_save_predictions_as_dict_zarr",
        _fake_save_predictions_as_dict_zarr,
    )

    opened_zarr = {
        "semantic": {"predictions": np.array([1])},
        "instance": {"predictions": np.array([2])},
    }

    def _fake_zarr_open(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        return opened_zarr

    monkeypatch.setattr(zarr, "open", _fake_zarr_open)

    saved_tasks: list[str] = []

    def _fake_save_predictions_as_json_store(
        processed_predictions: dict[str, object],
        task_name: str,
        save_path: Path,
        output_type: str,
        **kwargs: object,
    ) -> list:
        _ = processed_predictions, output_type, kwargs
        saved_tasks.append(task_name)
        return [save_path.with_name(f"{task_name}.db")]

    monkeypatch.setattr(
        segmentor,
        "_save_predictions_as_json_store",
        _fake_save_predictions_as_json_store,
    )

    result = segmentor.save_predictions(
        processed_predictions={"ignored": True},
        output_type="annotationstore",
        save_path=tmp_path / "output",
        class_dict={
            "semantic": {"0": "bg"},
            "instance": {"0": "cell"},
        },
        return_probabilities=True,
    )

    assert set(saved_tasks) == {"semantic", "instance"}

    assert result[0] == zarr_path

    assert set(result[1:]) == {
        tmp_path / "semantic.db",
        tmp_path / "instance.db",
    }

    assert "semantic" not in opened_zarr
    assert "instance" in opened_zarr


def test_save_predictions_dict_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test dict output delegates to dict/zarr saving."""
    segmentor = MultiTaskSegmentor.__new__(MultiTaskSegmentor)

    expected = {"result": 1}

    def _fake_save_predictions_as_dict_zarr(
        **kwargs: object,
    ) -> dict[str, int]:
        _ = kwargs
        return expected

    monkeypatch.setattr(
        segmentor,
        "_save_predictions_as_dict_zarr",
        _fake_save_predictions_as_dict_zarr,
    )

    result = segmentor.save_predictions(
        processed_predictions={},
        output_type="dict",
    )

    assert result is expected


def test_merge_multitask_vertical_chunkwise_overlap(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test merging overlapping vertical chunks."""
    canvas = da.from_array(
        np.array(
            [
                [[1.0]],
                [[1.0]],
                [[2.0]],
                [[2.0]],
            ],
            dtype=np.float32,
        ),
        chunks=(2, 1, 1),
    )

    count = da.from_array(
        np.ones(
            (4, 1, 1),
            dtype=np.int32,
        ),
        chunks=(2, 1, 1),
    )

    captured: list[np.ndarray] = []

    def _fake_clip_probabilities_to_shape(
        probabilities: np.ndarray,
        output_shape: tuple[int, int] | None,
        written_height: int,
    ) -> tuple[np.ndarray, int, bool]:
        _ = output_shape

        captured.append(probabilities.copy())

        return (
            probabilities,
            written_height + probabilities.shape[0],
            False,
        )

    def _fake_store_probabilities(
        probabilities: np.ndarray,
        chunk_shape: tuple[int, ...],
        probabilities_zarr: object,
        probabilities_da: object,
        zarr_group: object,
        name: str,
    ) -> tuple[None, np.ndarray]:
        """Keep probabilities in memory."""
        _ = (
            chunk_shape,
            probabilities_zarr,
            probabilities_da,
            zarr_group,
            name,
        )

        return None, probabilities

    def _fake_save_multitask_vertical_to_cache(**kwargs: object) -> tuple:
        """Do not spill to disk."""
        return (
            kwargs["probabilities_zarr"],
            kwargs["probabilities_da"],
            kwargs["zarr_group"],
        )

    def _fake_clear_zarr(
        probabilities_zarr: object,
        probabilities_da: object,
        zarr_group: object,
        idx: int,
        chunk_shape: tuple[int, ...],
        probabilities_shape: tuple[int, ...],
    ) -> object:
        """Return probabilities unchanged."""
        _ = (
            probabilities_zarr,
            zarr_group,
            idx,
            chunk_shape,
            probabilities_shape,
        )

        return probabilities_da

    monkeypatch.setattr(
        multi_task_segmentor,
        "clip_probabilities_to_shape",
        _fake_clip_probabilities_to_shape,
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "store_probabilities",
        _fake_store_probabilities,
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "_save_multitask_vertical_to_cache",
        _fake_save_multitask_vertical_to_cache,
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "_clear_zarr",
        _fake_clear_zarr,
    )

    output_locs_y_ = np.array(
        [
            [0, 2],
            [1, 3],
        ],
    )

    result = multi_task_segmentor.merge_multitask_vertical_chunkwise(
        canvas=[canvas],
        count=[count],
        output_locs_y_=output_locs_y_,
        zarr_group=None,
        save_path=track_tmp_path / "cache.zarr",
        verbose=False,
    )

    assert len(result) == 1
    assert len(captured) >= 1
    assert np.any(captured[0] == 1.5)


def test_save_multitask_to_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test spilling multitask outputs to cache."""
    canvas = [
        "canvas_0",
        "canvas_1",
    ]

    count = [
        "count_0",
        "count_1",
    ]

    canvas_zarr = [None, None]
    count_zarr = [None, None]

    calls: list[tuple[object, object]] = []

    def _fake_save_to_cache(
        canvas: object,
        count: object,
        canvas_zarr: object,
        count_zarr: object,
        save_path: str | Path,
        zarr_dataset_name: tuple[str, str],
        *,
        verbose: bool,
    ) -> tuple[str, str]:
        """Return synthetic cached arrays."""
        _ = canvas_zarr, count_zarr, save_path, verbose

        calls.append((canvas, count))

        return (
            f"{zarr_dataset_name[0]}_cached",
            f"{zarr_dataset_name[1]}_cached",
        )

    monkeypatch.setattr(
        multi_task_segmentor,
        "save_to_cache",
        _fake_save_to_cache,
    )

    returned_canvas_zarr, returned_count_zarr = (
        multi_task_segmentor.save_multitask_to_cache(
            canvas=canvas,
            count=count,
            canvas_zarr=canvas_zarr,
            count_zarr=count_zarr,
            verbose=False,
        )
    )

    assert calls == [
        ("canvas_0", "count_0"),
        ("canvas_1", "count_1"),
    ]

    assert returned_canvas_zarr == [
        "canvas/0_cached",
        "canvas/1_cached",
    ]

    assert returned_count_zarr == [
        "count/0_cached",
        "count/1_cached",
    ]


def test_save_multitask_vertical_to_cache_existing_zarr_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test caching with an existing Zarr group."""

    class FakeMemory:
        """Fake memory information."""

        available = 1

    class FakeArray:
        """Minimal Dask-like array."""

        nbytes = 1000
        shape = (4, 4)

        def compute(self) -> np.ndarray:
            """Return computed array."""
            _ = self
            return np.ones((4, 4), dtype=np.float32)

    class FakeZarrArray:
        """Minimal Zarr array."""

        def __setitem__(
            self,
            key: slice,
            value: np.ndarray,
        ) -> None:
            _ = key, value

    class FakeZarrGroup:
        """Minimal Zarr group."""

        def create_array(
            self,
            name: str,
            shape: tuple[int, ...],
            chunks: tuple[int, ...],
            dtype: np.dtype,
            *,
            overwrite: bool,
        ) -> FakeZarrArray:
            _ = self, name, shape, chunks, dtype, overwrite

            return FakeZarrArray()

    def _fake_virtual_memory() -> FakeMemory:
        """Return fake memory statistics."""
        return FakeMemory()

    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        _fake_virtual_memory,
    )

    def _fake_update_tqdm_desc(
        tqdm_loop: object,
        desc: str,
    ) -> None:
        """Mock progress-bar description update."""
        _ = tqdm_loop, desc

    monkeypatch.setattr(
        multi_task_segmentor,
        "update_tqdm_desc",
        _fake_update_tqdm_desc,
    )

    probabilities_zarr = [None]
    probabilities_da = [FakeArray()]
    existing_group = FakeZarrGroup()

    returned_zarr, returned_da, returned_group = _save_multitask_vertical_to_cache(
        probabilities_zarr=probabilities_zarr,
        probabilities_da=probabilities_da,
        zarr_group=existing_group,
        probabilities=np.ones((4, 4), dtype=np.float32),
        idx=0,
        tqdm_loop=SimpleNamespace(desc="test"),
        save_path=Path("cache.zarr"),
        chunk_shape=(2,),
        memory_threshold=80,
    )

    assert returned_group is existing_group
    assert returned_zarr[0] is not None
    assert returned_da[0] is None


def test_clear_zarr_removes_canvas_and_count_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test cached canvas/count arrays are removed from Zarr."""
    probabilities_zarr = object()

    canvas_group = {
        "0": "canvas_data",
    }

    count_group = {
        "0": "count_data",
    }

    zarr_group = {
        "canvas": canvas_group,
        "count": count_group,
    }

    captured: dict[str, object] = {}

    def _fake_from_zarr(
        array: object,
        chunks: tuple[int, ...],
    ) -> str:
        """Return synthetic dask array."""
        captured["array"] = array
        captured["chunks"] = chunks

        return "fake_dask_array"

    monkeypatch.setattr(
        da,
        "from_zarr",
        _fake_from_zarr,
    )

    result = _clear_zarr(
        probabilities_zarr=probabilities_zarr,
        probabilities_da=None,
        zarr_group=zarr_group,
        idx=0,
        chunk_shape=(128,),
        probabilities_shape=(256, 3),
    )

    assert result == "fake_dask_array"

    assert "0" not in canvas_group
    assert "0" not in count_group

    assert captured["array"] is probabilities_zarr
    assert captured["chunks"] == (128, 256, 3)


def test_calculate_probabilities_with_zarr_cache(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test probability calculation when cached Zarr arrays exist."""
    created_dask_arrays = []
    merge_kwargs: dict[str, object] = {}

    class FakeZarrArray:
        """Minimal Zarr array stand-in."""

        chunks = (4, 4)

        store = SimpleNamespace(
            root=str(track_tmp_path / "cache.zarr"),
        )

    fake_canvas_zarr = [FakeZarrArray()]
    fake_count_zarr = [FakeZarrArray()]

    def _fake_save_multitask_to_cache(
        canvas: list,
        count: list,
        canvas_zarr: list,
        count_zarr: list,
        *,
        verbose: bool,
        save_path: Path | None = None,
    ) -> tuple[list, list]:
        """Return cached arrays unchanged."""
        _ = canvas, count, verbose, save_path

        return canvas_zarr, count_zarr

    def _fake_from_zarr(
        array: FakeZarrArray,
        chunks: tuple[int, int],
    ) -> str:
        """Return synthetic Dask array."""
        created_dask_arrays.append((array, chunks))
        return "fake_dask_array"

    def _fake_zarr_open(
        path: str,
        mode: str = "a",
    ) -> str:
        """Return fake Zarr group."""
        _ = mode

        assert path == str(track_tmp_path / "cache.zarr")

        return "fake_zarr_group"

    def _fake_merge_multitask_vertical_chunkwise(
        *,
        canvas: list,
        count: list,
        output_locs_y_: np.ndarray,
        zarr_group: object,
        save_path: Path,
        memory_threshold: int,
        output_shape: tuple[int, int],
    ) -> list:
        """Capture arguments and return synthetic output."""
        _ = output_locs_y_, save_path, memory_threshold, output_shape
        merge_kwargs["canvas"] = canvas
        merge_kwargs["count"] = count
        merge_kwargs["zarr_group"] = zarr_group

        return ["probabilities"]

    monkeypatch.setattr(
        multi_task_segmentor,
        "save_multitask_to_cache",
        _fake_save_multitask_to_cache,
    )

    monkeypatch.setattr(
        da,
        "from_zarr",
        _fake_from_zarr,
    )

    monkeypatch.setattr(
        zarr,
        "open",
        _fake_zarr_open,
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "merge_multitask_vertical_chunkwise",
        _fake_merge_multitask_vertical_chunkwise,
    )

    result = multi_task_segmentor._calculate_probabilities(
        canvas_zarr=fake_canvas_zarr,
        count_zarr=fake_count_zarr,
        canvas=[None],
        count=[None],
        output_locs_y_=np.array([0, 100]),
        save_path=track_tmp_path / "output.zarr",
        memory_threshold=80,
        output_shape=(100, 100),
        verbose=False,
    )

    assert result == ["probabilities"]

    assert len(created_dask_arrays) == 2

    assert merge_kwargs["canvas"] == ["fake_dask_array"]
    assert merge_kwargs["count"] == ["fake_dask_array"]
    assert merge_kwargs["zarr_group"] == "fake_zarr_group"


def test_check_and_update_for_memory_overload(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test spilling intermediate results to disk when memory threshold is exceeded."""

    class FakeMemory:
        """Fake virtual memory statistics."""

        percent = 90
        available = 1024 * 1024 * 1024

    fake_canvas = [
        SimpleNamespace(nbytes=1024),
    ]
    fake_count = [
        SimpleNamespace(),
    ]

    fake_canvas_zarr = [None]
    fake_count_zarr = [None]

    captured: dict[str, object] = {}

    def _fake_update_tqdm_desc(
        tqdm_loop: object,
        desc: str,
    ) -> None:
        """Capture progress-bar descriptions."""
        _ = tqdm_loop
        captured.setdefault("descriptions", []).append(desc)

    def _fake_save_multitask_to_cache(
        canvas: list,
        count: list,
        canvas_zarr: list,
        count_zarr: list,
        save_path: Path,
        *,
        verbose: bool = True,
    ) -> tuple[list, list]:
        """Return synthetic cached arrays."""
        _ = canvas, count, save_path, verbose

        captured["cache_called"] = True

        return canvas_zarr, count_zarr

    def _fake_virtual_memory() -> FakeMemory:
        """Return fake memory statistics."""
        return FakeMemory()

    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        _fake_virtual_memory,
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "update_tqdm_desc",
        _fake_update_tqdm_desc,
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "save_multitask_to_cache",
        _fake_save_multitask_to_cache,
    )

    canvas, count, _, _, _ = multi_task_segmentor._check_and_update_for_memory_overload(
        canvas=fake_canvas,
        count=fake_count,
        canvas_zarr=fake_canvas_zarr,
        count_zarr=fake_count_zarr,
        memory_threshold=80,
        tqdm_loop=object(),
        save_path=track_tmp_path / "cache.zarr",
        num_expected_output=1,
        verbose=False,
    )

    assert captured["cache_called"] is True

    assert canvas == [None]
    assert count == [None]

    assert len(captured["descriptions"]) == 2
    assert "Current Memory usage" in captured["descriptions"][0]
    assert captured["descriptions"][1] == "Inferring patches"


def test_save_annotation_json_store_without_image_path(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test save annotation store when curr_image is not a Path."""
    captured: dict[str, object] = {}

    def _fake_dict_to_json_store(
        processed_predictions: dict,
        output_path: Path,
        output_type: str,
        class_dict: dict | None = None,
        origin: tuple[float, float] = (0, 0),
        scale_factor: tuple[float, float] = (1, 1),
        num_workers: int = 0,
        *,
        verbose: bool = True,
    ) -> Path:
        """Capture arguments and return output path."""
        captured["processed_predictions"] = processed_predictions
        captured["output_path"] = output_path
        captured["output_type"] = output_type
        captured["class_dict"] = class_dict
        captured["origin"] = origin
        captured["scale_factor"] = scale_factor
        captured["num_workers"] = num_workers
        captured["verbose"] = verbose

        return output_path

    monkeypatch.setattr(
        multi_task_segmentor,
        "dict_to_json_store",
        _fake_dict_to_json_store,
    )

    predictions = {
        "coordinates": np.array([[0, 0]]),
        "type": np.array([1]),
    }

    result = multi_task_segmentor._save_annotation_json_store(
        curr_image=None,
        predictions=predictions,
        task_name="nuclei",
        idx=5,
        save_path=track_tmp_path / "output.db",
        output_type="annotationstore",
        class_dict={1: "tumour"},
        scale_factor=(1.0, 1.0),
        num_workers=1,
        verbose=False,
    )

    expected_path = track_tmp_path / "5_nuclei.db"

    assert result == expected_path
    assert captured["output_path"] == expected_path

    # Covers removal of coordinates field
    assert "coordinates" not in captured["processed_predictions"]


def test_update_tile_based_predictions_array_updates_predictions() -> None:
    """Test updating predictions when prediction array is present.

    Covers:

        if wsi_info_dict[idx]["predictions"] is None:

    evaluating to False.

    """
    post_process_output = (
        {
            "predictions": np.array(
                [
                    [1, 2],
                    [3, 4],
                ],
                dtype=np.uint8,
            ),
            "seg_type": "semantic",
        },
    )

    wsi_info_dict = (
        {
            "predictions": np.zeros(
                (4, 4),
                dtype=np.uint8,
            ),
        },
    )

    updated_wsi_info_dict, max_inst_value = (
        multi_task_segmentor._update_tile_based_predictions_array(
            post_process_output=post_process_output,
            wsi_info_dict=wsi_info_dict,
            bounds=(0, 0, 2, 2),
            offset=(0, 0),
        )
    )

    expected = np.array(
        [
            [1, 2, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(
        updated_wsi_info_dict[0]["predictions"],
        expected,
    )

    assert max_inst_value is None


def test_process_instance_predictions_tile_mode_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test processing predictions for tile_mode 3."""
    new_inst_dict = {
        "new_uuid": {
            "box": np.array([0, 0, 10, 10]),
        },
    }

    def _fake_get_sel_indices_margin_lines(
        ioconfig: IOSegmentorConfig,
        tile_shape: tuple[int, int],
        inst_dict: dict,
        tile_tl: tuple[int, int],
        tile_mode: int,
        tile_flag: tuple[int, int, int, int],
    ) -> tuple[list[int], list[str]]:
        """Return synthetic indices and margin lines."""
        _ = (
            ioconfig,
            tile_shape,
            inst_dict,
            tile_tl,
            tile_mode,
            tile_flag,
        )

        return [0], ["margin_line"]

    call_count = 0

    def _fake_retrieve_sel_uids(
        sel_indices_: list[int],
        inst_dict_: dict,
    ) -> list:
        """Return different values for successive calls."""
        nonlocal call_count
        _ = sel_indices_, inst_dict_

        call_count += 1

        if call_count == 1:
            return ["remove_from_tile"]

        return ["remove_from_original"]

    new_inst_dict = {
        "new_uuid": {
            "box": np.array([0, 0, 10, 10]),
        },
    }

    def _fake_move_tile_space_to_wsi_space(
        inst_dict: dict,
        tile_tl: tuple[int, int],
        remove_insts_in_tile: list[str],
    ) -> dict:
        """Return synthetic merged instances."""
        _ = inst_dict, tile_tl

        assert remove_insts_in_tile == ["remove_from_tile"]

        return new_inst_dict

    class FakeSTRtree:
        """Minimal STRtree stand-in."""

        def query(self, bounds: object) -> list:
            """Return one matching geometry."""
            _ = self, bounds
            return [0]

    monkeypatch.setattr(
        multi_task_segmentor,
        "_get_sel_indices_margin_lines",
        _fake_get_sel_indices_margin_lines,
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "retrieve_sel_uids",
        _fake_retrieve_sel_uids,
    )

    monkeypatch.setattr(
        multi_task_segmentor,
        "_move_tile_space_to_wsi_space",
        _fake_move_tile_space_to_wsi_space,
    )

    result = multi_task_segmentor._process_instance_predictions(
        inst_dict={1: {"box": np.array([0, 0, 10, 10])}},
        ioconfig=SimpleNamespace(margin=None),
        tile_shape=(256, 256),
        tile_flag=(0, 0, 0, 0),
        tile_mode=3,
        tile_tl=(0, 0),
        ref_inst_dict={"ref_uuid": {}},
        ref_inst_rtree=FakeSTRtree(),
    )

    assert result == (
        new_inst_dict,
        ["remove_from_original"],
    )


def test_retrieve_sel_uids_empty_indices() -> None:
    """Test retrieve_sel_uids with no selected indices."""
    result = multi_task_segmentor.retrieve_sel_uids(
        sel_indices_=[],
        inst_dict_={
            "uid_1": {},
            "uid_2": {},
        },
    )

    assert result == []


def test_get_sel_indices_margin_lines_invalid_tile_mode() -> None:
    """Test invalid tile mode raises ValueError."""
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        patch_input_shape=(256, 256),
        patch_output_shape=(256, 256),
        stride_shape=(128, 128),
    )

    with pytest.raises(
        ValueError,
        match="Unknown tile mode 4",
    ):
        _get_sel_indices_margin_lines(
            ioconfig=ioconfig,
            tile_shape=(256, 256),
            tile_flag=(0, 0, 0, 0),
            tile_mode=4,
            tile_tl=(0, 0),
            inst_dict={},
        )


def test_update_tile_based_predictions_array_instance_overlap() -> None:
    """Test merging instance predictions with overlapping nuclei."""
    existing_predictions = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )

    new_predictions = np.array(
        [
            [2, 2],
            [2, 2],
        ],
        dtype=np.int32,
    )

    post_process_output = (
        {
            "predictions": new_predictions,
            "seg_type": "instance",
        },
    )

    wsi_info_dict = (
        {
            "predictions": existing_predictions.copy(),
        },
    )

    updated_wsi_info_dict, max_inst_value = (
        multi_task_segmentor._update_tile_based_predictions_array(
            post_process_output=post_process_output,
            wsi_info_dict=wsi_info_dict,
            bounds=(0, 0, 2, 2),
            offset=(0, 0),
            max_inst_value=10,
        )
    )

    np.testing.assert_array_equal(
        updated_wsi_info_dict[0]["predictions"],
        existing_predictions,
    )

    assert max_inst_value == 10


def test_update_tile_based_predictions_array_instance_no_overlap() -> None:
    """Test merging instance predictions when there is no overlap."""
    existing_predictions = np.zeros(
        (3, 3),
        dtype=np.int32,
    )

    new_predictions = np.array(
        [
            [1, 1],
            [1, 1],
        ],
        dtype=np.int32,
    )

    post_process_output = (
        {
            "predictions": new_predictions,
            "seg_type": "instance",
        },
    )

    wsi_info_dict = (
        {
            "predictions": existing_predictions.copy(),
        },
    )

    updated_wsi_info_dict, max_inst_value = (
        multi_task_segmentor._update_tile_based_predictions_array(
            post_process_output=post_process_output,
            wsi_info_dict=wsi_info_dict,
            bounds=(0, 0, 2, 2),
            offset=(0, 0),
            max_inst_value=10,
        )
    )

    expected = np.array(
        [
            [11, 11, 0],
            [11, 11, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(
        updated_wsi_info_dict[0]["predictions"],
        expected,
    )

    assert max_inst_value == 21


def test_compute_info_dict_for_merge_tile_mode_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test STRtree construction when tile_mode is 3."""
    inst_dict: dict[int, dict[str, object]] = {}

    ref_inst_info_dict = {
        1: {
            "box": np.array(
                [0, 0, 10, 10],
                dtype=np.int32,
            ),
        },
        2: {
            "box": np.array(
                [20, 20, 30, 30],
                dtype=np.int32,
            ),
        },
    }

    captured: dict[str, object] = {}

    def _fake_process_instance_predictions(
        inst_dict: dict,
        ioconfig: IOSegmentorConfig,
        tile_shape: tuple[int, int],
        tile_flag: tuple[int, int, int, int],
        tile_mode: int,
        tile_tl: tuple[int, int],
        ref_inst_dict: dict,
        ref_inst_rtree: STRtree,
    ) -> tuple[dict, list]:
        """Capture arguments passed into the merge helper."""
        _ = ioconfig, tile_shape, tile_flag, tile_tl
        captured["inst_dict"] = inst_dict
        captured["tile_mode"] = tile_mode
        captured["ref_inst_dict"] = ref_inst_dict
        captured["ref_inst_rtree"] = ref_inst_rtree

        return {}, []

    monkeypatch.setattr(
        "tiatoolbox.models.engine.multi_task_segmentor._process_instance_predictions",
        _fake_process_instance_predictions,
    )

    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        patch_input_shape=(256, 256),
        patch_output_shape=(256, 256),
        stride_shape=(128, 128),
    )

    result = multi_task_segmentor._compute_info_dict_for_merge(
        inst_dict=inst_dict,
        tile_mode=3,
        ref_inst_info_dict=ref_inst_info_dict,
        ioconfig=ioconfig,
        tile_shape=(256, 256),
        tile_tl=(0, 0),
        tile_flag=(0, 0, 0, 0),
    )

    assert result == ({}, [])

    assert captured["tile_mode"] == 3
    assert captured["ref_inst_dict"] == ref_inst_info_dict

    ref_inst_rtree = captured["ref_inst_rtree"]
    assert isinstance(ref_inst_rtree, STRtree)

    # Verify two geometries were inserted into the tree.
    assert len(ref_inst_rtree.geometries) == 2


def test_create_wsi_info_dict_with_return_predictions(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test _create_wsi_info_dict when return_predictions is provided."""
    fake_array = np.zeros((8, 8), dtype=np.uint8)

    post_process_output = (
        {
            "task_type": "nuclei_segmentation",
            "predictions": fake_array,
        },
    )

    created_arrays: list[dict[str, object]] = []

    def _fake_create_smart_array(
        shape: tuple[int, ...],
        dtype: np.dtype | str,
        memory_threshold: float,
        name: str,
        zarr_path: str | Path,
        chunks: tuple[int, ...] | str = "auto",
    ) -> np.ndarray:
        """Record array creation requests."""
        created_arrays.append(
            {
                "shape": shape,
                "dtype": dtype,
                "memory_threshold": memory_threshold,
                "name": name,
                "zarr_path": zarr_path,
                "chunks": chunks,
            },
        )

        return np.zeros(shape, dtype=dtype)

    monkeypatch.setattr(
        "tiatoolbox.models.engine.multi_task_segmentor.create_smart_array",
        _fake_create_smart_array,
    )

    result = multi_task_segmentor._create_wsi_info_dict(
        post_process_output=post_process_output,
        wsi_info_dict=None,
        wsi_proc_shape=(16, 32),
        save_path=track_tmp_path / "output.zarr",
        return_predictions=(True,),
        memory_threshold=80,
    )

    assert len(result) == 1

    assert result[0]["task_type"] == "nuclei_segmentation"

    assert isinstance(
        result[0]["predictions"],
        np.ndarray,
    )

    assert result[0]["predictions"].shape == (32, 16)

    assert result[0]["info_dict"] == {}

    assert len(created_arrays) == 1

    assert created_arrays[0]["name"] == "nuclei_segmentation/predictions"


def test_multitask_segmentor_cli_uses_yaml_config(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test that the multitask-segmentor CLI forwards the prepared IO config."""
    img_input = track_tmp_path / "input"
    img_input.mkdir()

    sample_image = img_input / "sample.png"
    sample_image.write_bytes(b"image")

    yaml_path = track_tmp_path / "config.yaml"
    yaml_path.write_text(
        "patch_input_shape: [224, 224]\n",
        encoding="utf-8",
    )

    fake_ioconfig = object()
    run_calls: list[dict[str, object]] = []

    def _fake_prepare_model_cli(
        img_input: str | Path,
        output_path: str | Path,
        masks: str | Path | None,
        file_types: str,
    ) -> tuple[list[Path], list[Path] | None, Path]:
        """Fake prepare model cli."""
        _ = img_input, masks, file_types
        return [sample_image], None, Path(output_path)

    def _fake_prepare_ioconfig(
        config_class: type[object],
        pretrained_weights: str | Path | None,
        yaml_config_path: str | Path,
    ) -> object:
        """Fake prepare ioconfig."""
        _ = config_class, pretrained_weights

        assert Path(yaml_config_path) == yaml_path

        return fake_ioconfig

    class _FakeMultiTaskSegmentor:
        """Minimal stand-in for MultiTaskSegmentor."""

        def __init__(
            self,
            model: str,
            weights: str | None,
            batch_size: int,
            num_workers: int,
            *,
            verbose: bool,
        ) -> None:
            _ = model, weights, batch_size, num_workers, verbose

        def run(self, **kwargs: object) -> None:
            """Record run arguments."""
            _ = self  # fix PYL-R0201
            run_calls.append(kwargs)

    # Patch where the CLI command uses these names.

    cli_module = importlib.import_module(
        "tiatoolbox.cli.multitask_segmentor",
    )

    monkeypatch.setattr(
        cli_module,
        "prepare_model_cli",
        _fake_prepare_model_cli,
    )

    monkeypatch.setattr(
        cli_module,
        "prepare_ioconfig",
        _fake_prepare_ioconfig,
    )

    monkeypatch.setattr(
        "tiatoolbox.models.MultiTaskSegmentor",
        _FakeMultiTaskSegmentor,
    )

    runner = CliRunner()

    result = runner.invoke(
        cli.main,
        [
            "multitask-segmentor",
            "--img-input",
            str(img_input),
            "--output-path",
            str(track_tmp_path / "output"),
            "--yaml-config-path",
            str(yaml_path),
        ],
    )

    assert result.exit_code == 0

    assert len(run_calls) == 1
    assert run_calls[0]["ioconfig"] is fake_ioconfig


def test_compute_qupath_json_with_explicit_class_dict(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test compute_qupath_json when class_dict is explicitly provided."""
    store = DaskDelayedJSONStore.__new__(DaskDelayedJSONStore)

    store._contours = [
        np.array(
            [
                [0, 0],
                [10, 0],
                [10, 10],
            ],
            dtype=float,
        ),
    ]

    store._processed_predictions = {
        "type": np.array([1], dtype=object),
    }

    class_dict = {
        0: "background",
        1: "tumour",
    }

    def _fake_build_single_qupath_feature(
        i: int,
        class_dict_in: dict[int, str],
        origin: tuple[float, float],
        scale_factor: tuple[float, float],
        class_colors: dict[int, list[int]],
    ) -> dict[str, object]:
        """Return a simple fake feature."""
        _ = i, origin, scale_factor

        assert class_dict_in is class_dict
        assert 0 in class_colors
        assert 1 in class_colors

        return {
            "type": "Feature",
            "properties": {
                "type": "tumour",
            },
        }

    def _fake_save_qupath_json(
        save_path: Path | None,
        qupath_json: dict[str, object],
    ) -> dict[str, object]:
        """Return JSON rather than touching disk."""
        assert save_path == track_tmp_path / "output.json"
        return qupath_json

    monkeypatch.setattr(
        store,
        "_build_single_qupath_feature",
        _fake_build_single_qupath_feature,
    )

    monkeypatch.setattr(
        "tiatoolbox.models.engine.multi_task_segmentor.save_qupath_json",
        _fake_save_qupath_json,
    )

    result = store.compute_qupath_json(
        class_dict=class_dict,
        save_path=track_tmp_path / "output.json",
        verbose=False,
    )

    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 1


def test_apply_coordinate_offset_zero_offset_returns_input() -> None:
    """Test early return when offset is zero."""
    data_array = np.array(
        [
            np.array([1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.int32),
        ],
        dtype=object,
    )

    result = multi_task_segmentor.apply_coordinate_offset(
        data_array=data_array,
        offset=(0, 0),
        key="centroid",
        verbose=False,
    )

    np.testing.assert_array_equal(result, data_array)
    assert result is data_array


def test_build_single_qupath_feature_non_type_property() -> None:
    """Test non-type properties are written to QuPath feature properties."""
    store = DaskDelayedJSONStore.__new__(DaskDelayedJSONStore)

    store._contours = [
        np.array(
            [
                [0, 0],
                [10, 0],
                [10, 10],
            ],
            dtype=float,
        ),
    ]

    store._processed_predictions = {
        "type": np.array([1], dtype=object),
        "prob": np.array([0.95], dtype=np.float32),
    }

    result = store._build_single_qupath_feature(
        i=0,
        class_dict={1: "Tumour"},
        origin=(0.0, 0.0),
        scale_factor=(1.0, 1.0),
        class_colors={1: [255, 0, 0]},
    )

    props = result["properties"]

    assert props["type"] == "Tumour"
    assert np.isclose(props["prob"], 0.95)

    assert props["classification"]["name"] == "Tumour"
    assert props["class_value"] == 1


def test_post_save_json_store_removes_keys() -> None:
    """Test removal of computed prediction keys."""
    processed_predictions = {
        "probabilities": np.array([1]),
        "predictions": np.array([2]),
    }

    _post_save_json_store(
        keys_to_compute=["probabilities"],
        processed_predictions=processed_predictions,
        save_path=None,
    )

    assert "probabilities" not in processed_predictions
    assert "predictions" in processed_predictions


def test_post_save_json_store_logs_probability_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test probability-map warning message is logged."""
    with caplog.at_level(logging.INFO):
        _post_save_json_store(
            keys_to_compute=[],
            processed_predictions={},
            save_path=Path("output.zarr"),
            return_probabilities=True,
        )

    assert "Probability maps cannot be saved as AnnotationStore or JSON" in caplog.text


def test_post_save_json_store_removes_empty_root_store(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test deletion of an empty root Zarr store."""
    root_path = track_tmp_path / "root.zarr"

    root = zarr.open_group(root_path, mode="w")

    class GroupProxy:
        """Minimal object behaving like a zarr.Group."""

        def __init__(self, group: zarr.Group) -> None:
            self._group = group
            self.store = group.store
            self.path = "nested"

        @property
        def __class__(self) -> type[zarr.Group]:
            return zarr.Group

        def keys(self) -> list[str]:
            """Return a non-empty key list."""
            _ = self
            return ["dummy"]

    removed_paths: list[Path | str] = []

    def _fake_rmtree(
        path: Path | str,
        *,
        ignore_errors: bool,
    ) -> None:
        """Record removed paths."""
        _ = ignore_errors
        removed_paths.append(path)

    class FakeEmptyStore:
        """Fake root store containing no datasets."""

        def keys(self) -> list:
            """Return an empty key list."""
            _ = self
            return []

    def _fake_open(
        store_root: Path | str,
        mode: str = "r",
    ) -> FakeEmptyStore:
        """Return an empty Zarr store."""
        _ = store_root, mode
        return FakeEmptyStore()

    monkeypatch.setattr(
        shutil,
        "rmtree",
        _fake_rmtree,
    )

    monkeypatch.setattr(
        zarr,
        "open",
        _fake_open,
    )

    _post_save_json_store(
        keys_to_compute=[],
        processed_predictions=GroupProxy(root),
        save_path=None,
    )

    assert root_path in removed_paths


def test_post_save_json_store_keeps_non_empty_root_store(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test that a non-empty root store is not deleted."""
    root_path = track_tmp_path / "root.zarr"

    root = zarr.open_group(root_path, mode="w")

    class GroupProxy:
        """Minimal object behaving like a zarr.Group."""

        def __init__(self, group: zarr.Group) -> None:
            self.store = group.store
            self.path = "nested"

        @property
        def __class__(self) -> type[zarr.Group]:
            return zarr.Group

        def keys(self) -> list:
            """Return a non-empty list.

            Prevent the first cleanup branch from executing.

            """
            _ = self
            return ["dummy"]

    removed_paths: list[Path | str] = []

    def _fake_rmtree(
        path: Path | str,
        *,
        ignore_errors: bool,
    ) -> None:
        """Record removed paths."""
        _ = ignore_errors
        removed_paths.append(path)

    class FakeNonEmptyStore:
        """Fake root store containing at least one dataset."""

        def keys(self) -> list:
            """Return a non-empty key list."""
            _ = self
            return ["predictions"]

    def _fake_open(
        store_root: Path | str,
        mode: str = "r",
    ) -> FakeNonEmptyStore:
        """Return a non-empty Zarr store."""
        _ = store_root, mode
        return FakeNonEmptyStore()

    monkeypatch.setattr(
        shutil,
        "rmtree",
        _fake_rmtree,
    )

    monkeypatch.setattr(
        zarr,
        "open",
        _fake_open,
    )

    _post_save_json_store(
        keys_to_compute=[],
        processed_predictions=GroupProxy(root),
        save_path=None,
    )

    assert removed_paths == []


def test_move_tile_space_to_wsi_space_without_centroid() -> None:
    """Test moving instance data when centroid is absent."""
    contours = np.array(
        [
            [1, 1],
            [2, 2],
            [3, 3],
        ],
        dtype=np.int32,
    )

    inst_dict = {
        1: {
            "box": np.array(
                [0, 0, 10, 10],
                dtype=np.int32,
            ),
            "contours": contours.copy(),
        },
    }

    result = multi_task_segmentor._move_tile_space_to_wsi_space(
        inst_dict=inst_dict,
        tile_tl=np.array([0, 0]),
        remove_insts_in_tile=[],
    )

    assert len(result) == 1

    inst_info = next(iter(result.values()))

    np.testing.assert_array_equal(
        inst_info["box"],
        np.array(
            [0, 0, 10, 10],
            dtype=np.int32,
        ),
    )

    np.testing.assert_array_equal(
        inst_info["contours"],
        contours,
    )

    assert "centroid" not in inst_info


def test_dict_to_json_store_qupath(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test dict_to_json_store with QuPath output."""
    pad_value = np.iinfo(np.int32).min

    contours = np.array(
        [
            [
                [0, 0],
                [10, 0],
                [10, 10],
                [pad_value, pad_value],
            ],
        ],
        dtype=np.int32,
    )

    processed_predictions = {
        "contours": contours,
        "type": np.array([1], dtype=np.int32),
    }

    output_path = track_tmp_path / "output.db"

    expected_output = output_path.with_suffix(".json")

    called: dict[str, object] = {}

    class FakeDaskDelayedJSONStore:
        """Minimal stand-in for DaskDelayedJSONStore."""

        def __init__(
            self,
            contours: np.ndarray,
            processed_predictions: dict,
        ) -> None:
            called["contours"] = contours
            called["processed_predictions"] = processed_predictions

        def compute_qupath_json(
            self,
            class_dict: dict[int, str] | None,
            origin: tuple[float, float],
            scale_factor: tuple[float, float],
            batch_size: int,
            num_workers: int,
            *,
            verbose: bool,
            save_path: Path,
        ) -> Path:
            """Return the expected JSON path."""
            _ = self
            called["class_dict"] = class_dict
            called["origin"] = origin
            called["scale_factor"] = scale_factor
            called["batch_size"] = batch_size
            called["num_workers"] = num_workers
            called["verbose"] = verbose
            called["save_path"] = save_path

            return save_path

    monkeypatch.setattr(
        "tiatoolbox.models.engine.multi_task_segmentor.DaskDelayedJSONStore",
        FakeDaskDelayedJSONStore,
    )

    result = multi_task_segmentor.dict_to_json_store(
        processed_predictions=processed_predictions,
        output_path=output_path,
        output_type="qupath",
        class_dict={1: "tumour"},
        verbose=False,
    )

    assert result == expected_output

    assert called["save_path"] == expected_output
    assert called["batch_size"] == 100
    assert called["class_dict"] == {1: "tumour"}


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
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


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
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


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
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


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
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


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
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


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
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


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
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


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
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


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
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


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
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
    probabilities_zarr = root.create_array(
        "probabilities",
        data=np.zeros((5, 3, 3)),
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
    new_zarr, new_da, zarr_group = _save_multitask_vertical_to_cache(
        probabilities_zarr=probabilities_zarr,
        probabilities_da=probabilities_da,
        zarr_group=None,
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
    assert zarr_group is not None

    # Data was written correctly
    assert np.array_equal(new_zarr[idx][:], np.array([[1, 2, 3]]))


def test_multitask_vertical_merge_continues_after_zarr_spill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test multitask vertical merge appends all chunks after spilling to Zarr."""

    class FakeVM:
        """Fake psutil.virtual_memory() with extremely low available memory."""

        available = 1

    monkeypatch.setattr(psutil, "virtual_memory", FakeVM)

    values = np.arange(8 * 3, dtype=np.float32).reshape(8, 3, 1)
    canvas = [da.from_array(values, chunks=(2, 3, 1))]
    count = [da.from_array(np.ones_like(values), chunks=(2, 3, 1))]
    output_locs_y = np.array([[0, 2], [2, 4], [4, 6], [6, 8]])

    result = merge_multitask_vertical_chunkwise(
        canvas=canvas,
        count=count,
        output_locs_y_=output_locs_y,
        zarr_group=None,
        save_path=tmp_path / "vertical.zarr",
        memory_threshold=0,
        output_shape=(8, 3),
        verbose=False,
    )

    assert result[0].shape == values.shape
    assert np.array_equal(result[0].compute(), values)


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


@pytest.mark.skipif(
    _RUNNING_ON_CI,
    reason="Local test only.",
)
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


class DummyStoreSingle:
    """Minimal mock of DaskDelayedJSONStore for testing a single feature build."""

    _contours: list[np.ndarray]
    _processed_predictions: dict[str, list[Any]]

    def __init__(self) -> None:
        """Initialize DummyStoreSingle."""
        self._contours = [
            np.array([[0, 0], [10, 0], [10, 10]], dtype=float),
        ]
        self._processed_predictions = {
            "type": [None],
            "area": [None],
        }

    def _build_single_qupath_feature(
        self,
        i: int,
        class_dict: dict[int, str] | None,
        origin: tuple[float, float],
        scale_factor: tuple[float, float],
        class_colors: dict[int, Any],
    ) -> dict[str, Any]:
        """Call the real method using this dummy instance."""
        return DaskDelayedJSONStore._build_single_qupath_feature(
            self, i, class_dict, origin, scale_factor, class_colors
        )


def test_build_single_qupath_feature_type_none() -> None:
    """Test that None class values are handled correctly."""
    store = DummyStoreSingle()

    class_dict = {0: "background"}
    origin = (5.0, 5.0)
    scale_factor = (1.0, 1.0)
    class_colors = {0: "#FFFFFF"}

    result = store._build_single_qupath_feature(
        i=0,
        class_dict=class_dict,
        origin=origin,
        scale_factor=scale_factor,
        class_colors=class_colors,
    )

    props = result["properties"]

    assert props["type"] == "background"
    assert props["classification"]["name"] == "background"
    assert props["classification"]["color"] == "#FFFFFF"
    assert props["class_value"] == 0
    assert result["geometry"]["type"] == "Polygon"
    assert result["name"] == "background"


# ----------------------------------------------------------------------
# Monkeypatch fixture for compute_qupath_json
# ----------------------------------------------------------------------
@pytest.fixture
def patch_save_qupath_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch save_qupath_json so compute_qupath_json returns JSON directly."""

    def fake_save_qupath_json(
        save_path: Path | None,  # noqa: ARG001
        qupath_json: dict[str, Any],
    ) -> dict[str, Any]:
        return qupath_json

    monkeypatch.setattr(
        "tiatoolbox.models.engine.multi_task_segmentor.save_qupath_json",
        fake_save_qupath_json,
    )


# ----------------------------------------------------------------------
# Dummy store for compute_qupath_json
# ----------------------------------------------------------------------
class DummyStoreCompute:
    """Minimal mock of DaskDelayedJSONStore for testing compute_qupath_json."""

    _contours: list[np.ndarray]
    _processed_predictions: dict[str, list[Any]]

    def __init__(self) -> None:
        """Initialize DummyStoreCompute."""
        self._contours = [
            np.array([[0, 0], [10, 0], [10, 10]], dtype=float),
            np.array([[5, 5], [15, 5], [15, 15]], dtype=float),
        ]
        self._processed_predictions = {
            "type": [None, None],
        }

    # --- REQUIRED: compute_qupath_json calls this internally ---
    def _build_single_qupath_feature(
        self,
        i: int,
        class_dict: dict[int, str] | None,
        origin: tuple[float, float],
        scale_factor: tuple[float, float],
        class_colors: dict[int, Any],
    ) -> dict[str, Any]:
        return DaskDelayedJSONStore._build_single_qupath_feature(
            self, i, class_dict, origin, scale_factor, class_colors
        )

    def compute_qupath_json(
        self,
        class_dict: dict[int, str] | None,
        origin: tuple[float, float],
        scale_factor: tuple[float, float],
        save_path: Path | None,
        batch_size: int = 100,
        num_workers: int = 0,
        *,
        verbose: bool,
    ) -> dict[str, Any]:
        """Call the real compute_qupath_json using this dummy instance."""
        return DaskDelayedJSONStore.compute_qupath_json(
            self,
            class_dict=class_dict,
            origin=origin,
            scale_factor=scale_factor,
            save_path=save_path,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
        )


def test_compute_qupath_json_valid_ids_empty(
    patch_save_qupath_json: None,  # noqa: ARG001
) -> None:
    """Test fallback class_dict={0:0} when all type predictions are None."""
    store = DummyStoreCompute()

    result = store.compute_qupath_json(
        class_dict=None,
        origin=(0, 0),
        scale_factor=(1, 1),
        save_path=None,
        verbose=False,
    )

    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 2

    for feature in result["features"]:
        props = feature["properties"]
        assert props["type"] == 0
        assert props["classification"]["name"] == 0
        assert props["class_value"] == 0
