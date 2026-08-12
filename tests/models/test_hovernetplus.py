"""Unit test package for HoVerNet+."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from tiatoolbox.models import HoVerNetPlus
from tiatoolbox.models.architecture import fetch_pretrained_weights, hovernetplus
from tiatoolbox.utils import imread
from tiatoolbox.utils.misc import select_device
from tiatoolbox.utils.transforms import imresize


def test_functionality(remote_sample: Callable) -> None:
    """Functionality test."""
    sample_patch = str(remote_sample("stainnorm-source"))
    patch_pre = imread(sample_patch)
    patch_pre = imresize(patch_pre, scale_factor=0.5)
    patch = patch_pre[0:256, 0:256]
    batch = torch.from_numpy(patch)[None]

    # Test functionality with both nuclei and layer segmentation
    model = HoVerNetPlus(num_types=3, num_layers=5)
    # Test decoder as expected
    assert len(model.decoder["np"]) > 0, "Decoder must contain np branch."
    assert len(model.decoder["hv"]) > 0, "Decoder must contain hv branch."
    assert len(model.decoder["tp"]) > 0, "Decoder must contain tp branch."
    assert len(model.decoder["ls"]) > 0, "Decoder must contain ls branch."
    weights_path = fetch_pretrained_weights("hovernetplus-oed")
    pretrained = torch.load(weights_path)
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, device=select_device(on_gpu=False))
    assert len(output) == 4, "Must contain predictions for: np, hv, tp and ls branches."
    output = [v[0] for v in output]
    output = model.postproc(output, offset=(0, 0))
    assert len(output[0]["info_dict"]) > 0, "Must have some nuclei."
    assert len(output[1]["info_dict"]) > 0, "Must have some layers."


def test_hovernetplus_postproc_empty_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HoVerNetPlus postproc with no detected instances."""
    model = HoVerNetPlus(
        num_types=3,
        num_layers=5,
    )

    def _fake_proc_np_hv(
        np_map: np.ndarray,
        hv_map: np.ndarray,
        scale_factor: float = 0.5,
    ) -> np.ndarray:
        """Return an empty nuclei prediction."""
        _ = np_map, hv_map, scale_factor

        return np.zeros((8, 8), dtype=np.int32)

    def _fake_proc_ls(ls_map: np.ndarray) -> np.ndarray:
        """Return an empty layer prediction."""
        _ = ls_map

        return np.zeros((8, 8), dtype=np.uint8)

    def _fake_get_instance_info(
        pred_inst: np.ndarray,
        pred_type: np.ndarray | None = None,
        offset: tuple[int, int] = (0, 0),
    ) -> dict[int, dict[str, object]]:
        """Return no nuclei instances."""
        _ = pred_inst, pred_type, offset

        return {}

    def _fake_get_layer_info(
        pred_layer: np.ndarray,
        offset: tuple[int, int] = (0, 0),
    ) -> dict[int, dict[str, object]]:
        """Return no layer objects."""
        _ = pred_layer, offset

        return {}

    monkeypatch.setattr(
        HoVerNetPlus,
        "_proc_np_hv",
        staticmethod(_fake_proc_np_hv),
    )

    monkeypatch.setattr(
        HoVerNetPlus,
        "_proc_ls",
        staticmethod(_fake_proc_ls),
    )

    monkeypatch.setattr(
        hovernetplus.HoVerNet,
        "get_instance_info",
        staticmethod(_fake_get_instance_info),
    )

    monkeypatch.setattr(
        HoVerNetPlus,
        "_get_layer_info",
        staticmethod(_fake_get_layer_info),
    )

    raw_maps = (
        np.zeros((8, 8, 1), dtype=np.float32),
        np.zeros((8, 8, 2), dtype=np.float32),
        np.zeros((8, 8, 1), dtype=np.uint8),
        np.zeros((8, 8, 1), dtype=np.uint8),
    )

    nuclei_seg, layer_seg = model.postproc(raw_maps)

    assert nuclei_seg["task_type"] == "nuclei_segmentation"
    assert layer_seg["task_type"] == "layer_segmentation"

    assert nuclei_seg["info_dict"]["box"].size == 0
    assert nuclei_seg["info_dict"]["centroid"].size == 0
    assert nuclei_seg["info_dict"]["contours"].size == 0

    assert layer_seg["info_dict"]["box"].size == 0
    assert layer_seg["info_dict"]["contours"].size == 0


def test_hovernetplus_postproc_populated_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HoVerNetPlus postproc with populated nuclei and layer outputs."""
    model = HoVerNetPlus(
        num_types=3,
        num_layers=5,
    )

    def _fake_proc_np_hv(
        np_map: np.ndarray,
        hv_map: np.ndarray,
        scale_factor: float = 0.5,
    ) -> np.ndarray:
        """Fake proc_np_hv."""
        _ = np_map, hv_map, scale_factor

        return np.array(
            [
                [0, 1],
                [0, 1],
            ],
            dtype=np.int32,
        )

    def _fake_proc_ls(ls_map: np.ndarray) -> np.ndarray:
        """Fake proc_ls."""
        _ = ls_map

        return np.array(
            [
                [1, 1],
                [1, 1],
            ],
            dtype=np.uint8,
        )

    def _fake_get_instance_info(
        pred_inst: np.ndarray,
        pred_type: np.ndarray | None = None,
        offset: tuple[int, int] = (0, 0),
    ) -> dict[int, dict[str, np.ndarray | float | int]]:
        """Fake get_instance_info."""
        _ = pred_inst, pred_type, offset

        return {
            1: {
                "box": np.array([0, 0, 2, 2]),
                "centroid": np.array([1.0, 1.0]),
                "contours": np.array([[0, 0], [1, 1]]),
                "prob": 0.95,
                "type": 1,
            },
        }

    def _fake_get_layer_info(
        pred_layer: np.ndarray,
        offset: tuple[int, int] = (0, 0),
    ) -> dict[int, dict[str, np.ndarray | int]]:
        """Fake get_layer_info."""
        _ = pred_layer, offset

        return {
            1: {
                "box": np.array([0, 0, 2, 2]),
                "contours": np.array([[0, 0], [1, 1]]),
                "type": 1,
            },
        }

    monkeypatch.setattr(
        HoVerNetPlus,
        "_proc_np_hv",
        staticmethod(_fake_proc_np_hv),
    )

    monkeypatch.setattr(
        HoVerNetPlus,
        "_proc_ls",
        staticmethod(_fake_proc_ls),
    )

    monkeypatch.setattr(
        hovernetplus.HoVerNet,
        "get_instance_info",
        staticmethod(_fake_get_instance_info),
    )

    monkeypatch.setattr(
        HoVerNetPlus,
        "_get_layer_info",
        staticmethod(_fake_get_layer_info),
    )

    raw_maps = (
        np.zeros((2, 2, 1), dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.float32),
        np.zeros((2, 2, 1), dtype=np.uint8),
        np.zeros((2, 2, 1), dtype=np.uint8),
    )

    nuclei_seg, layer_seg = model.postproc(raw_maps)

    assert len(nuclei_seg["info_dict"]["box"]) == 1
    assert len(nuclei_seg["info_dict"]["centroid"]) == 1
    assert len(nuclei_seg["info_dict"]["contours"]) == 1

    assert len(layer_seg["info_dict"]["box"]) == 1
    assert len(layer_seg["info_dict"]["contours"]) == 1
