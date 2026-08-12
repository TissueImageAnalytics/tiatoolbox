"""Unit test package for HoVerNet."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn

from tiatoolbox.models import HoVerNet
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.architecture.hovernet import (
    DenseBlock,
    ResidualBlock,
    TFSamepaddingLayer,
)
from tiatoolbox.utils.misc import select_device
from tiatoolbox.wsicore.wsireader import WSIReader


def test_functionality(remote_sample: Callable) -> None:
    """Functionality test."""
    sample_wsi = str(remote_sample("wsi1_2k_2k_svs"))
    reader = WSIReader.open(sample_wsi)

    # * test fast mode (architecture used in PanNuke paper)
    patch = reader.read_bounds(
        (0, 0, 256, 256),
        resolution=0.25,
        units="mpp",
        coord_space="resolution",
    )
    batch = torch.from_numpy(patch)[None]
    model = HoVerNet(num_types=6, mode="fast")
    weights_path = fetch_pretrained_weights("hovernet_fast-pannuke")
    pretrained = torch.load(weights_path)
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, device=select_device(on_gpu=False))
    output = [v[0] for v in output]
    output = model.postproc(output, offset=(0, 0))
    assert len(output[0]["info_dict"]) > 0, "Must have some nuclei."

    # * test original mode on CoNSeP dataset (architecture used in HoVerNet paper)
    patch = reader.read_bounds(
        (0, 0, 270, 270),
        resolution=0.25,
        units="mpp",
        coord_space="resolution",
    )
    batch = torch.from_numpy(patch)[None]
    model = HoVerNet(num_types=5, mode="original")
    weights_path = fetch_pretrained_weights("hovernet_original-consep")
    pretrained = torch.load(weights_path)
    model.load_state_dict(pretrained)
    output = model.infer_batch(model, batch, device=select_device(on_gpu=False))
    output = [v[0] for v in output]
    output = model.postproc(output, offset=(0, 0))
    assert len(output[0]["info_dict"]) > 0, "Must have some nuclei."

    # test crash when providing exotic mode
    with pytest.raises(ValueError, match=r".*Invalid mode.*"):
        _ = HoVerNet(num_types=None, mode="super")


def test_unit_blocks() -> None:
    """Test for blocks within HoVerNet."""
    # padding
    model = nn.Sequential(TFSamepaddingLayer(7, 1), nn.Conv2d(3, 3, 7, 1, padding=0))
    sample = torch.rand((1, 3, 14, 14), dtype=torch.float32)
    output = model(sample)
    assert np.sum(output.shape - np.array([1, 3, 14, 14])) == 0, f"{output.shape}"

    # padding with stride and odd shape
    model = nn.Sequential(TFSamepaddingLayer(7, 2), nn.Conv2d(3, 3, 7, 2, padding=0))
    sample = torch.rand((1, 3, 15, 15), dtype=torch.float32)
    output = model(sample)
    assert np.sum(output.shape - np.array([1, 3, 8, 8])) == 0, f"{output.shape}"

    # *
    sample = torch.rand((1, 16, 15, 15), dtype=torch.float32)

    block = ResidualBlock(16, [1, 3, 1], [16, 16, 16], 3)

    assert block.shortcut is None
    output = block(sample)
    assert np.sum(output.shape - np.array([1, 16, 15, 15])) == 0, f"{output.shape}"

    block = ResidualBlock(16, [1, 3, 1], [16, 16, 32], 3)
    assert block.shortcut is not None
    output = block(sample)
    assert np.sum(output.shape - np.array([1, 32, 15, 15])) == 0, f"{output.shape}"

    block = DenseBlock(16, [1, 3], [16, 16], 3)
    output = block(sample)
    assert output.shape[1] == 16 * 4, f"{output.shape}"

    # test crash when providing exotic mode
    with pytest.raises(ValueError, match=r".*Unbalance Unit Info.*"):
        _ = DenseBlock(16, [1, 3, 1], [16, 16], 3)
    with pytest.raises(ValueError, match=r".*Unbalance Unit Info.*"):
        _ = ResidualBlock(16, [1, 3, 1], [16, 16], 3)


def test_hovernet_postproc_empty_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HoVerNet postproc when no nuclei are detected."""
    model = HoVerNet()

    def _fake_proc_np_hv(
        np_map: np.ndarray,
        hv_map: np.ndarray,
        scale_factor: float = 1,
    ) -> np.ndarray:
        """Return an empty instance map."""
        _ = np_map, hv_map, scale_factor

        return np.zeros(
            (8, 8),
            dtype=np.int32,
        )

    monkeypatch.setattr(
        HoVerNet,
        "_proc_np_hv",
        staticmethod(_fake_proc_np_hv),
    )

    def _fake_get_instance_info_empty(
        pred_inst: np.ndarray,
        pred_type: np.ndarray | None = None,
        offset: tuple[int, int] = (0, 0),
    ) -> dict[int, dict]:
        """Return an empty instance dictionary."""
        _ = pred_inst, pred_type, offset

        return {}

    monkeypatch.setattr(
        HoVerNet,
        "get_instance_info",
        staticmethod(_fake_get_instance_info_empty),
    )

    raw_maps = (
        np.zeros((8, 8, 1), dtype=np.float32),
        np.zeros((8, 8, 2), dtype=np.float32),
    )

    output = model.postproc(raw_maps)

    nuclei_seg = output[0]

    assert nuclei_seg["task_type"] == "nuclei_segmentation"

    info_dict = nuclei_seg["info_dict"]

    assert info_dict["box"].size == 0
    assert info_dict["centroid"].size == 0
    assert info_dict["contours"].size == 0
    assert info_dict["prob"].size == 0
    assert info_dict["type"].size == 0


def test_hovernet_postproc_with_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HoVerNet postproc with detected instances."""
    model = HoVerNet()

    def _fake_proc_np_hv(
        np_map: np.ndarray,
        hv_map: np.ndarray,
        scale_factor: float = 1,
    ) -> np.ndarray:
        """Return an empty instance map."""
        _ = np_map, hv_map, scale_factor

        return np.zeros(
            (8, 8),
            dtype=np.int32,
        )

    monkeypatch.setattr(
        HoVerNet,
        "_proc_np_hv",
        staticmethod(_fake_proc_np_hv),
    )

    def _fake_get_instance_info(
        pred_inst: np.ndarray,
        pred_type: np.ndarray | None = None,
        offset: tuple[int, int] = (0, 0),
    ) -> dict[int, dict]:
        """Return a single nucleus instance."""
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

    monkeypatch.setattr(
        HoVerNet,
        "get_instance_info",
        staticmethod(_fake_get_instance_info),
    )

    raw_maps = (
        np.zeros((2, 2, 1), dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.float32),
    )

    output = model.postproc(raw_maps)

    nuclei_seg = output[0]

    assert nuclei_seg["task_type"] == "nuclei_segmentation"
    assert len(nuclei_seg["info_dict"]["box"]) == 1


def test_hovernet_postproc_with_type_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HoVerNet postproc when a type map is supplied."""
    model = HoVerNet(num_types=5)

    called = {"has_type_map": False}

    def _fake_get_instance_info(
        pred_inst: np.ndarray,
        pred_type: np.ndarray | None = None,
        offset: tuple[int, int] = (0, 0),
    ) -> dict[int, dict]:
        """Fake _get_instance_info method."""
        _ = pred_inst, offset
        called["has_type_map"] = pred_type is not None
        return {}

    def _fake_proc_np_hv(
        np_map: np.ndarray,
        hv_map: np.ndarray,
        scale_factor: float = 1,
    ) -> np.ndarray:
        """Return an empty instance map."""
        _ = np_map, hv_map, scale_factor

        return np.zeros(
            (8, 8),
            dtype=np.int32,
        )

    monkeypatch.setattr(
        HoVerNet,
        "_proc_np_hv",
        staticmethod(_fake_proc_np_hv),
    )

    monkeypatch.setattr(
        HoVerNet,
        "get_instance_info",
        staticmethod(_fake_get_instance_info),
    )

    output = model.postproc(
        (
            np.zeros((4, 4, 1), dtype=np.float32),
            np.zeros((4, 4, 2), dtype=np.float32),
            np.zeros((4, 4, 1), dtype=np.uint8),
        ),
    )

    assert len(output) == 1
    assert called["has_type_map"] is True
