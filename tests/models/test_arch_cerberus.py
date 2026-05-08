"""Unit tests for the Cerberus architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from tiatoolbox.models import Cerberus
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.engine.io_config import IOInstanceSegmentorConfig

if TYPE_CHECKING:
    import pytest

PATCH_OUTPUT_SHAPE = (144, 144)
INFER_INPUT_SHAPE = (256, 256)


def _module_prefixed_state_dict(model: Cerberus) -> dict[str, torch.Tensor]:
    """Return a Cerberus checkpoint state dict saved from DataParallel."""
    return {f"module.{key}": value for key, value in model.state_dict().items()}


def test_cerberus_load_weights_from_desc_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Cerberus checkpoint loading with ``desc`` and ``module.`` prefixes."""
    source_model = Cerberus()
    checkpoint = {"desc": _module_prefixed_state_dict(source_model)}

    def _mock_torch_load(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return checkpoint

    monkeypatch.setattr(torch, "load", _mock_torch_load)

    model = Cerberus()
    model.load_weights_from_file("weights.tar")

    state_key = "backbone.conv1.weight"
    assert torch.equal(
        model.state_dict()[state_key],
        source_model.state_dict()[state_key],
    )


def test_cerberus_pretrained_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the Cerberus pretrained registry entry and model IO config."""
    checkpoint = {"desc": _module_prefixed_state_dict(Cerberus())}

    def _mock_torch_load(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return checkpoint

    monkeypatch.setattr(torch, "load", _mock_torch_load)

    model, ioconfig = get_pretrained_model(
        "cerberus-resnet34",
        pretrained_weights="weights.tar",
    )

    assert isinstance(model, Cerberus)
    assert isinstance(ioconfig, IOInstanceSegmentorConfig)
    assert tuple(ioconfig.patch_input_shape) == (448, 448)
    assert tuple(ioconfig.patch_output_shape) == PATCH_OUTPUT_SHAPE
    assert tuple(ioconfig.stride_shape) == PATCH_OUTPUT_SHAPE
    assert len(ioconfig.output_resolutions) == len(Cerberus.head_names)


def test_cerberus_infer_batch_output_shapes() -> None:
    """Test Cerberus inference output order and shape."""
    model = Cerberus()
    batch = torch.zeros((1, *INFER_INPUT_SHAPE, 3), dtype=torch.uint8)

    outputs = model.infer_batch(model, batch, device="cpu")

    assert len(outputs) == len(Cerberus.head_names)
    expected_shapes = (
        (1, *PATCH_OUTPUT_SHAPE, 2),
        (1, *PATCH_OUTPUT_SHAPE, 1),
        (1, *PATCH_OUTPUT_SHAPE, 2),
        (1, *PATCH_OUTPUT_SHAPE, 1),
        (1, *PATCH_OUTPUT_SHAPE, 2),
        (1, *PATCH_OUTPUT_SHAPE, 1),
    )
    for output, expected_shape in zip(outputs, expected_shapes, strict=True):
        assert output.shape == expected_shape
        assert output.dtype == np.float32


def test_cerberus_postproc_empty_maps() -> None:
    """Test Cerberus post-processing output structure for empty predictions."""
    raw_maps = [
        np.zeros((*PATCH_OUTPUT_SHAPE, 2), dtype=np.float32),
        np.zeros((*PATCH_OUTPUT_SHAPE, 1), dtype=np.float32),
        np.zeros((*PATCH_OUTPUT_SHAPE, 2), dtype=np.float32),
        np.zeros((*PATCH_OUTPUT_SHAPE, 1), dtype=np.float32),
        np.zeros((*PATCH_OUTPUT_SHAPE, 2), dtype=np.float32),
        np.zeros((*PATCH_OUTPUT_SHAPE, 1), dtype=np.float32),
    ]

    outputs = Cerberus().postproc(raw_maps, offset=(3, 5))

    assert [output["task_type"] for output in outputs] == ["nuclei", "gland", "lumen"]
    for output in outputs:
        assert output["seg_type"] == "instance"
        assert output["predictions"].shape == PATCH_OUTPUT_SHAPE
        assert output["predictions"].dtype == np.int32

        info_dict = output["info_dict"]
        assert info_dict["box"].shape == (0, 4)
        assert info_dict["box"].dtype == np.int32
        assert info_dict["centroid"].shape == (0, 2)
        assert info_dict["centroid"].dtype == np.float32
        assert info_dict["contours"].shape == (0, 0, 2)
        assert info_dict["contours"].dtype == np.int32
        assert info_dict["prob"].shape == (0,)
        assert info_dict["prob"].dtype == np.float32
        assert info_dict["type"].shape == (0,)
        assert info_dict["type"].dtype == np.int32
