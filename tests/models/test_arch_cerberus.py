"""Unit tests for the Cerberus architecture."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import torch

from tiatoolbox.models import Cerberus
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.architecture.cerberus.model import (
    _build_tissue_raw_map,
    _crop_center_tensor,
    _inst_dict_for_dask_processing,
    _pad_contours,
)
from tiatoolbox.models.architecture.cerberus.postproc import (
    PostProcInstErodedContourMap,
    get_bounding_box,
)
from tiatoolbox.models.engine.io_config import IOInstanceSegmentorConfig

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
        """Return a synthetic Cerberus checkpoint for load-weight tests."""
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
        """Return a synthetic Cerberus checkpoint for registry loading."""
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


def test_cerberus_postproc_dask_maps_and_lumen_gland_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Cerberus post-processing Dask output and lumen-in-gland masking."""
    output_shape = (16, 16)
    raw_maps = [
        da.from_array(
            np.zeros((*output_shape, channels), dtype=np.float32),
            chunks=(8, 8, channels),
        )
        for channels in (2, 1, 2, 1, 2, 1)
    ]
    calls = []

    def _mock_post_process(
        raw_map: np.ndarray,
        idx_dict: dict[str, list[int]],
        tissue_mode: str,
        ds_factor: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return deterministic task maps for Cerberus postproc testing."""
        calls.append((tissue_mode, raw_map.shape, idx_dict, ds_factor))
        inst_map = np.zeros(output_shape, dtype=np.int32)
        type_map = np.ones(output_shape, dtype=np.uint8)
        if tissue_mode == "Nuclei":
            inst_map[2:5, 2:5] = 1
        elif tissue_mode == "Gland":
            inst_map[1:8, 1:8] = 1
        else:
            inst_map[3:6, 3:6] = 1
            inst_map[10:13, 10:13] = 2
            type_map = None
        return inst_map, type_map

    def _mock_get_instance_info(
        inst_map: np.ndarray,
        type_map: np.ndarray | None,
        offset: tuple[int, int],
        verbose: object,
    ) -> dict[int, dict]:
        """Return deterministic instance metadata for Cerberus postproc tests."""
        assert offset == (7, 11)
        assert verbose is False
        type_value = 0 if type_map is None else int(type_map[inst_map > 0][0])
        return {
            1: {
                "box": np.array([1, 2, 3, 4], dtype=np.int32),
                "centroid": np.array([2.5, 3.5], dtype=np.float32),
                "contours": np.array([[1, 2], [3, 4]], dtype=np.int32),
                "prob": 0.75,
                "type": type_value,
            },
        }

    monkeypatch.setattr(
        PostProcInstErodedContourMap,
        "post_process",
        _mock_post_process,
    )
    monkeypatch.setattr(
        "tiatoolbox.models.architecture.cerberus.model.HoVerNet.get_instance_info",
        _mock_get_instance_info,
    )

    outputs = Cerberus().postproc(raw_maps, offset=(7, 11))

    assert [call[0] for call in calls] == ["Nuclei", "Gland", "Lumen"]
    assert calls[0][1:] == (
        (16, 16, 3),
        {"Nuclei-INST": [0, 2], "Nuclei-TYPE": [2, 3]},
        1.0,
    )
    assert [output["task_type"] for output in outputs] == ["nuclei", "gland", "lumen"]
    lumen_map = outputs[2]["predictions"].compute()
    assert np.all(lumen_map[3:6, 3:6] == 1)
    assert np.all(lumen_map[10:13, 10:13] == 0)
    for output in outputs:
        assert isinstance(output["predictions"], da.Array)
        assert output["predictions"].dtype == np.int32
        assert output["info_dict"]["box"].compute().dtype == np.int32
        assert output["info_dict"]["centroid"].compute().dtype == np.float32
        assert output["info_dict"]["contours"].compute().shape == (1, 2, 2)
        assert output["info_dict"]["prob"].compute().dtype == np.float32
        assert output["info_dict"]["type"].compute().dtype == np.int32


def test_cerberus_model_helpers() -> None:
    """Test Cerberus private helper conversions."""
    tissue_map, idx_dict = _build_tissue_raw_map(
        {
            "Nuclei-INST": np.zeros((4, 5, 2), dtype=np.float32),
            "Nuclei-TYPE": np.ones((4, 5), dtype=np.float32),
        },
        "Nuclei",
    )
    assert tissue_map.shape == (4, 5, 3)
    assert idx_dict == {"Nuclei-INST": [0, 2], "Nuclei-TYPE": [2, 3]}

    tensor = torch.arange(1 * 5 * 6 * 1, dtype=torch.float32).reshape(1, 5, 6, 1)
    cropped = _crop_center_tensor(tensor, (3, 4))
    assert cropped.shape == (1, 3, 4, 1)
    assert torch.equal(cropped, tensor[:, 1:4, 1:5, :])

    contours = np.array(
        [
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[5, 6]], dtype=np.int32),
        ],
        dtype=object,
    )
    padded = _pad_contours(contours)
    assert padded.shape == (2, 2, 2)
    assert np.array_equal(padded[1, 0], [5, 6])
    assert np.array_equal(padded[1, 1], [np.iinfo(np.int32).min] * 2)

    dask_info = _inst_dict_for_dask_processing({}, is_dask=True)
    assert dask_info["contours"].compute().shape == (0, 0, 2)
    assert dask_info["type"].compute().dtype == np.int32


def test_cerberus_eroded_contour_postproc_non_empty_and_errors() -> None:
    """Test non-empty Cerberus contour post-processing and validation errors."""
    nuclei_raw_map = np.zeros((40, 40, 2), dtype=np.float32)
    nuclei_raw_map[6:18, 6:18, 0] = 0.9
    nuclei_raw_map[22:34, 22:34, 0] = 0.9
    nuclei_inst_map, nuclei_type_map = PostProcInstErodedContourMap.post_process(
        raw_map=nuclei_raw_map,
        idx_dict={"Nuclei-INST": [0, 2]},
        tissue_mode="Nuclei",
    )
    assert nuclei_inst_map.shape == (40, 40)
    assert nuclei_inst_map.max() == 2
    assert get_bounding_box(nuclei_inst_map > 0) == (7, 33, 7, 33)
    assert nuclei_type_map is None

    gland_raw_map = np.zeros((80, 80, 3), dtype=np.float32)
    gland_raw_map[10:60, 10:60, 0] = 0.9
    gland_raw_map[..., 2] = 2

    inst_map, type_map = PostProcInstErodedContourMap.post_process(
        raw_map=gland_raw_map,
        idx_dict={"Gland-INST": [0, 2], "Gland-TYPE": [2, 3]},
        tissue_mode="Gland",
    )

    assert inst_map.shape == (80, 80)
    assert inst_map.max() == 1
    assert type_map is not None
    assert type_map.shape == (80, 80)
    assert np.all(type_map == 2)
    assert get_bounding_box(inst_map > 0) == (6, 65, 6, 65)

    lumen_raw_map = np.zeros((40, 40, 2), dtype=np.float32)
    lumen_raw_map[8:25, 8:25, 0] = 0.9
    lumen_inst_map, lumen_type_map = PostProcInstErodedContourMap.post_process(
        raw_map=lumen_raw_map,
        idx_dict={"Lumen-INST": [0, 2]},
        tissue_mode="Lumen",
    )
    assert lumen_inst_map.max() == 1
    assert lumen_type_map is None

    with pytest.raises(ValueError, match="Unsupported Cerberus tissue mode"):
        PostProcInstErodedContourMap.post_process(
            raw_map=lumen_raw_map,
            idx_dict={"Lumen-INST": [0, 2]},
            tissue_mode="Stroma",
        )

    with pytest.raises(KeyError, match="Missing required Cerberus map"):
        PostProcInstErodedContourMap.post_process(
            raw_map=lumen_raw_map,
            idx_dict={},
            tissue_mode="Lumen",
        )
