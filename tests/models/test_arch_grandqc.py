"""Unit test package for GrandQC Tissue Model."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.models.architecture import (
    fetch_pretrained_weights,
    get_pretrained_model,
)
from tiatoolbox.models.architecture.grandqc import (
    CenterBlock,
    GrandQCModel,
    SegmentationHead,
    UnetPlusPlusDecoder,
)
from tiatoolbox.models.engine.io_config import IOSegmentorConfig
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.wsicore.wsireader import VirtualWSIReader

device = "cuda" if toolbox_env.has_gpu() else "cpu"


def test_functional_grandqc() -> None:
    """Test for GrandQC model."""
    # test fetch pretrained weights
    pretrained_weights = fetch_pretrained_weights("grandqc_tissue_detection")
    assert pretrained_weights is not None

    # test creation
    model = GrandQCModel(num_output_channels=2)
    assert model is not None

    # load pretrained weights
    pretrained = torch.load(pretrained_weights, map_location=device)
    model.load_state_dict(pretrained)

    # test get pretrained model
    model, ioconfig = get_pretrained_model("grandqc_tissue_detection")
    assert isinstance(model, GrandQCModel)
    assert isinstance(ioconfig, IOSegmentorConfig)
    assert model.num_output_channels == 2
    assert model.decoder_channels == (256, 128, 64, 32, 16)

    # test inference
    generator = np.random.default_rng(1337)
    test_image = generator.integers(0, 256, size=(2048, 2048, 3), dtype=np.uint8)
    reader = VirtualWSIReader.open(test_image)
    read_kwargs = {"resolution": 0, "units": "level", "coord_space": "resolution"}
    batch = np.array(
        [
            reader.read_bounds((0, 0, 512, 512), **read_kwargs),
            reader.read_bounds((512, 512, 1024, 1024), **read_kwargs),
        ],
    )
    batch = torch.from_numpy(batch)
    output = model.infer_batch(model, batch, device=device)
    assert output.shape == (2, 512, 512, 2)


def test_grandqc_preproc_postproc() -> None:
    """Test GrandQC preproc and postproc functions."""
    model = GrandQCModel(num_output_channels=2)

    generator = np.random.default_rng(1337)
    # test preproc
    dummy_image = generator.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    preproc_image = model.preproc(dummy_image)
    assert preproc_image.shape == dummy_image.shape
    assert preproc_image.dtype == np.float64

    # test postproc
    dummy_output = generator.random(size=(512, 512, 2), dtype=np.float32)
    postproc_image = model.postproc(dummy_output)
    assert postproc_image.shape == (512, 512)
    assert postproc_image.dtype == np.int64


def test_grandqc_with_semantic_segmentor(
    remote_sample: Callable, track_tmp_path: Path
) -> None:
    """Test GrandQC tissue mask generation."""
    segmentor = SemanticSegmentor(model="grandqc_tissue_detection")

    sample_image = remote_sample("svs-1-small")
    inputs = [str(sample_image)]

    output = segmentor.run(
        images=inputs,
        device=device,
        patch_mode=False,
        output_type="annotationstore",
        save_dir=track_tmp_path / "grandqc_test_outputs",
        overwrite=True,
        class_dict={0: "background", 1: "tissue"},
    )

    assert len(output) == 1
    assert Path(output[sample_image]).exists()

    store = SQLiteStore.open(output[sample_image])
    assert len(store) == 4

    unique_types = set()

    tissue_area_px = 0.0
    for annotation in store.values():
        unique_types.add(annotation.properties["type"])
        if annotation.properties["type"] == "tissue":
            tissue_area_px += annotation.geometry.area
    assert 2999000 < tissue_area_px < 3004000

    assert unique_types == {"background", "tissue"}
    store.close()


def test_segmentation_head_behaviour() -> None:
    """Verify SegmentationHead defaults and upsampling."""
    head = SegmentationHead(3, 5, activation=None, upsampling=1)
    assert isinstance(head[1], nn.Identity)
    assert isinstance(head[2], nn.Identity)

    x = torch.randn(1, 3, 6, 8)
    out = head(x)
    assert out.shape == (1, 5, 6, 8)

    head = SegmentationHead(3, 2, activation=nn.Sigmoid(), upsampling=2)
    x = torch.ones(1, 3, 4, 4)
    out = head(x)
    assert out.shape == (1, 2, 8, 8)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_unetplusplus_decoder_forward_shapes() -> None:
    """Ensure UnetPlusPlusDecoder handles dense connections."""
    decoder = UnetPlusPlusDecoder(
        encoder_channels=[1, 2, 4, 8],
        decoder_channels=[8, 4, 2],
        n_blocks=3,
    )

    features = [
        torch.randn(1, 1, 32, 32),
        torch.randn(1, 2, 16, 16),
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 8, 4, 4),
    ]

    output = decoder(features)
    assert output.shape == (1, 2, 32, 32)


def test_center_block_behavior() -> None:
    """Test CenterBlock behavior in UnetPlusPlusDecoder."""
    center_block = CenterBlock(in_channels=8, out_channels=8)

    x = torch.randn(1, 8, 4, 4)
    out = center_block(x)
    assert out.shape == (1, 8, 4, 4)


def test_unetpp_raises_value_error() -> None:
    """Test UnetPlusPlusDecoder raises ValueError."""
    with pytest.raises(
        ValueError, match=r".*depth is 4, but you provide `decoder_channels` for 3.*"
    ):
        _ = UnetPlusPlusDecoder(
            encoder_channels=[1, 2, 4, 8],
            decoder_channels=[8, 4, 2],
            n_blocks=4,
        )
