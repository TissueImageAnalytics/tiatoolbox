"""Unit tests for SegFormer architecture components."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import torch

from tiatoolbox.models.architecture.segformer import (
    MIN_SEGFORMER_DECODER_DEPTH,
    MLP,
    Segformer,
    SegformerDecoder,
)


def test_mlp_forward_shape() -> None:
    """Decoder MLP should preserve spatial dimensions after projection."""
    mlp = MLP(skip_channels=8, segmentation_channels=16)
    x = torch.randn(2, 8, 16, 16)
    output = mlp(x)
    assert output.shape == (2, 16, 16, 16)


def test_segformer_decoder_depth_validation() -> None:
    """SegFormer decoder should reject depth values below the minimum."""
    with pytest.raises(ValueError, match="cannot be less than"):
        SegformerDecoder(
            encoder_channels=[3, 0, 8, 16, 24, 32],
            encoder_depth=MIN_SEGFORMER_DECODER_DEPTH - 1,
            segmentation_channels=8,
        )


def test_segformer_decoder_forward_with_empty_second_feature() -> None:
    """Decoder should handle the encoder dummy feature branch."""
    decoder = SegformerDecoder(
        encoder_channels=[3, 0, 8, 16, 32, 64],
        segmentation_channels=8,
    )

    features = [
        torch.randn(2, 3, 64, 64),
        torch.randn(2, 0, 32, 32),
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 32, 4, 4),
        torch.randn(2, 64, 2, 2),
    ]

    output = decoder(features)
    assert output.shape == (2, 8, 16, 16)


def test_segformer_decoder_forward_with_non_empty_second_feature() -> None:
    """Decoder should handle feature lists without a dummy second tensor."""
    decoder = SegformerDecoder(
        encoder_channels=[3, 4, 8, 16, 32, 64],
        segmentation_channels=8,
    )

    features = [
        torch.randn(1, 3, 64, 64),
        torch.randn(1, 4, 32, 32),
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 32, 4, 4),
        torch.randn(1, 64, 2, 2),
    ]

    output = decoder(features)
    assert output.shape == (1, 8, 16, 16)


def test_segformer_forward_and_metadata() -> None:
    """Full SegFormer forward should produce class logits at input resolution."""
    model = Segformer(
        encoder_name="mit_b0",
        in_channels=3,
        classes=2,
        upsampling=4,
    )
    x = torch.randn(1, 3, 64, 64)
    output = model(x)

    assert model.requires_divisible_input_shape is True
    assert model.name == "segformer-mit_b0"
    assert output.shape == (1, 2, 64, 64)


def test_segformer_preproc_imagenet_normalization() -> None:
    """Preprocessing should apply ImageNet channel normalization."""
    image = np.ones((8, 8, 3), dtype=np.uint8) * 128
    processed = Segformer.preproc(image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    expected = (128 / 255.0 - mean) / std

    assert processed.shape == image.shape
    assert np.allclose(processed[0, 0, :], expected, rtol=1e-5)


def test_segformer_postproc_with_numpy_and_dask() -> None:
    """Postprocessing should support NumPy arrays and Dask arrays."""
    model = Segformer(encoder_name="mit_b0", classes=2)

    probs_np = np.random.default_rng(5).random((32, 32, 2), dtype=np.float32)
    mask_np = model.postproc(probs_np)

    probs_da = da.from_array(probs_np, chunks=(16, 16, 2))
    mask_da = model.postproc(probs_da)

    assert mask_np.shape == (32, 32)
    assert mask_np.dtype == np.uint8
    assert mask_da.shape == (32, 32)
    assert mask_da.dtype == np.uint8


def test_segformer_infer_batch_probability_output() -> None:
    """Batch inference should return NHWC softmax probabilities."""
    model = Segformer(encoder_name="mit_b0", classes=2)
    batch = torch.randn(2, 64, 64, 3)

    probs = Segformer.infer_batch(model, batch, device="cpu")

    assert probs.shape == (2, 64, 64, 2)
    assert isinstance(probs, np.ndarray)
    assert np.allclose(np.sum(probs, axis=-1), 1.0, atol=1e-5)
