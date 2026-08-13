"""Unit tests for SegFormer architecture."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import torch

from tiatoolbox.models.architecture.segformer import Segformer
from tiatoolbox.models.architecture.segformer_hf import (
    is_legacy_segformer_state_dict,
    remap_legacy_segformer_state_dict,
)


def _hf_to_legacy_state_dict(
    hf_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Invert HF keys into a minimal legacy TIA/SMP-style state dict for tests."""
    legacy: dict[str, torch.Tensor] = {}
    pending_kv: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}

    for key, value in hf_state.items():
        if key.startswith("model."):
            key = key[len("model.") :]

        if key.startswith("segformer.encoder.patch_embeddings."):
            # segformer.encoder.patch_embeddings.{s}.{proj|layer_norm}.{wb}
            parts = key.split(".")
            stage = int(parts[3]) + 1
            which = "proj" if parts[4] == "proj" else "norm"
            legacy[f"encoder.patch_embed{stage}.{which}.{parts[5]}"] = value
            continue

        if key.startswith("segformer.encoder.layer_norm."):
            parts = key.split(".")
            stage = int(parts[3]) + 1
            legacy[f"encoder.norm{stage}.{parts[4]}"] = value
            continue

        if key.startswith("segformer.encoder.block."):
            parts = key.split(".")
            stage = int(parts[3]) + 1
            block_idx = parts[4]
            rest = ".".join(parts[5:])
            base = f"encoder.block{stage}.{block_idx}."

            if rest.startswith("layer_norm_1."):
                legacy[base + "norm1." + rest.split(".", 1)[1]] = value
            elif rest.startswith("layer_norm_2."):
                legacy[base + "norm2." + rest.split(".", 1)[1]] = value
            elif rest.startswith("attention.self.query."):
                legacy[base + "attn.q." + rest.split(".", 3)[3]] = value
            elif rest.startswith("attention.self.key."):
                wb = rest.split(".", 3)[3]
                pending_kv[(str(stage), block_idx, wb)] = {
                    **pending_kv.get((str(stage), block_idx, wb), {}),
                    "key": value,
                }
            elif rest.startswith("attention.self.value."):
                wb = rest.split(".", 3)[3]
                pending_kv[(str(stage), block_idx, wb)] = {
                    **pending_kv.get((str(stage), block_idx, wb), {}),
                    "value": value,
                }
            elif rest.startswith("attention.self.sr."):
                legacy[base + "attn.sr." + rest.split(".", 3)[3]] = value
            elif rest.startswith("attention.self.layer_norm."):
                legacy[base + "attn.norm." + rest.split(".", 3)[3]] = value
            elif rest.startswith("attention.output.dense."):
                legacy[base + "attn.proj." + rest.split(".", 3)[3]] = value
            elif rest.startswith("mlp.dense1."):
                legacy[base + "mlp.fc1." + rest.split(".", 2)[2]] = value
            elif rest.startswith("mlp.dense2."):
                legacy[base + "mlp.fc2." + rest.split(".", 2)[2]] = value
            elif rest.startswith("mlp.dwconv.dwconv."):
                legacy[base + "mlp.dwconv.dwconv." + rest.split(".", 3)[3]] = value
            continue

        if key.startswith("decode_head.linear_c."):
            # decode_head.linear_c.{hf_i}.proj.{wb} -> decoder.mlp_stage.{3-hf_i}
            parts = key.split(".")
            hf_i = int(parts[2])
            tia_i = 3 - hf_i
            legacy[f"decoder.mlp_stage.{tia_i}.linear.{parts[4]}"] = value
            continue

        if key == "decode_head.linear_fuse.weight":
            legacy["decoder.fuse_stage.0.weight"] = value
            continue

        if key.startswith("decode_head.batch_norm."):
            legacy["decoder.fuse_stage.1." + key.split(".", 2)[2]] = value
            continue

        if key.startswith("decode_head.classifier."):
            legacy["segmentation_head.0." + key.split(".", 2)[2]] = value
            continue

    for (stage, block_idx, wb), parts in pending_kv.items():
        legacy[f"encoder.block{stage}.{block_idx}.attn.kv.{wb}"] = torch.cat(
            [parts["key"], parts["value"]],
            dim=0,
        )

    return legacy


def test_unknown_encoder_name_raises() -> None:
    """Unsupported encoder names should raise a clear ValueError."""
    with pytest.raises(ValueError, match="Unknown encoder_name"):
        Segformer(encoder_name="mit_b99", classes=2)


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


def test_remap_kv_split_and_decoder_reverse() -> None:
    """Remapper should split fused kv weights and reverse decoder MLP indices."""
    legacy = {
        "encoder.block1.0.attn.kv.weight": torch.arange(32, dtype=torch.float32).reshape(
            8,
            4,
        ),
        "encoder.block1.0.attn.kv.bias": torch.arange(8, dtype=torch.float32),
        "decoder.mlp_stage.0.linear.weight": torch.ones(16, 32),
        "decoder.mlp_stage.1.linear.weight": torch.ones(16, 24),
        "decoder.mlp_stage.2.linear.weight": torch.ones(16, 16),
        "decoder.mlp_stage.3.linear.weight": torch.ones(16, 8),
        "segmentation_head.0.weight": torch.ones(5, 16, 1, 1),
        "segmentation_head.0.bias": torch.zeros(5),
    }

    assert is_legacy_segformer_state_dict(legacy)
    remapped = remap_legacy_segformer_state_dict(legacy)

    key_w = remapped["segformer.encoder.block.0.0.attention.self.key.weight"]
    val_w = remapped["segformer.encoder.block.0.0.attention.self.value.weight"]
    assert key_w.shape == (4, 4)
    assert val_w.shape == (4, 4)
    assert torch.equal(key_w, legacy["encoder.block1.0.attn.kv.weight"][:4])
    assert torch.equal(val_w, legacy["encoder.block1.0.attn.kv.weight"][4:])

    assert remapped["decode_head.linear_c.3.proj.weight"].shape == (16, 32)
    assert remapped["decode_head.linear_c.0.proj.weight"].shape == (16, 8)
    assert remapped["decode_head.classifier.weight"].shape == (5, 16, 1, 1)


@pytest.mark.parametrize("num_classes", [2, 6])
def test_legacy_checkpoint_load_for_variable_classes(num_classes: int) -> None:
    """Legacy remapping should work for checkpoints that only differ by class count."""
    model = Segformer(
        encoder_name="mit_b0",
        decoder_segmentation_channels=64,
        classes=num_classes,
        upsampling=4,
    )
    legacy = _hf_to_legacy_state_dict(model.state_dict())
    assert is_legacy_segformer_state_dict(legacy)
    assert legacy["segmentation_head.0.weight"].shape[0] == num_classes

    fresh = Segformer(
        encoder_name="mit_b0",
        decoder_segmentation_channels=64,
        classes=num_classes,
        upsampling=4,
    )
    incompatible = fresh.load_state_dict(legacy, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert fresh(x).shape == (1, num_classes, 64, 64)


def test_native_hf_state_dict_still_loads() -> None:
    """Native Hugging Face-shaped state dicts should load without remapping."""
    model = Segformer(encoder_name="mit_b0", classes=3, decoder_segmentation_channels=32)
    state = model.state_dict()
    assert not is_legacy_segformer_state_dict(state)

    other = Segformer(
        encoder_name="mit_b0",
        classes=3,
        decoder_segmentation_channels=32,
    )
    incompatible = other.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
