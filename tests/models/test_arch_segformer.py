"""Unit tests for SegFormer architecture."""

from __future__ import annotations

import re

import dask.array as da
import numpy as np
import pytest
import torch

from tiatoolbox.models.architecture.segformer import Segformer
from tiatoolbox.models.architecture.segformer_hf import (
    is_smp_segformer_state_dict,
    remap_smp_segformer_state_dict,
)

_BLOCK_REST_RULES: tuple[tuple[str, str, int], ...] = (
    ("layernorm_before.", "norm1.", 1),
    ("layernorm_after.", "norm2.", 1),
    ("attention.q_proj.", "attn.q.", 2),
    ("attention.sequence_reduction.sequence_reduction.", "attn.sr.", 3),
    ("attention.sequence_reduction.layer_norm.", "attn.norm.", 3),
    ("attention.o_proj.", "attn.proj.", 2),
    ("mlp.fc1.", "mlp.fc1.", 2),
    ("mlp.fc2.", "mlp.fc2.", 2),
    ("mlp.dwconv.dwconv.", "mlp.dwconv.dwconv.", 3),
)


def _strip_model_prefix(key: str) -> str:
    """Remove optional ``model.`` prefix from a wrapped HF state-dict key."""
    if key.startswith("model."):
        return key[len("model.") :]
    return key


def _invert_encoder_block_rest(
    base: str,
    rest: str,
    value: torch.Tensor,
    smp: dict[str, torch.Tensor],
    pending_kv: dict[tuple[str, str, str], dict[str, torch.Tensor]],
    stage: str,
    block_idx: str,
) -> None:
    """Map one HF encoder-block parameter back to SMP naming."""
    if rest.startswith("attention.k_proj."):
        wb = rest.rsplit(".", 1)[-1]
        pending_kv[(stage, block_idx, wb)] = {
            **pending_kv.get((stage, block_idx, wb), {}),
            "key": value,
        }
        return
    if rest.startswith("attention.v_proj."):
        wb = rest.rsplit(".", 1)[-1]
        pending_kv[(stage, block_idx, wb)] = {
            **pending_kv.get((stage, block_idx, wb), {}),
            "value": value,
        }
        return

    for prefix, smp_prefix, split_at in _BLOCK_REST_RULES:
        if rest.startswith(prefix):
            tail = rest.split(".", split_at)[split_at]
            smp[base + smp_prefix + tail] = value
            return


def _invert_hf_key(
    key: str,
    value: torch.Tensor,
    smp: dict[str, torch.Tensor],
    pending_kv: dict[tuple[str, str, str], dict[str, torch.Tensor]],
) -> None:
    """Map a single HF state-dict entry into ``smp`` / ``pending_kv``."""
    if key.startswith("segformer.stages.") and ".patch_embeddings." in key:
        parts = key.split(".")
        stage = int(parts[2]) + 1
        which = "proj" if parts[4] == "proj" else "norm"
        smp[f"encoder.patch_embed{stage}.{which}.{parts[5]}"] = value
        return

    match = re.match(r"segformer\.stages\.(\d+)\.layer_norm\.(weight|bias)$", key)
    if match:
        stage = int(match.group(1)) + 1
        smp[f"encoder.norm{stage}.{match.group(2)}"] = value
        return

    if key.startswith("segformer.stages.") and ".blocks." in key:
        parts = key.split(".")
        stage = str(int(parts[2]) + 1)
        block_idx = parts[4]
        rest = ".".join(parts[5:])
        base = f"encoder.block{stage}.{block_idx}."
        _invert_encoder_block_rest(
            base,
            rest,
            value,
            smp,
            pending_kv,
            stage,
            block_idx,
        )
        return

    if key.startswith("decode_head.linear_projections."):
        parts = key.split(".")
        hf_i = int(parts[2])
        tia_i = 3 - hf_i
        smp[f"decoder.mlp_stage.{tia_i}.linear.{parts[4]}"] = value
        return

    if key == "decode_head.linear_fuse.weight":
        smp["decoder.fuse_stage.0.weight"] = value
        return

    if key.startswith("decode_head.batch_norm."):
        smp["decoder.fuse_stage.1." + key.split(".", 2)[2]] = value
        return

    if key.startswith("decode_head.classifier."):
        smp["segmentation_head.0." + key.split(".", 2)[2]] = value


def _hf_to_smp_state_dict(
    hf_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Invert HF keys into a minimal SMP-style state dict for tests."""
    smp: dict[str, torch.Tensor] = {}
    pending_kv: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}

    for raw_key, value in hf_state.items():
        _invert_hf_key(_strip_model_prefix(raw_key), value, smp, pending_kv)

    for (stage, block_idx, wb), parts in pending_kv.items():
        smp[f"encoder.block{stage}.{block_idx}.attn.kv.{wb}"] = torch.cat(
            [parts["key"], parts["value"]],
            dim=0,
        )

    return smp


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
    smp = {
        "encoder.block1.0.attn.kv.weight": torch.arange(
            32, dtype=torch.float32
        ).reshape(
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

    assert is_smp_segformer_state_dict(smp)
    remapped = remap_smp_segformer_state_dict(smp)

    key_w = remapped["segformer.stages.0.blocks.0.attention.k_proj.weight"]
    val_w = remapped["segformer.stages.0.blocks.0.attention.v_proj.weight"]
    assert key_w.shape == (4, 4)
    assert val_w.shape == (4, 4)
    assert torch.equal(key_w, smp["encoder.block1.0.attn.kv.weight"][:4])
    assert torch.equal(val_w, smp["encoder.block1.0.attn.kv.weight"][4:])

    assert remapped["decode_head.linear_projections.3.proj.weight"].shape == (16, 32)
    assert remapped["decode_head.linear_projections.0.proj.weight"].shape == (16, 8)
    assert remapped["decode_head.classifier.weight"].shape == (5, 16, 1, 1)


@pytest.mark.parametrize("num_classes", [2, 6])
def test_smp_checkpoint_load_for_variable_classes(num_classes: int) -> None:
    """SMP remapping should work for checkpoints that only differ by class count."""
    model = Segformer(
        encoder_name="mit_b0",
        decoder_segmentation_channels=64,
        classes=num_classes,
        upsampling=4,
    )
    smp = _hf_to_smp_state_dict(model.state_dict())
    assert is_smp_segformer_state_dict(smp)
    assert smp["segmentation_head.0.weight"].shape[0] == num_classes

    fresh = Segformer(
        encoder_name="mit_b0",
        decoder_segmentation_channels=64,
        classes=num_classes,
        upsampling=4,
    )
    incompatible = fresh.load_state_dict(smp, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert fresh(x).shape == (1, num_classes, 64, 64)


def test_native_hf_state_dict_still_loads() -> None:
    """Native Hugging Face-shaped state dicts should load without remapping."""
    model = Segformer(
        encoder_name="mit_b0", classes=3, decoder_segmentation_channels=32
    )
    state = model.state_dict()
    assert not is_smp_segformer_state_dict(state)

    other = Segformer(
        encoder_name="mit_b0",
        classes=3,
        decoder_segmentation_channels=32,
    )
    incompatible = other.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
