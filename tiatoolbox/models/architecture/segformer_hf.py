"""Hugging Face SegFormer helpers: MiT configs and legacy checkpoint remapping.

Architecture weights are loaded into ``transformers.SegformerForSemanticSegmentation``.
This module remaps older SMP-style SegFormer state dicts.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, MutableMapping

    import torch

# Number of hierarchical MiT encoder / decoder projection stages.
NUM_SEGFORMER_STAGES = 4

MIT_ENCODER_CONFIGS: dict[str, dict[str, object]] = {
    "mit_b0": {
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [32, 64, 160, 256],
        "num_attention_heads": [1, 2, 5, 8],
        "sr_ratios": [8, 4, 2, 1],
        "patch_sizes": [7, 3, 3, 3],
        "strides": [4, 2, 2, 2],
        "mlp_ratios": [4, 4, 4, 4],
    },
    "mit_b1": {
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [64, 128, 320, 512],
        "num_attention_heads": [1, 2, 5, 8],
        "sr_ratios": [8, 4, 2, 1],
        "patch_sizes": [7, 3, 3, 3],
        "strides": [4, 2, 2, 2],
        "mlp_ratios": [4, 4, 4, 4],
    },
    "mit_b2": {
        "depths": [3, 4, 6, 3],
        "hidden_sizes": [64, 128, 320, 512],
        "num_attention_heads": [1, 2, 5, 8],
        "sr_ratios": [8, 4, 2, 1],
        "patch_sizes": [7, 3, 3, 3],
        "strides": [4, 2, 2, 2],
        "mlp_ratios": [4, 4, 4, 4],
    },
    "mit_b3": {
        "depths": [3, 4, 18, 3],
        "hidden_sizes": [64, 128, 320, 512],
        "num_attention_heads": [1, 2, 5, 8],
        "sr_ratios": [8, 4, 2, 1],
        "patch_sizes": [7, 3, 3, 3],
        "strides": [4, 2, 2, 2],
        "mlp_ratios": [4, 4, 4, 4],
    },
    "mit_b4": {
        "depths": [3, 8, 27, 3],
        "hidden_sizes": [64, 128, 320, 512],
        "num_attention_heads": [1, 2, 5, 8],
        "sr_ratios": [8, 4, 2, 1],
        "patch_sizes": [7, 3, 3, 3],
        "strides": [4, 2, 2, 2],
        "mlp_ratios": [4, 4, 4, 4],
    },
    "mit_b5": {
        "depths": [3, 6, 40, 3],
        "hidden_sizes": [64, 128, 320, 512],
        "num_attention_heads": [1, 2, 5, 8],
        "sr_ratios": [8, 4, 2, 1],
        "patch_sizes": [7, 3, 3, 3],
        "strides": [4, 2, 2, 2],
        "mlp_ratios": [4, 4, 4, 4],
    },
}

# Legacy block suffix -> (HF suffix template with ``{tail}``, dots to split for tail).
_BLOCK_SUFFIX_RULES: tuple[tuple[str, str, int], ...] = (
    ("norm1.", "layer_norm_1.{tail}", 1),
    ("norm2.", "layer_norm_2.{tail}", 1),
    ("attn.q.", "attention.self.query.{tail}", 2),
    ("attn.proj.", "attention.output.dense.{tail}", 2),
    ("attn.sr.", "attention.self.sr.{tail}", 2),
    ("attn.norm.", "attention.self.layer_norm.{tail}", 2),
    ("mlp.fc1.", "mlp.dense1.{tail}", 2),
    ("mlp.fc2.", "mlp.dense2.{tail}", 2),
    ("mlp.dwconv.dwconv.", "mlp.dwconv.dwconv.{tail}", 3),
)


def is_legacy_segformer_state_dict(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Return True if ``state_dict`` uses TIA/SMP SegFormer key names."""
    return any(
        key.startswith(
            ("encoder.patch_embed", "decoder.mlp_stage", "segmentation_head.")
        )
        for key in state_dict
    )


def remap_legacy_segformer_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap a TIA/SMP SegFormer checkpoint to Hugging Face key names.

    Classifier / head tensor shapes are passed through unchanged, so checkpoints that
    only differ by ``num_labels`` (output classes) are supported when the model is
    constructed with a matching ``classes`` value.
    """
    out: dict[str, torch.Tensor] = {}
    mlp_indices = {
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"decoder\.mlp_stage\.(\d+)\.", key)) is not None
    }
    num_mlp_stages = max(mlp_indices) + 1 if mlp_indices else NUM_SEGFORMER_STAGES

    for key, value in state_dict.items():
        match = re.match(
            r"encoder\.patch_embed(\d)\.(proj|norm)\.(weight|bias)$",
            key,
        )
        if match:
            stage = int(match.group(1)) - 1
            which = "proj" if match.group(2) == "proj" else "layer_norm"
            out[
                f"segformer.encoder.patch_embeddings.{stage}.{which}.{match.group(3)}"
            ] = value
            continue

        match = re.match(r"encoder\.norm(\d)\.(weight|bias)$", key)
        if match:
            stage = int(match.group(1)) - 1
            out[f"segformer.encoder.layer_norm.{stage}.{match.group(2)}"] = value
            continue

        match = re.match(r"encoder\.block(\d)\.(\d+)\.(.*)$", key)
        if match:
            stage = int(match.group(1)) - 1
            block_idx = match.group(2)
            rest = match.group(3)
            base = f"segformer.encoder.block.{stage}.{block_idx}."
            _remap_encoder_block_param(base, rest, value, out)
            continue

        match = re.match(
            r"decoder\.mlp_stage\.(\d+)\.linear\.(weight|bias)$",
            key,
        )
        if match:
            tia_index = int(match.group(1))
            hf_index = num_mlp_stages - 1 - tia_index
            out[f"decode_head.linear_c.{hf_index}.proj.{match.group(2)}"] = value
            continue

        if key == "decoder.fuse_stage.0.weight":
            out["decode_head.linear_fuse.weight"] = value
            continue

        if key.startswith("decoder.fuse_stage.1."):
            out["decode_head.batch_norm." + key.split(".", 3)[3]] = value
            continue

        if key.startswith("segmentation_head.0."):
            out["decode_head.classifier." + key.split(".", 2)[2]] = value
            continue

        msg = f"Unhandled legacy SegFormer state dict key: {key}"
        raise KeyError(msg)

    return out


def _remap_encoder_block_param(
    base: str,
    rest: str,
    value: torch.Tensor,
    out: MutableMapping[str, torch.Tensor],
) -> None:
    """Map one encoder block parameter from legacy naming into ``out``."""
    if rest.startswith("attn.kv."):
        weight_or_bias = rest.split(".", 2)[2]
        half = value.shape[0] // 2
        out[base + f"attention.self.key.{weight_or_bias}"] = value[:half].clone()
        out[base + f"attention.self.value.{weight_or_bias}"] = value[half:].clone()
        return

    for prefix, template, split_at in _BLOCK_SUFFIX_RULES:
        if rest.startswith(prefix):
            tail = rest.split(".", split_at)[split_at]
            out[base + template.format(tail=tail)] = value
            return

    msg = f"Unhandled legacy SegFormer block parameter: {rest}"
    raise KeyError(msg)
