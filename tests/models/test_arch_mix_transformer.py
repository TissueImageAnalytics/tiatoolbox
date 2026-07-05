"""Unit tests for Mix Transformer architecture components."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

import pytest
import torch

from tiatoolbox.models.architecture.mix_transformer import (
    MAX_ENCODER_DEPTH,
    MIN_ENCODER_DEPTH,
    Attention,
    Block,
    DWConvBlock,
    LayerNorm,
    MixVisionTransformer,
    MixVisionTransformerEncoder,
    Mlp,
    OverlapPatchEmbed,
)


def tiny_transformer_kwargs() -> dict[str, object]:
    """Return a compact model configuration to keep tests fast."""
    return {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "embed_dims": [8, 16, 24, 32],
        "num_heads": [1, 2, 4, 8],
        "mlp_ratios": [2.0, 2.0, 2.0, 2.0],
        "depths": [1, 1, 1, 1],
        "sr_ratios": [8, 4, 2, 1],
        "drop_path_rate": 0.1,
    }


def test_layer_norm_handles_3d_and_4d_inputs() -> None:
    """LayerNorm should normalize both token and image tensor layouts."""
    norm = LayerNorm(6)

    image_tensor = torch.randn(2, 6, 8, 8)
    token_tensor = torch.randn(2, 64, 6)

    out_image = norm(image_tensor)
    out_token = norm(token_tensor)

    assert out_image.shape == image_tensor.shape
    assert out_token.shape == token_tensor.shape


def test_mlp_forward_shape() -> None:
    """MLP should preserve token count and produce expected channel width."""
    mlp = Mlp(in_features=8, hidden_features=16, out_features=8)
    x = torch.randn(2, 64, 8)
    output = mlp(x, height=8, width=8)
    assert output.shape == (2, 64, 8)


def test_attention_invalid_head_partition_raises() -> None:
    """Attention should reject channel dimensions not divisible by heads."""
    with pytest.raises(ValueError, match="should be divided by num_heads"):
        Attention(dim=10, num_heads=3)


@pytest.mark.parametrize("sr_ratio", [1, 2])
def test_attention_forward_across_sr_ratio_paths(sr_ratio: int) -> None:
    """Attention should run for both direct and spatial-reduction branches."""
    attention = Attention(dim=8, num_heads=2, sr_ratio=sr_ratio)
    x = torch.randn(2, 64, 8)
    output = attention(x, height=8, width=8)
    assert output.shape == (2, 64, 8)


def test_block_forward_shape() -> None:
    """Block should preserve spatial size and channel dimension."""
    block = Block(dim=8, num_heads=2, mlp_ratio=2.0, drop_path=0.0, sr_ratio=1)
    x = torch.randn(1, 8, 8, 8)
    output = block(x)
    assert output.shape == (1, 8, 8, 8)


def test_overlap_patch_embed_shape() -> None:
    """Overlap patch embedding should produce a feature map tensor."""
    embed = OverlapPatchEmbed(
        img_size=32,
        patch_size=7,
        stride=4,
        in_chans=3,
        embed_dim=8,
    )
    x = torch.randn(1, 3, 32, 32)
    output = embed(x)
    assert output.shape == (1, 8, 8, 8)


def test_mix_vision_transformer_forward_features_and_forward() -> None:
    """Backbone should return four staged feature maps."""
    model = MixVisionTransformer(**tiny_transformer_kwargs())
    x = torch.randn(1, 3, 32, 32)

    features = model.forward_features(x)
    forwarded = model(x)

    assert len(features) == 4
    assert len(forwarded) == 4
    assert torch.equal(features[0], forwarded[0])
    assert features[0].shape[1] == 8
    assert features[1].shape[1] == 16
    assert features[2].shape[1] == 24
    assert features[3].shape[1] == 32


def test_mix_vision_transformer_helpers_and_classifier_paths() -> None:
    """Helper methods should expose expected metadata and edge behaviors."""
    model = MixVisionTransformer(**tiny_transformer_kwargs())

    no_decay = model.no_weight_decay()
    assert {
        "pos_embed1",
        "pos_embed2",
        "pos_embed3",
        "pos_embed4",
        "cls_token",
    } <= no_decay


def test_reset_drop_path_updates_drop_probabilities() -> None:
    """reset_drop_path should update drop probabilities across all blocks."""
    model = MixVisionTransformer(**tiny_transformer_kwargs())
    model.reset_drop_path(0.2)

    assert hasattr(model.block1[0].drop_path, "drop_prob")
    assert model.block1[0].drop_path.drop_prob == pytest.approx(0.0)
    assert model.block4[0].drop_path.drop_prob == pytest.approx(0.2)


def test_dwconv_forward_shape() -> None:
    """Depthwise convolution token mixer should preserve token dimensions."""
    dwconv = DWConvBlock(dim=8)
    x = torch.randn(2, 64, 8)
    output = dwconv(x, height=8, width=8)
    assert output.shape == (2, 64, 8)


def test_mix_vision_transformer_encoder_depth_validation() -> None:
    """Encoder depth should be validated at construction."""
    kwargs = tiny_transformer_kwargs()
    out_channels = [3, 0, 8, 16, 24, 32]

    with pytest.raises(ValueError, match="depth should be in range"):
        MixVisionTransformerEncoder(
            out_channels=out_channels,
            depth=MIN_ENCODER_DEPTH - 1,
            **kwargs,
        )

    with pytest.raises(ValueError, match="depth should be in range"):
        MixVisionTransformerEncoder(
            out_channels=out_channels,
            depth=MAX_ENCODER_DEPTH + 1,
            **kwargs,
        )


def test_mix_vision_transformer_encoder_stages_forward_and_load_state_dict() -> None:
    """Encoder should expose stages, depth-gated branches, and state loading."""
    kwargs = tiny_transformer_kwargs()
    encoder = MixVisionTransformerEncoder(
        out_channels=[3, 0, 8, 16, 24, 32],
        depth=5,
        output_stride=32,
        **kwargs,
    )

    stages = encoder.get_stages()
    assert stages.keys() == {16, 32}

    x = torch.randn(1, 3, 32, 32)

    features = encoder(x)
    assert len(features) == 6
    assert features[1].shape[1] == 0

    encoder._depth = 1
    assert len(encoder(x)) == 2

    encoder._depth = 2
    assert len(encoder(x)) == 3

    encoder._depth = 3
    assert len(encoder(x)) == 4

    encoder._depth = 4
    assert len(encoder(x)) == 5

    state_dict: Mapping[str, torch.Tensor] = dict(encoder.state_dict())
    mutable_state = dict(state_dict)
    mutable_state["head.weight"] = torch.randn(1)
    mutable_state["head.bias"] = torch.randn(1)
    load_result = encoder.load_state_dict(mutable_state)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys
