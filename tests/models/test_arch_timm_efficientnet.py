"""Unit tests for timm EfficientNet encoder helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import pytest
import torch
from torch import nn

from tiatoolbox.models.architecture import timm_efficientnet as effnet_mod
from tiatoolbox.models.architecture.timm_efficientnet import (
    DEFAULT_IN_CHANNELS,
    EfficientNetEncoder,
    EncoderMixin,
)


class DummyEncoder(nn.Module, EncoderMixin):
    """Lightweight encoder for testing mixin behavior."""

    def __init__(self) -> None:
        """Initialize EncoderMixin for testing."""
        nn.Module.__init__(self)
        EncoderMixin.__init__(self)
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(4, 4, 3)
        self._out_channels = [DEFAULT_IN_CHANNELS, 4, 8]
        self._depth = 2

    def get_stages(self) -> dict[int, Sequence[torch.nn.Module]]:
        """Get stages for dilation modification.

        Returns:
            Dictionary with keys as output stride and values as list of modules.
        """
        return {16: [self.conv], 32: [self.conv32]}


def test_encoder_mixin_properties_and_set_in_channels() -> None:
    """EncoderMixin should expose out_channels/output_stride and patch convs."""
    # use dummy encoder to check property logic
    encoder = DummyEncoder()
    assert encoder.out_channels == [3, 4, 8]
    # adjust internals to check min logic in output_stride
    encoder._output_stride = 4
    encoder._depth = 3
    assert encoder.output_stride == 4  # min(output_stride, 2**depth)

    # calling set_in_channels should patch first conv and update bookkeeping
    encoder.set_in_channels(5, pretrained=False)
    assert encoder._in_channels == 5
    assert encoder.out_channels[0] == 5
    assert encoder.conv.in_channels == 5


def test_set_in_channels_noop_for_default() -> None:
    """Calling with DEFAULT_IN_CHANNELS should skip patching."""
    encoder = DummyEncoder()
    encoder.set_in_channels(DEFAULT_IN_CHANNELS, pretrained=True)
    assert encoder._in_channels == DEFAULT_IN_CHANNELS


def test_set_in_channels_modify_out_channels() -> None:
    """First output channels should change when in_channels is modified."""
    encoder = DummyEncoder()
    encoder._out_channels[0] = DEFAULT_IN_CHANNELS

    encoder.set_in_channels(5, pretrained=False)

    assert encoder._out_channels[0] == 5
    assert encoder._in_channels == 5


def test_set_in_channels_preserves_custom_out_channels() -> None:
    """When first out_channels is customized, set_in_channels should not override."""
    encoder = DummyEncoder()
    encoder._out_channels[0] = 7

    encoder.set_in_channels(5, pretrained=False)

    assert encoder._out_channels[0] == 7
    assert encoder._in_channels == 5
    assert encoder.conv.in_channels == 5


def test_encoder_mixin_make_dilated_and_validation() -> None:
    """make_dilated should error on invalid stride and patch convs otherwise."""
    encoder = DummyEncoder()

    # invalid stride raises
    with pytest.raises(ValueError, match="Output stride should be 16 or 8"):
        encoder.make_dilated(output_stride=4)

    # valid stride should touch both stage groups
    encoder.make_dilated(output_stride=8)
    conv16, conv32 = encoder.get_stages()[16][0], encoder.get_stages()[32][0]
    assert conv16.stride == (1, 1)
    assert conv16.dilation == (2, 2)
    assert conv32.stride == (1, 1)
    assert conv32.dilation == (4, 4)


def test_make_dilated_skips_stages_below_output_stride() -> None:
    """Stages at or below the target stride should be left untouched."""
    encoder = DummyEncoder()
    encoder.conv.stride = (2, 2)  # stage_stride == 16, so should be skipped
    encoder.conv.dilation = (1, 1)

    encoder.make_dilated(output_stride=16)

    # stage at stride 16 skipped
    assert encoder.conv.stride == (2, 2)
    assert encoder.conv.dilation == (1, 1)

    # stage at stride 32 modified
    conv32 = encoder.get_stages()[32][0]
    assert conv32.dilation == (2, 2)
    assert conv32.padding == (2, 2)


def test_efficientnet_encoder_get_stages_splits_blocks() -> None:
    """Test get_stages for dilation modification."""
    encoder = EfficientNetEncoder(
        stage_idxs=[1, 2, 4],
        out_channels=[3, 8, 16, 32, 64, 128],
        depth=3,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
    )
    stages = encoder.get_stages()
    assert len(stages) == 2
    assert stages.keys() == {16, 32}


def test_get_efficientnet_kwargs_shapes_and_values() -> None:
    """get_efficientnet_kwargs should produce expected keys and scaling."""
    # confirm output contains decoded blocks and scaled channels
    kwargs = effnet_mod.get_efficientnet_kwargs(
        channel_multiplier=1.2, depth_multiplier=1.4, drop_rate=0.3
    )
    assert kwargs.get("block_args")
    assert kwargs["num_features"] == effnet_mod.round_channels(1280, 1.2, 8, None)
    assert kwargs["drop_rate"] == 0.3


def test_efficientnet_encoder_depth_validation_and_forward() -> None:
    """EfficientNetEncoder should validate depth and run forward returning features."""
    # invalid depth should fail fast
    with pytest.raises(
        ValueError, match=r"EfficientNetEncoder depth should be in range\s+\[1, 5\]"
    ):
        EfficientNetEncoder(
            stage_idxs=[2, 3, 5],
            out_channels=[3, 32, 24, 40, 112, 320],
            depth=6,
        )

    # build shallow encoder and run a forward pass
    encoder = EfficientNetEncoder(
        stage_idxs=[2, 3, 5],
        out_channels=[3, 32, 24, 40, 112, 320],
        depth=3,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
    )
    x = torch.randn(1, 3, 32, 32)
    features = encoder(x)
    assert len(features) == encoder._depth + 1
    assert torch.equal(features[0], x)
    # cover depth-gated forward branches up to depth 3
    assert features[1].shape[1] == 32
    assert features[2].shape[1] == 24
    assert features[3].shape[1] == 40

    encoder = EfficientNetEncoder(
        stage_idxs=[2, 3, 5],
        out_channels=[3, 32, 24, 40, 112, 320],
        depth=1,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
    )
    x = torch.randn(1, 3, 32, 32)
    features = encoder(x)
    assert len(features) == encoder._depth + 1
    assert torch.equal(features[0], x)
    assert features[1].shape[1] == 32

    encoder = EfficientNetEncoder(
        stage_idxs=[2, 3, 5],
        out_channels=[3, 32, 24, 40, 112, 320],
        depth=2,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
    )
    x = torch.randn(1, 3, 32, 32)
    features = encoder(x)
    assert len(features) == encoder._depth + 1
    assert torch.equal(features[0], x)
    assert features[1].shape[1] == 32
    assert features[2].shape[1] == 24


def test_efficientnet_encoder_load_state_dict_drops_classifier_keys() -> None:
    """Loading state dict with classifier keys should drop them silently."""
    # ensure classifier keys are dropped before loading into the model
    encoder = EfficientNetEncoder(
        stage_idxs=[2, 3, 5],
        out_channels=[3, 32, 24, 40, 112, 320],
        depth=3,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
    )
    extended_state = dict(encoder.state_dict())
    extended_state["classifier.bias"] = torch.tensor([1.0])
    extended_state["classifier.weight"] = torch.tensor([[1.0]])
    load_result = encoder.load_state_dict(extended_state, strict=True)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys
    assert "classifier.bias" not in encoder.state_dict()
    assert "classifier.weight" not in encoder.state_dict()


def test_forward_skips_depth_1_branch() -> None:
    """Tests forward skips depth 1 branch."""
    # Create your real encoder
    encoder = EfficientNetEncoder(
        stage_idxs=[2, 3, 5],
        out_channels=[3, 32, 24, 40, 112, 320],
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        drop_rate=0.2,
    )  # whatever args your class needs

    # Force depth < 1 so the branch is skipped
    encoder._depth = 0

    # Input tensor
    x = torch.randn(1, 3, 32, 32)

    # Run forward
    features = encoder.forward(x)

    # Only the original input should be returned
    assert len(features) == 1
    assert torch.equal(features[0], x)
