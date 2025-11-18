"""Defines EfficientNet encoder using timm library."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

import torch
from timm.layers.activations import Swish
from timm.models._efficientnet_builder import decode_arch_def, round_channels
from timm.models.efficientnet import EfficientNet
from torch import nn

MAX_DEPTH = 5
MIN_DEPTH = 1
DEFAULT_IN_CHANNELS = 3


def patch_first_conv(
    model: nn.Module,
    new_in_channels: int,
    default_in_channels: int = 3,
    *,
    pretrained: bool = True,
) -> None:
    """Change first convolution layer input channels.

    Args:
        model: The neural network model to patch.
        new_in_channels: Number of input channels for the new first layer.
        default_in_channels: Original number of input channels. Defaults to 3.
        pretrained: Whether to reuse pretrained weights. Defaults to True.

    Note:
        In case:
        - in_channels == 1 or in_channels == 2 -> reuse original weights
        - in_channels > 3 -> make random kaiming normal initialization
    """
    # get first conv
    conv_module: nn.Conv2d | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            conv_module = module
            break

    if conv_module is None:
        return

    weight = conv_module.weight.detach()
    conv_module.in_channels = new_in_channels

    if not pretrained:
        conv_module.weight = nn.parameter.Parameter(
            torch.Tensor(
                conv_module.out_channels,
                new_in_channels // conv_module.groups,
                *conv_module.kernel_size,
            )
        )
        conv_module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        conv_module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(
            conv_module.out_channels,
            new_in_channels // conv_module.groups,
            *conv_module.kernel_size,
        )

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        conv_module.weight = nn.parameter.Parameter(new_weight)


def replace_strides_with_dilation(module: nn.Module, dilation_rate: int) -> None:
    """Patch Conv2d modules replacing strides with dilation.

    Args:
        module: The module containing Conv2d layers to patch.
        dilation_rate: The dilation rate to apply.
    """
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, _ = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

            # Workaround for EfficientNet
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()  # type: ignore[attr-defined]


class EncoderMixin:
    """Add encoder functionality.

    Such as:
    - output channels specification of feature tensors (produced by encoder)
    - patching first convolution for arbitrary input channels
    """

    _is_torch_scriptable = True
    _is_torch_exportable = True
    _is_torch_compilable = True

    def __init__(self) -> None:
        """Initialize EncoderMixin with default parameters."""
        self._depth = 5
        self._in_channels = 3
        self._output_stride = 32
        self._out_channels: list[int] = []

    @property
    def out_channels(self) -> list[int]:
        """Return channels dimensions for each tensor of forward output of encoder.

        Returns:
            List of output channel dimensions for each depth level.
        """
        return self._out_channels[: self._depth + 1]

    @property
    def output_stride(self) -> int:
        """Return the output stride of the encoder.

        Returns:
            The minimum of configured output stride and 2^depth.
        """
        return min(self._output_stride, 2**self._depth)

    def set_in_channels(self, in_channels: int, *, pretrained: bool = True) -> None:
        """Change first convolution channels.

        Args:
            in_channels: Number of input channels.
            pretrained: Whether to use pretrained weights. Defaults to True.
        """
        if in_channels == DEFAULT_IN_CHANNELS:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == DEFAULT_IN_CHANNELS:
            self._out_channels = [in_channels, *self._out_channels[1:]]

        # Type ignore needed because self is a mixin that will be used with nn.Module
        patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)  # type: ignore[arg-type]

    def get_stages(self) -> dict[int, Sequence[torch.nn.Module]]:
        """Get stages for dilation modification.

        Override this method in your implementation.

        Returns:
            Dictionary with keys as output stride and values as list of modules.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def make_dilated(self, output_stride: int) -> None:
        """Convert encoder to dilated version.

        Args:
            output_stride: Target output stride (8 or 16).

        Raises:
            ValueError: If output_stride is not 8 or 16.
        """
        if output_stride not in [8, 16]:
            msg = f"Output stride should be 16 or 8, got {output_stride}."
            raise ValueError(msg)

        stages = self.get_stages()
        for stage_stride, stage_modules in stages.items():
            if stage_stride <= output_stride:
                continue

            dilation_rate = stage_stride // output_stride
            for module in stage_modules:
                replace_strides_with_dilation(module, dilation_rate)


def get_efficientnet_kwargs(
    channel_multiplier: float = 1.0,
    depth_multiplier: float = 1.0,
    drop_rate: float = 0.2,
) -> dict[str, Any]:
    """Create EfficientNet model kwargs.

    Reference implementation:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py

    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet parameters:
    - 'efficientnet-b0': (1.0, 1.0, 224, 0.2)
    - 'efficientnet-b1': (1.0, 1.1, 240, 0.2)
    - 'efficientnet-b2': (1.1, 1.2, 260, 0.3)
    - 'efficientnet-b3': (1.2, 1.4, 300, 0.3)
    - 'efficientnet-b4': (1.4, 1.8, 380, 0.4)
    - 'efficientnet-b5': (1.6, 2.2, 456, 0.4)
    - 'efficientnet-b6': (1.8, 2.6, 528, 0.5)
    - 'efficientnet-b7': (2.0, 3.1, 600, 0.5)
    - 'efficientnet-b8': (2.2, 3.6, 672, 0.5)
    - 'efficientnet-l2': (4.3, 5.3, 800, 0.5)

    Args:
        channel_multiplier: Multiplier to number of channels per layer. Defaults to 1.0.
        depth_multiplier: Multiplier to number of repeats per stage. Defaults to 1.0.
        drop_rate: Dropout rate. Defaults to 0.2.

    Returns:
        Dictionary containing model configuration parameters.
    """
    arch_def = [
        ["ds_r1_k3_s1_e1_c16_se0.25"],
        ["ir_r2_k3_s2_e6_c24_se0.25"],
        ["ir_r2_k5_s2_e6_c40_se0.25"],
        ["ir_r3_k3_s2_e6_c80_se0.25"],
        ["ir_r3_k5_s1_e6_c112_se0.25"],
        ["ir_r4_k5_s2_e6_c192_se0.25"],
        ["ir_r1_k3_s1_e6_c320_se0.25"],
    ]
    return {
        "block_args": decode_arch_def(arch_def, depth_multiplier),
        "num_features": round_channels(1280, channel_multiplier, 8, None),
        "stem_size": 32,
        "round_chs_fn": partial(round_channels, multiplier=channel_multiplier),
        "act_layer": Swish,
        "drop_rate": drop_rate,
        "drop_path_rate": 0.2,
    }


class EfficientNetBaseEncoder(EfficientNet, EncoderMixin):
    """EfficientNet encoder base class.

    Combines EfficientNet architecture with encoder functionality.
    """

    def __init__(
        self,
        stage_idxs: list[int],
        out_channels: list[int],
        depth: int = 5,
        output_stride: int = 32,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize EfficientNetBaseEncoder.

        Args:
            stage_idxs: Indices of stages for feature extraction.
            out_channels: Output channels for each depth level.
            depth: Encoder depth (1-5). Defaults to 5.
            output_stride: Output stride of encoder. Defaults to 32.
            **kwargs: Additional keyword arguments for EfficientNet.

        Raises:
            ValueError: If depth is not in range [1, 5].
        """
        if depth > MAX_DEPTH or depth < MIN_DEPTH:
            msg = f"{self.__class__.__name__} depth should be in range \
            [1, 5], got {depth}"
            raise ValueError(msg)
        super().__init__(**kwargs)

        self._stage_idxs = stage_idxs
        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        self._output_stride = output_stride

        del self.classifier

    def get_stages(self) -> dict[int, Sequence[torch.nn.Module]]:
        """Get stages for dilation modification.

        Returns:
            Dictionary mapping output strides to corresponding module sequences.
        """
        return {
            16: [self.blocks[self._stage_idxs[1] : self._stage_idxs[2]]],  # type: ignore[attr-defined]
            32: [self.blocks[self._stage_idxs[2] :]],  # type: ignore[attr-defined]
        }

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:  # type: ignore[override]
        """Forward pass through encoder.

        Args:
            x: Input tensor.

        Returns:
            List of feature tensors from different encoder depths.
        """
        features = [x]

        if self._depth >= 1:
            x = self.conv_stem(x)  # type: ignore[attr-defined]
            x = self.bn1(x)  # type: ignore[attr-defined]
            features.append(x)

        if self._depth >= 2:  # noqa: PLR2004
            x = self.blocks[0](x)  # type: ignore[attr-defined]
            x = self.blocks[1](x)  # type: ignore[attr-defined]
            features.append(x)

        if self._depth >= 3:  # noqa: PLR2004
            x = self.blocks[2](x)  # type: ignore[attr-defined]
            features.append(x)

        if self._depth >= 4:  # noqa: PLR2004
            x = self.blocks[3](x)  # type: ignore[attr-defined]
            x = self.blocks[4](x)  # type: ignore[attr-defined]
            features.append(x)

        if self._depth >= 5:  # noqa: PLR2004
            x = self.blocks[5](x)  # type: ignore[attr-defined]
            x = self.blocks[6](x)  # type: ignore[attr-defined]
            features.append(x)

        return features

    def load_state_dict(
        self, state_dict: Mapping[str, Any], **kwargs: bool
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load state dictionary, excluding classifier weights.

        Args:
            state_dict: State dictionary to load.
            **kwargs: Additional keyword arguments for load_state_dict.

        Returns:
            Result of parent class load_state_dict method.
        """
        # Create a mutable copy of the state dict to modify
        state_dict_copy = dict(state_dict)
        state_dict_copy.pop("classifier.bias", None)
        state_dict_copy.pop("classifier.weight", None)
        return super().load_state_dict(state_dict_copy, **kwargs)


class EfficientNetEncoder(EfficientNetBaseEncoder):
    """EfficientNet encoder with configurable scaling parameters.

    Provides a configurable EfficientNet encoder that can be scaled
    in terms of depth and channel multipliers.
    """

    def __init__(
        self,
        stage_idxs: list[int],
        out_channels: list[int],
        depth: int = 5,
        channel_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        drop_rate: float = 0.2,
        output_stride: int = 32,
    ) -> None:
        """Initialize EfficientNetEncoder.

        Args:
            stage_idxs: Indices of stages for feature extraction.
            out_channels: Output channels for each depth level.
            depth: Encoder depth (1-5). Defaults to 5.
            channel_multiplier: Channel scaling factor. Defaults to 1.0.
            depth_multiplier: Depth scaling factor. Defaults to 1.0.
            drop_rate: Dropout rate. Defaults to 0.2.
            output_stride: Output stride of encoder. Defaults to 32.
        """
        kwargs = get_efficientnet_kwargs(
            channel_multiplier, depth_multiplier, drop_rate
        )
        super().__init__(
            stage_idxs=stage_idxs,
            depth=depth,
            out_channels=out_channels,
            output_stride=output_stride,
            **kwargs,
        )


timm_efficientnet_encoders = {
    "timm-efficientnet-b0": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-efficientnet-b0.imagenet",
                "revision": "8419e9cc19da0b68dcd7bb12f19b7c92407ad7c4",
            },
            "advprop": {
                "repo_id": "smp-hub/timm-efficientnet-b0.advprop",
                "revision": "a5870af2d24ce79e0cc7fae2bbd8e0a21fcfa6d8",
            },
            "noisy-student": {
                "repo_id": "smp-hub/timm-efficientnet-b0.noisy-student",
                "revision": "bea8b0ff726a50e48774d2d360c5fb1ac4815836",
            },
        },
        "params": {
            "out_channels": [3, 32, 24, 40, 112, 320],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 1.0,
            "depth_multiplier": 1.0,
            "drop_rate": 0.2,
        },
    },
}
