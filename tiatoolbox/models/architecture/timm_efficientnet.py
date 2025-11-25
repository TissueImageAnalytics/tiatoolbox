"""EfficientNet Encoder Implementation using timm.

This module provides an implementation of EfficientNet-based encoders for use in
semantic segmentation and other computer vision tasks. It leverages the `timm`
library for model components and adds encoder-specific functionality such as
custom input channels, dilation support, and configurable scaling parameters.

Key Components:
---------------
- patch_first_conv:
    Utility to modify the first convolution layer for arbitrary input channels.
- replace_strides_with_dilation:
    Utility to convert strides into dilations for atrous convolutions.
- EncoderMixin:
    Mixin class adding encoder-specific features like output channels and stride.
- EfficientNetBaseEncoder:
    Base encoder combining EfficientNet backbone with encoder functionality.
- EfficientNetEncoder:
    Configurable EfficientNet encoder supporting depth and channel scaling.
- timm_efficientnet_encoders:
    Dictionary of available EfficientNet encoder configurations and pretrained settings.

Features:
---------
- Supports arbitrary input channels (e.g., grayscale or multi-channel images).
- Allows conversion to dilated versions for semantic segmentation.
- Provides pretrained weights from multiple sources (ImageNet, AdvProp, Noisy Student).
- Implements scaling rules for EfficientNet architecture.

Example:
    >>> from tiatoolbox.models.architecture.timm_efficientnet import EfficientNetEncoder
    >>> encoder = EfficientNetEncoder(
    ...     stage_idxs=[2, 3, 5],
    ...     out_channels=[3, 32, 24, 40, 112, 320],
    ...     channel_multiplier=1.0,
    ...     depth_multiplier=1.0,
    ...     drop_rate=0.2
    ... )
    >>> x = torch.randn(1, 3, 224, 224)
    >>> features = encoder(x)
    >>> [f.shape for f in features]
    [torch.Size([1, 3, 224, 224]), torch.Size([1, 32, 112, 112]), ...]

References:
    - Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking Model Scaling for
      Convolutional Neural Networks." arXiv preprint arXiv:1905.11946 (2019).
      URL: https://arxiv.org/abs/1905.11946

"""

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
    """Update the first convolution layer for a new input channel size.

    This function updates the first convolutional layer of a model to handle
    arbitrary input channels. It optionally reuses pretrained weights or
    initializes weights randomly.

    Args:
        model (nn.Module):
            The neural network model whose first convolution layer will be patched.
        new_in_channels (int):
            Number of input channels for the new first layer.
        default_in_channels (int):
            Original number of input channels. Defaults to 3.
        pretrained (bool):
            Whether to reuse pretrained weights. Defaults to True.

    Notes:
        - If `new_in_channels` == 1 or 2 → reuse original weights.
        - If `new_in_channels` > 3 → initialize weights using Kaiming normal.

    Example:
        >>> patch_first_conv(model, new_in_channels=1, pretrained=True)

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
    """Replace strides with dilation in Conv2d layers.

    Converts convolutional layers to use dilation instead of stride, enabling
    atrous convolutions for semantic segmentation tasks.

    Args:
        module (nn.Module):
            Module containing Conv2d layers to patch.
        dilation_rate (int):
            Dilation rate to apply to all Conv2d layers.

    Example:
        >>> replace_strides_with_dilation(model, dilation_rate=2)

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
    """Mixin class adding encoder-specific functionality.

    Provides methods for:
    - Managing output channels for encoder feature maps.
    - Patching the first convolution for arbitrary input channels.
    - Converting encoder to dilated version for segmentation tasks.

    Attributes:
        _depth (int):
            Encoder depth (number of stages).
        _in_channels (int):
            Number of input channels.
        _output_stride (int):
            Output stride of the encoder.
        _out_channels (list[int]):
            List of output channel dimensions for each depth level.

    Example:
        >>> encoder = EncoderMixin()
        >>> encoder.set_in_channels(1)

    """

    _is_torch_scriptable = True
    _is_torch_exportable = True
    _is_torch_compilable = True

    def __init__(self) -> None:
        """Initialize EncoderMixin with default parameters.

        Sets default values for encoder depth, input channels, output stride,
        and output channel list.

        """
        self._depth = 5
        self._in_channels = 3
        self._output_stride = 32
        self._out_channels: list[int] = []

    @property
    def out_channels(self) -> list[int]:
        """Return output channel dimensions for encoder feature maps.

        Returns:
            list[int]:
                List of output channel dimensions for each depth level.

        Example:
            >>> encoder.out_channels
            ... [3, 32, 64, 128, 256, 512]

        """
        return self._out_channels[: self._depth + 1]

    @property
    def output_stride(self) -> int:
        """Return the effective output stride of the encoder.

        The output stride is the minimum of the configured stride and 2^depth.

        Returns:
            int:
                Effective output stride.

        Example:
            >>> encoder.output_stride
            ... 32

        """
        return min(self._output_stride, 2**self._depth)

    def set_in_channels(self, in_channels: int, *, pretrained: bool = True) -> None:
        """Update the encoder to accept a different number of input channels.

        Args:
            in_channels (int):
                Number of input channels.
            pretrained (bool):
                Whether to use pretrained weights. Defaults to True.

        Example:
            >>> encoder.set_in_channels(1, pretrained=False)

        """
        if in_channels == DEFAULT_IN_CHANNELS:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == DEFAULT_IN_CHANNELS:
            self._out_channels = [in_channels, *self._out_channels[1:]]

        # Type ignore needed because self is a mixin that will be used with nn.Module
        patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)  # type: ignore[arg-type]

    def get_stages(self) -> dict[int, Sequence[torch.nn.Module]]:
        """Return encoder stages for dilation modification.

        This method should be overridden by subclasses to provide stage mappings
        for converting strides to dilations.

        Returns:
            dict[int, Sequence[torch.nn.Module]]:
                Dictionary mapping output stride to corresponding module sequences.

        Raises:
            NotImplementedError:
                If the method is not implemented by the subclass.

        Example:
            >>> stages = encoder.get_stages()

        """
        raise NotImplementedError

    def make_dilated(self, output_stride: int) -> None:
        """Convert encoder to a dilated version for segmentation.

        Args:
            output_stride (int):
                Target output stride (must be 8 or 16).

        Raises:
            ValueError:
                If `output_stride` is not 8 or 16.

        Example:
            >>> encoder.make_dilated(output_stride=16)

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
    """Generate configuration parameters for EfficientNet.

    Args:
        channel_multiplier (float):
            Multiplier for number of channels per layer. Defaults to 1.0.
        depth_multiplier (float):
            Multiplier for number of repeats per stage. Defaults to 1.0.
        drop_rate (float):
            Dropout rate. Defaults to 0.2.

    Reference implementation:
        https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py

    Paper:
        https://arxiv.org/abs/1905.11946

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
        dict[str, Any]:
            Dictionary containing EfficientNet configuration parameters

    Example:
        >>> kwargs = get_efficientnet_kwargs(
        ...  channel_multiplier=1.2,
        ...  depth_multiplier=1.4,
        ... )

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
    """Base class for EfficientNet encoder.

    Combines EfficientNet backbone from `timm` with encoder-specific functionality
    for feature extraction in segmentation and classification tasks.

    Features:
        - Supports configurable depth and output stride.
        - Provides intermediate feature maps for multi-scale processing.
        - Removes classifier for encoder-only usage.

    Raises:
        ValueError:
            If `depth` is not in range [1, 5].

    Example:
        >>> encoder = EfficientNetBaseEncoder(
        ...     stage_idxs=[2, 3, 5],
        ...     out_channels=[3, 32, 24, 40, 112, 320],
        ...     depth=5,
        ...     output_stride=32
        ... )
        >>> x = torch.randn(1, 3, 224, 224)
        >>> features = encoder(x)
        >>> [f.shape for f in features]
        ... [torch.Size([1, 3, 224, 224]), torch.Size([1, 32, 112, 112]), ...]

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
            stage_idxs (list[int]):
                Indices of stages for feature extraction.
            out_channels (list[int]):
                Output channels for each depth level.
            depth (int):
                Encoder depth (1-5). Defaults to 5.
            output_stride (int):
                Output stride of encoder. Defaults to 32.
            **kwargs (dict[str, Any]):
                Additional keyword arguments for EfficientNet initialization.

        Raises:
            ValueError:
                If `depth` is not in range [1, 5].

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
        """Return encoder stages for dilation modification.

        Provides mapping of output strides to corresponding module sequences,
        enabling conversion to dilated versions for segmentation tasks.

        Returns:
            dict[int, Sequence[torch.nn.Module]]:
                Dictionary mapping output stride to module sequences.

        Example:
            >>> stages = encoder.get_stages()
            >>> print(stages.keys())
            ... dict_keys([16, 32])

        """
        return {
            16: [self.blocks[self._stage_idxs[1] : self._stage_idxs[2]]],  # type: ignore[attr-defined]
            32: [self.blocks[self._stage_idxs[2] :]],  # type: ignore[attr-defined]
        }

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through EfficientNet encoder.

        Extracts feature maps from multiple stages of the encoder for use in
        decoder networks or multi-scale processing.

        Args:
            x (torch.Tensor):
                Input tensor of shape (N, C, H, W).

        Returns:
            list[torch.Tensor]:
                List of feature maps from different encoder depths.

        Example:
            >>> x = torch.randn(1, 3, 224, 224)
            >>> features = encoder(x)
            >>> len(features)
            ... 6

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
        self,
        state_dict: Mapping[str, Any],
        **kwargs: bool,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load state dictionary, excluding classifier weights.

        Removes classifier weights from the state dictionary before loading,
        as the encoder does not include a classification head.

        Args:
            state_dict (Mapping[str, Any]):
                State dictionary to load.
            **kwargs (bool):
                Additional keyword arguments for `load_state_dict`.

        Returns:
            torch.nn.modules.module._IncompatibleKeys:
                Result of parent class `load_state_dict` method.

        Example:
            >>> encoder.load_state_dict(torch.load("efficientnet_weights.pth"))

        """
        # Create a mutable copy of the state dict to modify
        state_dict_copy = dict(state_dict)
        state_dict_copy.pop("classifier.bias", None)
        state_dict_copy.pop("classifier.weight", None)
        return super().load_state_dict(state_dict_copy, **kwargs)


class EfficientNetEncoder(EfficientNetBaseEncoder):
    """EfficientNet encoder with configurable scaling parameters.

    This class extends `EfficientNetBaseEncoder` to provide scaling options
    for depth and channel multipliers, enabling flexible encoder configurations
    for segmentation and classification tasks.

    Features:
        - Supports depth and channel scaling.
        - Provides pretrained weights for multiple variants.
        - Outputs multi-scale feature maps for downstream tasks.

    Example:
        >>> encoder = EfficientNetEncoder(
        ...     stage_idxs=[2, 3, 5],
        ...     out_channels=[3, 32, 24, 40, 112, 320],
        ...     channel_multiplier=1.0,
        ...     depth_multiplier=1.0,
        ...     drop_rate=0.2
        ... )
        >>> x = torch.randn(1, 3, 224, 224)
        >>> features = encoder(x)
        >>> [f.shape for f in features]
        ... [torch.Size([1, 3, 224, 224]), torch.Size([1, 32, 112, 112]), ...]

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

        Creates an EfficientNet encoder with configurable scaling parameters
        for depth and channel multipliers.

        Args:
            stage_idxs (list[int]):
                Indices of stages for feature extraction.
            out_channels (list[int]):
                Output channels for each depth level.
            depth (int):
                Encoder depth (1-5). Defaults to 5.
            channel_multiplier (float):
                Channel scaling factor. Defaults to 1.0.
            depth_multiplier (float):
                Depth scaling factor. Defaults to 1.0.
            drop_rate (float):
                Dropout rate. Defaults to 0.2.
            output_stride (int):
                Output stride of encoder. Defaults to 32.

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
