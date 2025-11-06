"""Define Unet++ architecture from Segmentation Models Pytorch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from tiatoolbox.models.models_abc import ModelABC
from tiatoolbox.models.architecture.timm_universal import TimmUniversalEncoder


class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        self.activation: nn.Module
        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        else:
            self.activation = nn.Identity(**params)
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class SegmentationHead(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):
    def __init__(
        self, in_channels, classes, pooling="avg", dropout=0.2, activation=None
    ):
        if pooling not in ("max", "avg"):
            raise ValueError(
                "Pooling should be one of ('max', 'avg'), got {}.".format(pooling)
            )
        pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)

def get_norm_layer(
    use_norm: Union[bool, str, Dict[str, Any]], out_channels: int
) -> nn.Module:
    supported_norms = ("inplace", "batchnorm", "identity", "layernorm", "instancenorm")

    # Step 1. Convert tot dict representation

    ## Check boolean
    if use_norm is True:
        norm_params = {"type": "batchnorm"}
    elif use_norm is False:
        norm_params = {"type": "identity"}

    ## Check string
    elif isinstance(use_norm, str):
        norm_str = use_norm.lower()
        if norm_str == "inplace":
            norm_params = {
                "type": "inplace",
                "activation": "leaky_relu",
                "activation_param": 0.0,
            }
        elif norm_str in supported_norms:
            norm_params = {"type": norm_str}
        else:
            raise ValueError(
                f"Unrecognized normalization type string provided: {use_norm}. Should be in "
                f"{supported_norms}"
            )

    ## Check dict
    elif isinstance(use_norm, dict):
        norm_params = use_norm

    else:
        raise ValueError(
            f"Invalid type for use_norm should either be a bool (batchnorm/identity), "
            f"a string in {supported_norms}, or a dict like {{'type': 'batchnorm', **kwargs}}"
        )

    # Step 2. Check if the dict is valid
    if "type" not in norm_params:
        raise ValueError(
            f"Malformed dictionary given in use_norm: {use_norm}. Should contain key 'type'."
        )
    if norm_params["type"] not in supported_norms:
        raise ValueError(
            f"Unrecognized normalization type string provided: {use_norm}. Should be in {supported_norms}"
        )

    # Step 3. Initialize the norm layer
    norm_type = norm_params["type"]
    norm_kwargs = {k: v for k, v in norm_params.items() if k != "type"}


    if norm_type == "batchnorm":
        norm = nn.BatchNorm2d(out_channels, **norm_kwargs)
    elif norm_type == "identity":
        norm = nn.Identity()
    elif norm_type == "layernorm":
        norm = nn.LayerNorm(out_channels, **norm_kwargs)
    elif norm_type == "instancenorm":
        norm = nn.InstanceNorm2d(out_channels, **norm_kwargs)
    else:
        raise ValueError(f"Unrecognized normalization type: {norm_type}")

    return norm

class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
    ):
        norm = get_norm_layer(use_norm, out_channels)

        is_identity = isinstance(norm, nn.Identity)
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=is_identity,
        )

        is_inplaceabn = InPlaceABN is not None and isinstance(norm, InPlaceABN)
        activation = nn.Identity() if is_inplaceabn else nn.ReLU(inplace=True)

        super(Conv2dReLU, self).__init__(conv, norm, activation)

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
    
class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        attention_type: Optional[str] = None,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        self.attention1 = Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)
        self.interpolation_mode = interpolation_mode

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode=self.interpolation_mode)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
    ):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        super().__init__(conv1, conv2)


class UnetPlusPlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        n_blocks: int = 5,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        attention_type: Optional[str] = None,
        interpolation_mode: str = "nearest",
        center: bool = False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but you provide `decoder_channels` for {len(decoder_channels)} blocks."
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(
                head_channels,
                head_channels,
                use_norm=use_norm,
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(
            use_norm=use_norm,
            attention_type=attention_type,
            interpolation_mode=interpolation_mode,
        )

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (
                        layer_idx + 1 - depth_idx
                    )
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(
                    in_ch, skip_ch, out_ch, **kwargs
                )
        blocks[f"x_{0}_{len(self.in_channels) - 1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](
                        features[depth_idx], features[depth_idx + 1]
                    )
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [
                        dense_x[f"x_{idx}_{dense_l_i}"]
                        for idx in range(depth_idx + 1, dense_l_i + 1)
                    ]
                    cat_features = torch.cat(
                        cat_features + [features[dense_l_i + 1]], dim=1
                    )
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[
                        f"x_{depth_idx}_{dense_l_i}"
                    ](dense_x[f"x_{depth_idx}_{dense_l_i - 1}"], cat_features)
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](
            dense_x[f"x_{0}_{self.depth - 1}"]
        )
        return dense_x[f"x_{0}_{self.depth}"]


class UNetPlusPlusModel(ModelABC):
    """UNet++ Model."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        decoder_interpolation: str = "nearest",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        if encoder_name.startswith("mit_b"):
            raise ValueError(
                "UnetPlusPlus is not support encoder_name={}".format(encoder_name)
            )

        decoder_use_batchnorm = kwargs.pop("decoder_use_batchnorm", None)
        if decoder_use_batchnorm is not None:
            warnings.warn(
                "The usage of decoder_use_batchnorm is deprecated. Please modify your code for decoder_use_norm",
                DeprecationWarning,
                stacklevel=2,
            )
            decoder_use_norm = decoder_use_batchnorm

        self.encoder = TimmUniversalEncoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_norm=decoder_use_norm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            interpolation_mode=decoder_interpolation,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "unetplusplus-{}".format(encoder_name)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features = self.encoder(x)
        decoder_output = self.decoder(features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks