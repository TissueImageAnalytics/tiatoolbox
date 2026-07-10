# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# Licensed under the NVIDIA Source Code License. For full license
# terms, please refer to the LICENSE file provided with this code
# or visit NVIDIA's official repository at
# https://github.com/NVlabs/SegFormer/tree/master.
#
# This code has been modified.
# ---------------------------------------------------------------

"""Mix Transformer backbone components used by SegFormer models."""

from collections.abc import Callable, MutableMapping, Sequence
from functools import partial

import torch
from timm.layers import DropPath, to_2tuple
from torch import nn
from torch.nn import functional
from torch.nn.modules.module import _IncompatibleKeys

from tiatoolbox.models.architecture.utils import EncoderMixin

FOUR_DIMENSIONAL_TENSOR_NDIM = 4
MIN_ENCODER_DEPTH = 1
MAX_ENCODER_DEPTH = 5
STAGE_1_DEPTH = 2
STAGE_2_DEPTH = 3
STAGE_3_DEPTH = 4
STAGE_4_DEPTH = 5


class LayerNorm(nn.LayerNorm):
    """LayerNorm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for LayerNorm."""
        if x.ndim == FOUR_DIMENSIONAL_TENSOR_NDIM:
            batch_size, channels, height, width = x.shape
            x = x.view(batch_size, channels, -1).transpose(1, 2)
            x = functional.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )
            x = x.transpose(1, 2).view(batch_size, channels, height, width)
        else:
            x = functional.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )
        return x


class Mlp(nn.Module):
    """MLP module for Mix Transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Initializes the MLP module for Mix Transformer."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConvBlock(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Forward pass for the MLP module."""
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    """Attention module for Mix Transformer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        *,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ) -> None:
        """Initializes the Attention module for Mix Transformer."""
        super().__init__()
        if dim % num_heads != 0:
            msg = f"dim {dim} should be divided by num_heads {num_heads}."
            raise ValueError(msg)

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = LayerNorm(dim)
        else:
            # for torchscript compatibility
            self.sr = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Forward pass for the Attention module."""
        batch_size, token_count, channels = x.shape
        q = (
            self.q(x)
            .reshape(
                batch_size,
                token_count,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(batch_size, channels, height, width)
            x_ = self.sr(x_).reshape(batch_size, channels, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(
                    batch_size,
                    -1,
                    2,
                    self.num_heads,
                    channels // self.num_heads,
                )
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(
                    batch_size,
                    -1,
                    2,
                    self.num_heads,
                    channels // self.num_heads,
                )
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    """Block module for Mix Transformer."""

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = LayerNorm,
        sr_ratio: int = 1,
    ) -> None:
        """Initializes the Block module for Mix Transformer."""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        # Drop path applies stochastic depth between attention and MLP blocks.
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Block module."""
        batch_size, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x.transpose(1, 2).view(batch_size, -1, height, width)


class OverlapPatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """Initializes the OverlapPatchEmbed module for Mix Transformer."""
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the OverlapPatchEmbed module."""
        return self.norm(self.proj(x))


class MixVisionTransformer(nn.Module):
    """Mix Vision Transformer module."""

    def __init__(  # noqa: PLR0913
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        *,
        num_classes: int = 1000,
        embed_dims: Sequence[int] = (64, 128, 256, 512),
        num_heads: Sequence[int] = (1, 2, 4, 8),
        mlp_ratios: Sequence[float] = (4, 4, 4, 4),
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = LayerNorm,
        depths: Sequence[int] = (3, 4, 6, 3),
        sr_ratios: Sequence[int] = (8, 4, 2, 1),
    ) -> None:
        """Initializes the Mix Vision Transformer module."""
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.patch_size = patch_size

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.Sequential(
            *[
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.Sequential(
            *[
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.Sequential(
            *[
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.Sequential(
            *[
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

    def reset_drop_path(self, drop_path_rate: float) -> None:
        """Reset the drop path rate for all blocks in the Mix Vision Transformer."""
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        """Return the names of parameters that should not be subject to weight decay."""
        return {
            "pos_embed1",
            "pos_embed2",
            "pos_embed3",
            "pos_embed4",
            "cls_token",
        }  # has pos_embed may be better

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass for the Mix Vision Transformer to extract features."""
        outs: list[torch.Tensor] = []

        # stage 1
        x = self.patch_embed1(x)
        x = self.block1(x)
        x = self.norm1(x).contiguous()
        outs.append(x)

        # stage 2
        x = self.patch_embed2(x)
        x = self.block2(x)
        x = self.norm2(x).contiguous()
        outs.append(x)

        # stage 3
        x = self.patch_embed3(x)
        x = self.block3(x)
        x = self.norm3(x).contiguous()
        outs.append(x)

        # stage 4
        x = self.patch_embed4(x)
        x = self.block4(x)
        x = self.norm4(x).contiguous()
        outs.append(x)

        return outs

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass for the Mix Vision Transformer."""
        return self.forward_features(x)


class DWConvBlock(nn.Module):
    """Depthwise Convolution module for Mix Transformer."""

    def __init__(self, dim: int = 768) -> None:
        """Initializes the Depthwise Convolution module for Mix Transformer."""
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Forward pass for the Depthwise Convolution module."""
        batch_size, _, channels = x.shape
        x = x.transpose(1, 2).view(batch_size, channels, height, width)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


# ---------------------------------------------------------------
# End of NVIDIA code
# ---------------------------------------------------------------


class MixVisionTransformerEncoder(MixVisionTransformer, EncoderMixin):
    """Mix Vision Transformer Encoder module."""

    def __init__(
        self,
        out_channels: list[int],
        depth: int = 5,
        output_stride: int = 32,
        **kwargs: object,
    ) -> None:
        """Initializes the Mix Vision Transformer Encoder module."""
        if depth > MAX_ENCODER_DEPTH or depth < MIN_ENCODER_DEPTH:
            msg = (
                f"{self.__class__.__name__} depth should be in range "
                f"[{MIN_ENCODER_DEPTH}, {MAX_ENCODER_DEPTH}], got {depth}"
            )
            raise ValueError(msg)
        super().__init__(**kwargs)

        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        self._output_stride = output_stride

    def get_stages(self) -> dict[int, Sequence[torch.nn.Module]]:
        """Return the stages of the Mix Vision Transformer Encoder."""
        return {
            16: [self.patch_embed3, self.block3, self.norm3],
            32: [self.patch_embed4, self.block4, self.norm4],
        }

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass for the Mix Vision Transformer Encoder."""
        # create dummy output for the first block
        batch_size, _, height, width = x.shape
        dummy = torch.empty(
            [batch_size, 0, height // 2, width // 2], dtype=x.dtype, device=x.device
        )

        features = [x, dummy]

        if self._depth >= STAGE_1_DEPTH:
            x = self.patch_embed1(x)
            x = self.block1(x)
            x = self.norm1(x)
            x = x.contiguous()
            features.append(x)

        if self._depth >= STAGE_2_DEPTH:
            x = self.patch_embed2(x)
            x = self.block2(x)
            x = self.norm2(x)
            x = x.contiguous()
            features.append(x)

        if self._depth >= STAGE_3_DEPTH:
            x = self.patch_embed3(x)
            x = self.block3(x)
            x = self.norm3(x)
            x = x.contiguous()
            features.append(x)

        if self._depth >= STAGE_4_DEPTH:
            x = self.patch_embed4(x)
            x = self.block4(x)
            x = self.norm4(x)
            x = x.contiguous()
            features.append(x)

        return features

    def load_state_dict(
        self, state_dict: MutableMapping[str, torch.Tensor]
    ) -> _IncompatibleKeys:
        """Load the state dict for the Mix Vision Transformer Encoder."""
        state_dict.pop("head.weight", None)
        state_dict.pop("head.bias", None)
        return super().load_state_dict(state_dict)


mix_transformer_encoders = {
    "mit_b0": {
        "params": {
            "out_channels": [3, 0, 32, 64, 160, 256],
            "patch_size": 4,
            "embed_dims": [32, 64, 160, 256],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(LayerNorm, eps=1e-6),
            "depths": [2, 2, 2, 2],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b1": {
        "params": {
            "out_channels": [3, 0, 64, 128, 320, 512],
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(LayerNorm, eps=1e-6),
            "depths": [2, 2, 2, 2],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b2": {
        "params": {
            "out_channels": [3, 0, 64, 128, 320, 512],
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(LayerNorm, eps=1e-6),
            "depths": [3, 4, 6, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b3": {
        "params": {
            "out_channels": [3, 0, 64, 128, 320, 512],
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(LayerNorm, eps=1e-6),
            "depths": [3, 4, 18, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b4": {
        "params": {
            "out_channels": [3, 0, 64, 128, 320, 512],
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(LayerNorm, eps=1e-6),
            "depths": [3, 8, 27, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b5": {
        "params": {
            "out_channels": [3, 0, 64, 128, 320, 512],
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(LayerNorm, eps=1e-6),
            "depths": [3, 6, 40, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
}
