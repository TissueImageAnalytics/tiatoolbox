from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils import checkpoint

from .gconv_utils import get_rotated_basis_filters, get_rotated_filters


class GConv2d(nn.Module):
    """2D Steerable Filter G-Convolution layer

    Args:
        in_ch: number of input feature maps (per orientation)
        out_ch: number of output feature maps produced (per orientation)
        ksize: size of kernel
        basis_filters: atomic basis filters
        rot_info: array that determines how to rotate filters
        domain: the domain of the operation - choose Z2 (input layer) or G (hidden layer)
        strides: stride of kernel for convolution
        use_bias: whether to use bias

    """

    def __init__(
        self,
        in_ch,
        out_ch,
        ksize,
        nr_orients_in,
        nr_orients_out,
        stride=1,
        use_bias=False,
        dilation=1,
        padding=0,
        groups=1,
    ):
        super().__init__()

        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.cycle_filter = nr_orients_in > 1
        basis_filters = get_rotated_basis_filters(ksize, nr_orients_out)

        nr_b_filts = basis_filters.shape[2]

        # init weights
        w1 = np.zeros(
            [1, nr_b_filts, 1, 1, nr_orients_in, in_ch, out_ch], dtype=np.float32
        )  # real component
        w2 = np.zeros(
            [1, nr_b_filts, 1, 1, nr_orients_in, in_ch, out_ch], dtype=np.float32
        )  # imag component
        weight = torch.tensor(np.stack([w1, w2]), requires_grad=True)
        # stack real and imaginary coefficients
        self.weight = Parameter(weight)
        if use_bias:
            bias = np.zeros(out_ch, dtype=np.float32)
            bias = torch.tensor(bias, requires_grad=True)
            self.bias = Parameter(bias)
        else:
            self.bias = None

        self.ksize = ksize
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nr_orients_out = nr_orients_out
        self.nr_orients_in = nr_orients_in

        self.register_buffer("basis_filters", basis_filters)

    def _conv_forward(self, input, weight):
        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Generate filters at different orientations- also perform cyclic permutation of channels if f: G -> G
        # Cyclic permutation of filters happenens for all rotation equivariant layers except for the input layer
        # [nr_orients_out, K, K, nr_orients_in, in_ch, out_ch]
        filters = get_rotated_filters(
            self.weight, self.nr_orients_out, self.basis_filters, self.cycle_filter
        )

        # reshape filters for 2D convolution
        # [nr_orients_out, out_ch, nr_orients_in, in_ch, K, K]
        filters = filters.permute(0, 5, 3, 4, 1, 2).contiguous()
        filters = filters.reshape(
            self.nr_orients_out * self.out_ch,
            self.nr_orients_in * self.in_ch,
            self.ksize,
            self.ksize,
        )
        feat = self._conv_forward(input, filters)
        return feat


class _DenseLayer(nn.Module):
    def __init__(
        self,
        in_ch,
        unit_ksize,
        unit_feat,
        nr_orients,
        drop_rate,
        memory_efficient=False,
    ):
        super().__init__()
        unit_pad = [int(v // 2) for v in unit_ksize]
        self.units = nn.ModuleList()

        unit_out_orients = nr_orients
        self.nr_orients = nr_orients

        unit_idx = 0
        unit_in_ch = in_ch
        unit_in_orient = 1 if unit_ksize[unit_idx] == 1 else nr_orients
        (self.add_module("norm1", GBatchNorm2d(unit_in_ch, nr_orients)),)
        (self.add_module("relu1", nn.ReLU(inplace=True)),)
        self.add_module(
            "conv1",
            GConv2d(
                unit_in_ch,
                unit_feat[unit_idx],
                unit_ksize[unit_idx],
                unit_in_orient,
                unit_out_orients,
                padding=unit_pad[unit_idx],
            ),
        )

        unit_idx = 1
        unit_in_ch = unit_feat[unit_idx - 1]
        unit_in_orient = 1 if unit_ksize[unit_idx] == 1 else nr_orients
        (self.add_module("norm2", GBatchNorm2d(unit_in_ch, nr_orients)),)
        (self.add_module("relu2", nn.ReLU(inplace=True)),)
        self.add_module(
            "conv2",
            GConv2d(
                unit_in_ch,
                unit_feat[unit_idx],
                unit_ksize[unit_idx],
                unit_in_orient,
                unit_out_orients,
                padding=unit_pad[unit_idx],
            ),
        )

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, *inputs):
        # type: (List[Tensor]) -> Tensor
        # ! input is a list where each item of shape N x Orient x C x H x W
        feat = torch.cat(inputs, 2)  # cat the list along the C, not orient
        # ! reshape into N x Orient * C x H x W
        b, o, c, h, w = feat.shape
        feat = torch.reshape(feat, (-1, o * c, h, w))

        feat = self.norm1(feat)
        feat = self.relu1(feat)
        feat = self.conv1(feat)
        return feat

    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input, freeze):
        prev_features = input
        if self.training:
            if not freeze:
                if self.memory_efficient and self.any_requires_grad(prev_features):
                    if torch.jit.is_scripting():
                        raise Exception("Memory Efficient not supported in JIT")
                    bottleneck_output = checkpoint.checkpoint(
                        self.bn_function, *prev_features
                    )
                else:
                    bottleneck_output = self.bn_function(*prev_features)
                new_features = self.norm2(bottleneck_output)
                new_features = self.relu2(new_features)
                new_features = self.conv2(new_features)
            else:
                with torch.set_grad_enabled(False):
                    bottleneck_output = self.bn_function(*prev_features)
                    new_features = self.norm2(bottleneck_output)
                    new_features = self.relu2(new_features)
                    new_features = self.conv2(new_features)
        else:
            bottleneck_output = self.bn_function(*prev_features)
            new_features = self.norm2(bottleneck_output)
            new_features = self.relu2(new_features)
            new_features = self.conv2(new_features)

        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return new_features


class GDenseBlock(nn.Module):
    """Dense Block as defined in:

    Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger.
    "Densely connected convolutional networks." In Proceedings of the IEEE conference
    on computer vision and pattern recognition, pp. 4700-4708. 2017.
    Only performs `valid` convolution.

    """

    def __init__(
        self,
        in_ch,
        out_ch,
        unit_ksize,
        unit_ch,
        unit_count,
        nr_orients,
        memory_efficient,
        drop_rate=0.0,
    ):
        super().__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalanced Unit Info"

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch
        self.nr_orients = nr_orients
        self.sub_ch = in_ch + unit_count * unit_ch[-1]

        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            self.units.append(
                _DenseLayer(
                    unit_in_ch,
                    unit_ksize,
                    unit_ch,
                    nr_orients,
                    drop_rate,
                    memory_efficient,
                )
            )
            unit_in_ch = in_ch + unit_ch[1] * (idx + 1)

        sub_ch = in_ch + unit_count * unit_ch[-1]
        # transition layer
        self.transition = nn.Sequential(
            OrderedDict(
                [
                    ("bn", GBatchNorm2d(sub_ch, nr_orients)),
                    ("relu", nn.ReLU(inplace=True)),
                    (
                        "conv",
                        GConv2d(sub_ch, out_ch, 5, nr_orients, nr_orients, padding=2),
                    ),
                ]
            )
        )

    def forward(self, prev_feat, freeze=False):
        b, c, h, w = prev_feat.shape
        prev_feat = torch.reshape(prev_feat, (b, self.nr_orients, -1, h, w))

        feat_list = [prev_feat]
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](feat_list, freeze)
            b, c, h, w = new_feat.shape
            new_feat = torch.reshape(new_feat, (b, self.nr_orients, -1, h, w))
            feat_list.append(new_feat)
        # ! input is a list where each item of shape N x Orient x C x H x W
        feat = torch.cat(feat_list, 2)  # cat the list along the C, not orient
        # ! reshape into N x Orient * C x H x W
        b, o, c, h, w = feat.shape
        feat = feat.reshape(-1, o * c, h, w)

        # transition layer
        if self.training:
            with torch.set_grad_enabled(not freeze):
                new_feat = self.transition(feat)
        else:
            new_feat = self.transition(feat)

        return new_feat


class _GConvLayer(nn.Module):
    def __init__(
        self, in_ch, out_ch, ksize, nr_orients_in, nr_orients_out, pad=True, preact=True
    ):
        super().__init__()

        pad_size = int(ksize // 2) if pad else 0
        self.preact = preact

        if preact:
            self.pre_bn = GBatchNorm2d(in_ch, nr_orients_in)
        else:
            self.post_bn = GBatchNorm2d(out_ch, nr_orients_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv = GConv2d(
            in_ch, out_ch, ksize, nr_orients_in, nr_orients_out, padding=pad_size
        )

    def forward(self, prev_feat, freeze=False):
        feat = prev_feat
        if self.training:
            with torch.set_grad_enabled(not freeze):
                if self.preact:
                    feat = self.pre_bn(feat)
                    feat = self.relu(feat)
                    feat = self.conv(feat)
                else:
                    feat = self.conv(feat)
                    feat = self.post_bn(feat)
                    feat = self.relu(feat)
        elif self.preact:
            feat = self.pre_bn(feat)
            feat = self.relu(feat)
            feat = self.conv(feat)
        else:
            feat = self.conv(feat)
            feat = self.post_bn(feat)
            feat = self.relu(feat)

        return feat


class GConvBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        unit_ch,
        ksize,
        nr_orients_in,
        nr_orients_out,
        pad=True,
        preact=True,
    ):
        super().__init__()

        if not isinstance(unit_ch, list):
            unit_ch = [unit_ch]

        self.nr_layers = len(unit_ch)
        self.block = nn.ModuleList()

        for idx in range(self.nr_layers):
            self.block.append(
                _GConvLayer(
                    in_ch,
                    unit_ch[idx],
                    ksize,
                    nr_orients_in,
                    nr_orients_out,
                    pad=pad,
                    preact=preact,
                )
            )
            in_ch = unit_ch[idx]
            if idx > 0:
                nr_orients_in = nr_orients_out

    def forward(self, prev_feat, freeze=False):
        feat = prev_feat
        if self.training:
            with torch.set_grad_enabled(not freeze):
                for idx in range(self.nr_layers):
                    feat = self.block[idx](feat)
        else:
            for idx in range(self.nr_layers):
                feat = self.block[idx](feat)

        return feat


class GBatchNorm2d(nn.Module):
    """A shorthand of Group Equivariant Batch Normalization.

    Args:
       ch: number of channels
       nr_orients: number of filter orientations

    """

    def __init__(self, ch, nr_orients, eps=1e-5):
        super().__init__()
        self.ch = ch
        self.nr_orients = nr_orients
        self.norm = nn.BatchNorm3d(self.ch, eps)
        self.eps = eps

    def forward(self, x):
        shape = x.size()
        x = torch.reshape(x, (-1, self.nr_orients, self.ch, shape[2], shape[3]))
        x = x.permute(0, 2, 1, 3, 4)
        x = self.norm(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = torch.reshape(x, (-1, self.nr_orients * self.ch, shape[2], shape[3]))
        return x


class GroupPool(nn.Module):
    """Perform pooling along the orientation axis.

    Args:
        nr_orients: number of filter orientations
        pool_type: choose either 'max' or 'mean'

    """

    def __init__(self, nr_orients, pool_type="max"):
        super().__init__()
        self.nr_orients = nr_orients
        self.pool_type = pool_type

        assert pool_type == "max" or pool_type == "mean", (
            "Pool type must be either `max` or `mean`"
        )

    def forward(self, x):
        shape = x.size()
        new_shape = [
            -1,
            self.nr_orients,
            shape[1] // self.nr_orients,
            shape[2],
            shape[3],
        ]
        x = x.view(new_shape)
        x = x.permute(0, 2, 1, 3, 4)
        if self.pool_type == "max":
            x, _ = torch.max(x, dim=2)
        elif self.pool_type == "mean":
            x = torch.mean(x, dim=2)
        return x
