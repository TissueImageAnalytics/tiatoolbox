import torch
from torch import nn


class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, pad=True):
        super().__init__()

        pad_size = int(ksize // 2) if pad else 0
        self.conv = nn.Conv2d(
            in_ch, out_ch, ksize, stride=1, padding=pad_size, bias=True
        )

    def forward(self, prev_feat, freeze=False):
        if self.training:
            with torch.set_grad_enabled(not freeze):
                new_feat = self.conv(prev_feat)
        else:
            new_feat = self.conv(prev_feat)

        return new_feat


class _ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, pad=True, preact=True, dilation=1):
        super().__init__()

        pad_size = int(ksize // 2) if pad else 0
        self.preact = preact

        if preact:
            self.bn = nn.BatchNorm2d(in_ch, eps=1e-5)
        else:
            self.bn = nn.BatchNorm2d(out_ch, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_ch, out_ch, ksize, padding=pad_size, bias=True, dilation=dilation
        )

    def forward(self, prev_feat, freeze=False):
        feat = prev_feat
        if self.training:
            with torch.set_grad_enabled(not freeze):
                if self.preact:
                    feat = self.bn(feat)
                    feat = self.relu(feat)
                    feat = self.conv(feat)
                else:
                    feat = self.conv(feat)
                    feat = self.bn(feat)
                    feat = self.relu(feat)
        elif self.preact:
            feat = self.bn(feat)
            feat = self.relu(feat)
            feat = self.conv(feat)
        else:
            feat = self.conv(feat)
            feat = self.bn(feat)
            feat = self.relu(feat)

        return feat


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        unit_ch,
        ksize,
        pad=True,
        dilation=1,
    ):
        super().__init__()

        if not isinstance(unit_ch, list):
            unit_ch = [unit_ch]

        self.nr_layers = len(unit_ch)
        self.block = nn.ModuleList()

        for idx in range(self.nr_layers):
            self.block.append(
                _ConvLayer(
                    in_ch, unit_ch[idx], ksize, pad=pad, preact=False, dilation=dilation
                )
            )
            in_ch = unit_ch[idx]

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


class ConvBlock_PreAct(nn.Module):
    def __init__(
        self,
        in_ch,
        unit_ch,
        ksize,
        pad=True,
        dilation=1,
    ):
        super().__init__()

        if not isinstance(unit_ch, list):
            unit_ch = [unit_ch]

        self.nr_layers = len(unit_ch)
        self.block = nn.ModuleList()

        for idx in range(self.nr_layers):
            self.block.append(
                _ConvLayer(
                    in_ch,
                    unit_ch[idx],
                    ksize,
                    pad=pad,
                    preact=True,
                    dilation=dilation,
                )
            )
            in_ch = unit_ch[idx]

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


class DilatedBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = ConvBlock(in_ch, [out_ch], ksize=3, dilation=1)
        self.conv2 = ConvBlock(in_ch, [out_ch], ksize=3, dilation=3)
        self.conv3 = ConvBlock(in_ch, [out_ch], ksize=3, dilation=6)
        self.conv4 = nn.Conv2d(out_ch * 3, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x4 = torch.cat((x1, x2, x3), dims=1)
        dropout = self.dropout(x4)
        x5 = self.conv4(dropout)

        return x5
