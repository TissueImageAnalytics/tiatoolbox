"""Minimal Cerberus network definition for the released ResNet-34 checkpoint."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional

from .backbone.resnet import resnet34
from .utils.conv_layers import Conv2d, ConvBlock, PytorchBase

DECODER_KWARGS = {
    "Gland": {"INST": 3},
    "Gland#TYPE": {"TYPE": 3},
    "Lumen": {"INST": 3},
    "Nuclei": {"INST": 3},
    "Nuclei#TYPE": {"TYPE": 7},
    "Patch-Class": {"OUT": 9},
}

CONSIDERED_TASKS = {
    "Nuclei",
    "Nuclei#TYPE",
    "Gland",
    "Gland#TYPE",
    "Lumen",
    "Patch-Class",
}


def cropping_center(x: torch.Tensor, crop_shape: tuple[int, int]) -> torch.Tensor:
    """Crop a batched NCHW tensor at the centre."""
    h0 = int((x.shape[2] - crop_shape[0]) * 0.5)
    w0 = int((x.shape[3] - crop_shape[1]) * 0.5)
    return x[:, :, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]


class NetDesc(nn.Module):
    """Cerberus model topology used by ``resnet34_cerberus`` weights."""

    def __init__(self) -> None:
        """Initialize the fixed Cerberus model topology."""
        super().__init__()
        self.encoder_backbone_name = "resnet34"
        self.decoder_info_list = DECODER_KWARGS
        self.decoder_info = [64, 64, 128, 256, 512]

        self.backbone = resnet34()
        self.conv_map = nn.Conv2d(512, 256, (1, 1), bias=False)
        self.decoder_head = nn.ModuleDict()
        self.output_head = nn.ModuleDict()

        for decoder_name, output_head in self.decoder_info_list.items():
            if decoder_name not in CONSIDERED_TASKS:
                continue
            if decoder_name == "Patch-Class":
                self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                for output_ch in output_head.values():
                    self.decoder_head["Patch-Class"] = nn.Sequential(
                        OrderedDict(
                            [
                                ("bn1", nn.BatchNorm2d(512, eps=1e-5)),
                                ("relu1", nn.ReLU(inplace=True)),
                                ("dropout", nn.Dropout(p=0.3)),
                                ("conv1", nn.Conv2d(512, 256, 1)),
                                ("bn2", nn.BatchNorm2d(256, eps=1e-5)),
                                ("relu2", nn.ReLU(inplace=True)),
                                ("conv2", nn.Conv2d(256, output_ch, 1)),
                            ]
                        )
                    )
                continue

            self.decoder_head[decoder_name] = nn.ModuleList(
                [
                    ConvBlock(256, [256, 128], 3),
                    ConvBlock(128, [128, 64], 3),
                    ConvBlock(64, [64, 64], 3),
                    ConvBlock(64, [64, 64], 3),
                ]
            )
            decoder_output_head = nn.ModuleDict()
            for output_name, output_ch in output_head.items():
                decoder_output_head[output_name] = PytorchBase(
                    ConvBlock(64, [96], ksize=1),
                    Conv2d(96, output_ch, ksize=1),
                )
            self.output_head[decoder_name] = decoder_output_head

    def forward(
        self,
        imgs: torch.Tensor,
        train_decoder_list: list[str] | None = None,
    ) -> OrderedDict:
        """Return a dictionary of Cerberus output heads."""
        _ = train_decoder_list
        imgs = imgs / 255.0
        feat_list = self.backbone(imgs)
        bottom_feats = feat_list[-1]
        feat_list[-1] = self.conv_map(bottom_feats)

        output_dict = OrderedDict()
        for decoder_name, decoder in self.decoder_head.items():
            if decoder_name == "Patch-Class":
                patch_feats = bottom_feats
                if patch_feats.shape[-2:] != (9, 9):
                    patch_feats = cropping_center(patch_feats, (9, 9))
                patch_feats = self.global_avg_pool(patch_feats)
                output_dict[decoder_name] = decoder(patch_feats)
                continue

            prev_feat = feat_list[-1]
            for idx in range(1, len(feat_list)):
                prev_feat = functional.interpolate(
                    prev_feat,
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                )
                prev_feat = decoder[idx - 1](feat_list[-(idx + 1)] + prev_feat)

            decoder_output_head = self.output_head[decoder_name]
            for clf_name, clf in decoder_output_head.items():
                output_dict[decoder_name.split("#")[0] + "-" + clf_name] = clf(
                    prev_feat
                )

        return output_dict
