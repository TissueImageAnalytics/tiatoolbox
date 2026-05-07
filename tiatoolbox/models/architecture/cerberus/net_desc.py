from collections import OrderedDict

import torch
from torch import nn

from .backbone import get_backbone
from .utils import weights_init_cnn, weights_init_dsf
from .utils.misc_utils import cropping_center
from .utils.net_layers import (
    get_classification_head,
    get_decoder,
    group_pool_layer,
    upsample2x,
)


class NetDesc(nn.Module):
    """Initialise U-Net style network with a shared backbone
    and multiple branch decoders, each decoder may have different
    number of output channels and names.

    """

    def __init__(
        self,
        encoder_backbone_name=None,
        backbone_imagenet_pretrained=False,
        fullnet_custom_pretrained=False,
        decoder_kwargs={},
        considered_tasks=[],
        subtype_gland=False,
        subtype_nuclei=False,
    ):
        super().__init__()

        # build network depending on which tasks are considered
        self.considered_tasks = considered_tasks
        self.subtype_gland = subtype_gland  # whether to freeze all weights apart from gland semantic seg decoder
        self.subtype_nuclei = subtype_nuclei  # whether to freeze all weights apart from nuclei semantic seg decoder

        self.encoder_backbone_name = encoder_backbone_name
        self.net_code = encoder_backbone_name[:3]

        self.decoder_info_list = decoder_kwargs

        # ========= Get Encoder =========
        self.backbone, filters, self.gspace_info = get_backbone(
            encoder_backbone_name, backbone_imagenet_pretrained
        )
        self.decoder_info = filters

        if self.net_code != "dsf":
            self.conv_map = nn.Conv2d(filters[-1], filters[-2], (1, 1), bias=False)
        else:
            self.conv_map = nn.Identity()

        self.decoder_head = nn.ModuleDict()
        self.output_head = nn.ModuleDict()

        # ========= Get Decoders =========

        for decoder_name, output_head in self.decoder_info_list.items():
            # only build the network for tasks being considered
            if decoder_name in self.considered_tasks:
                if decoder_name == "Patch-Class":
                    self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                    for output_name, output_ch in output_head.items():
                        module_list = [
                            ("bn1", nn.BatchNorm2d(512, eps=1e-5)),
                            ("relu1", nn.ReLU(inplace=True)),
                            ("dropout", nn.Dropout(p=0.3)),
                            ("conv1", nn.Conv2d(512, 256, 1, stride=1, padding=0)),
                            ("bn2", nn.BatchNorm2d(256, eps=1e-5)),
                            ("relu2", nn.ReLU(inplace=True)),
                            (
                                "conv2",
                                nn.Conv2d(
                                    256, output_ch, 1, stride=1, padding=0, bias=True
                                ),
                            ),
                        ]
                        self.decoder_head["Patch-Class"] = nn.Sequential(
                            OrderedDict(module_list)
                        )
                else:
                    up_blk_list = get_decoder(encoder_backbone_name, self.decoder_info)
                    decoder_list = nn.ModuleList(up_blk_list)
                    self.decoder_head[decoder_name] = decoder_list
                    decoder_output_head = nn.ModuleDict()
                    for output_name, output_ch in output_head.items():
                        clf = get_classification_head(
                            encoder_backbone_name, filters, out_ch=output_ch
                        )
                        decoder_output_head[output_name] = clf
                    self.output_head[decoder_name] = decoder_output_head

        # ======= Initialise Weights =======
        if self.net_code != "dsf":
            if not (backbone_imagenet_pretrained or fullnet_custom_pretrained):
                self.backbone.apply(weights_init_cnn)
            if not fullnet_custom_pretrained:
                self.decoder_head.apply(weights_init_cnn)
        else:
            if not fullnet_custom_pretrained:
                self.backbone.apply(weights_init_dsf)
            if not fullnet_custom_pretrained:
                self.decoder_head.apply(weights_init_dsf)
        if not fullnet_custom_pretrained:
            self.output_head.apply(weights_init_cnn)

    def _freeze_weight(self):
        """Helper to manage freezing instead of random injection.

        Must be called outside of forward else bonker may happen.

        """

        def _freeze(container):
            for module in container.modules():
                for param in module.parameters():
                    param.requires_grad = False
                # for BatchNormalization, weight and bias have grad.
                # however, running statistics also get updated, but they are
                # not parameters, hence require_grad will have no effect.
                # To prevent update running statistics, must set the module
                # to be in eval mode
                # ! warning, doing this will unset the flag from the
                # ! external `with` block
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

        _freeze(self.backbone)
        _freeze(self.conv_map)

        for decoder_name, decoder in self.decoder_head.items():
            if decoder_name == "Patch-Class":
                _freeze(decoder)
            else:
                decoder_output_head = self.output_head[decoder_name]
                for head_name, head in decoder_output_head.items():
                    if (
                        head_name != "TYPE"
                        or (decoder_name == "Gland#TYPE" and not self.subtype_gland)
                        or (decoder_name == "Nuclei#TYPE" and not self.subtype_nuclei)
                    ):
                        _freeze(decoder)
                        _freeze(head)

    def forward(self, imgs, train_decoder_list=[]):
        """Output is a dictionary with key is `%s-%s` % (decoder_head, output_head)."""
        imgs = imgs / 255.0  # to 0-1 range

        # similar to torch no grad but flag with condition built in
        feat_list = self.backbone(imgs)
        # mapping the last channel block #ch to align
        bottom_feats = feat_list[-1]
        feat_list[-1] = self.conv_map(bottom_feats)

        output_dict = OrderedDict()
        for decoder_name, blk_list in self.decoder_head.items():
            # allow freezing decoder branch basing on name alone, dynamically
            # within training schedule (such as alternate between batch)
            decoder_train_flag = decoder_name in train_decoder_list

            # no gradient if using subtype mode - only train relevant decoders!
            if self.subtype_gland or self.subtype_nuclei:
                if (
                    "TYPE" not in decoder_name
                    or ("Gland" in decoder_name and not self.subtype_gland)
                    or ("Nuclei" in decoder_name and not self.subtype_nuclei)
                ):
                    decoder_train_flag = False
            if decoder_name == "Patch-Class":
                with torch.set_grad_enabled(decoder_train_flag):
                    feat_shape = bottom_feats[-2:].detach().cpu().numpy().shape[-2:]
                    # dimensions of features may be different during inference
                    if feat_shape[0] != 9 and feat_shape[1] != 9:
                        bottom_feats = cropping_center(bottom_feats, [9, 9], batch=True)
                    prev_feat = self.global_avg_pool(bottom_feats)
                    if self.net_code == "dsf":
                        prev_feat = group_pool_layer(
                            self.encoder_backbone_name, self.decoder_info[-1]
                        )(prev_feat)
                    output = self.decoder_head["Patch-Class"](prev_feat)
                    output_dict[decoder_name] = output
            else:
                with torch.set_grad_enabled(decoder_train_flag):
                    prev_feat = feat_list[-1]
                    for idx in range(1, len(feat_list)):
                        prev_feat = upsample2x(
                            prev_feat, self.net_code, self.decoder_info[-(idx + 1)]
                        )
                        down_feat = feat_list[-(idx + 1)]
                        new_feat = down_feat + prev_feat
                        prev_feat = blk_list[idx - 1](new_feat)

                    if self.net_code == "dsf":
                        prev_feat = group_pool_layer(
                            self.encoder_backbone_name, self.decoder_info[0]
                        )(prev_feat)

                    decoder_output_head = self.output_head[decoder_name]
                    for clf_name, clf in decoder_output_head.items():
                        output = clf(prev_feat)
                        output_dict[decoder_name.split("#")[0] + "-" + clf_name] = (
                            output
                        )

        return output_dict


def create_model(**kwargs):
    return NetDesc(**kwargs)
