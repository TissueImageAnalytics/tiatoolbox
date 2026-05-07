from torch import nn

from .densenet import densenet121
from .dsf_cnn import dsf_cnn_4, dsf_cnn_8, dsf_cnn_12
from .mobilenet import mobilenet_v2

# import e2cnn.nn as enn
# from e2cnn import gspaces
from .resnet import resnet18, resnet34, resnet50
from .unet_encoder import UnetEncoder

# from .e2wrn import wrn16_2_stl_d8d8d8d8, wrn16_4_stl_d8d8d8d8, wrn16_4_stl_c8c8c8c8


def get_backbone(backbone_name, pretrained=False):
    """Helper function to get backbone network."""
    backbone_dict = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "densenet121": densenet121,
        "mobilenet_v2": mobilenet_v2,
        "unet_encoder": UnetEncoder,
        "dsf_cnn_4": dsf_cnn_4,
        "dsf_cnn_8": dsf_cnn_8,
        "dsf_cnn_12": dsf_cnn_12,
        # "wrn16_2_stl_d8d8d8d8": wrn16_2_stl_d8d8d8d8,
        # "wrn16_4_stl_d8d8d8d8": wrn16_4_stl_d8d8d8d8,
        # "wrn16_4_stl_c8c8c8c8": wrn16_4_stl_c8c8c8c8,
    }
    filter_info_dict = {
        "resnet18": [64, 64, 128, 256, 512],
        "resnet34": [64, 64, 128, 256, 512],
        "resnet50": [64, 256, 512, 1024, 2048],
        "densenet121": [64, 256, 512, 1024, 1024],
        "mobilenet_v2": [32, 24, 32, 96, 1280],
        "unet_encoder": [64, 128, 256, 512, 1024],
        "dsf_cnn_4": [10, 16, 32, 32, 32],
        "dsf_cnn_8": [10, 16, 32, 32, 32],
        "dsf_cnn_12": [10, 16, 32, 32, 32],
        "wrn16_2_stl_d8d8d8d8": [4, 8, 16, 32, 32],
        "wrn16_4_stl_d8d8d8d8": [4, 16, 32, 64, 64],
        "wrn16_4_stl_c8c8c8c8": [5, 22, 45, 90, 90],
    }
    gspace_dict = {
        # "wrn16_2_stl_d8d8d8d8": [
        #     enn.FieldType(gspaces.FlipRot2dOnR2(N=8), 4 * [gspaces.FlipRot2dOnR2(N=8).regular_repr]),
        #     enn.FieldType(gspaces.FlipRot2dOnR2(N=8), 8 * [gspaces.FlipRot2dOnR2(N=8).regular_repr]),
        #     enn.FieldType(gspaces.FlipRot2dOnR2(N=8), 16 * [gspaces.FlipRot2dOnR2(N=8).regular_repr]),
        #     enn.FieldType(gspaces.FlipRot2dOnR2(N=8), 32 * [gspaces.FlipRot2dOnR2(N=8).regular_repr]),
        #     enn.FieldType(gspaces.FlipRot2dOnR2(N=8), 32 * [gspaces.FlipRot2dOnR2(N=8).regular_repr])
        #     ],
        # "wrn16_4_stl_d8d8d8d8": [
        #     enn.FieldType(gspaces.FlipRot2dOnR2(N=8), 4 * [gspaces.FlipRot2dOnR2(N=8).regular_repr]),
        #     enn.FieldType(gspaces.FlipRot2dOnR2(N=8), 16 * [gspaces.FlipRot2dOnR2(N=8).regular_repr]),
        #     enn.FieldType(gspaces.FlipRot2dOnR2(N=8), 32 * [gspaces.FlipRot2dOnR2(N=8).regular_repr]),
        #     enn.FieldType(gspaces.FlipRot2dOnR2(N=8), 64 * [gspaces.FlipRot2dOnR2(N=8).regular_repr]),
        #     enn.FieldType(gspaces.FlipRot2dOnR2(N=8), 64 * [gspaces.FlipRot2dOnR2(N=8).regular_repr])
        #     ],
        # "wrn16_4_stl_c8c8c8c8": [
        #     enn.FieldType(gspaces.Rot2dOnR2(N=8), 5 * [gspaces.Rot2dOnR2(N=8).regular_repr]),
        #     enn.FieldType(gspaces.Rot2dOnR2(N=8), 22 * [gspaces.Rot2dOnR2(N=8).regular_repr]),
        #     enn.FieldType(gspaces.Rot2dOnR2(N=8), 45 * [gspaces.Rot2dOnR2(N=8).regular_repr]),
        #     enn.FieldType(gspaces.Rot2dOnR2(N=8), 90 * [gspaces.Rot2dOnR2(N=8).regular_repr]),
        #     enn.FieldType(gspaces.Rot2dOnR2(N=8), 90 * [gspaces.Rot2dOnR2(N=8).regular_repr])
        #     ]
    }

    backbone = backbone_dict[backbone_name](pretrained=pretrained)
    filter_info = filter_info_dict[backbone_name]

    gspace_info = None
    if backbone_name in gspace_dict:
        gspace_info = gspace_dict[backbone_name]
    return backbone, filter_info, gspace_info
