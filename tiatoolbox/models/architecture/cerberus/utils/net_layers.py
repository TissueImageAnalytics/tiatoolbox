import torch.nn.functional as F

from .conv_layers import Conv2d, ConvBlock, ConvBlock_PreAct
from .gconv_layers import GConvBlock, GroupPool
from .misc_utils import Pytorch_Base


def get_decoder(backbone_name, f):
    """Build the decoder block basing on the given convolution layer with `backbone_name`
    for each up sampling level. The number of block is correspond with the given list of input
    down-sampling filter info `f` and return as lowest resolution to highest
    """
    if backbone_name[:3] == "dsf":
        nr_orients = int(backbone_name.split("_")[-1])
        u4 = GConvBlock(f[-2], [f[-2], f[-3]], 7, nr_orients, nr_orients)
        u3 = GConvBlock(f[-3], [f[-3], f[-4]], 7, nr_orients, nr_orients)
        u2 = GConvBlock(f[-4], [f[-4], f[-5]], 7, nr_orients, nr_orients)
        u1 = GConvBlock(f[-5], [f[-5], f[-5]], 7, nr_orients, nr_orients)
    else:
        u4 = ConvBlock(f[-2], [f[-2], f[-3]], 3)
        u3 = ConvBlock(f[-3], [f[-3], f[-4]], 3)
        u2 = ConvBlock(f[-4], [f[-4], f[-5]], 3)
        u1 = ConvBlock(f[-5], [f[-5], f[-5]], 3)

    return [u4, u3, u2, u1]


def get_classification_head(backbone_name, f, out_ch, int_ch=96):

    if backbone_name[:3] == "dsf":
        return ConvBlock_PreAct(f[-5], [int_ch, out_ch], ksize=1)
    conv_blk = ConvBlock(f[-5], [int_ch], ksize=1)
    conv = Conv2d(int_ch, out_ch, ksize=1)
    return Pytorch_Base(conv_blk, conv)


def group_pool_layer(backbone_name, out_type=None):
    nr_orients = int(backbone_name.split("_")[-1])
    gpool = GroupPool(nr_orients, pool_type="max")
    return gpool


def upsample2x(feat, net_code, out_type=None):
    return F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=False)
