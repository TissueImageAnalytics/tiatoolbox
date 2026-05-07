from torch import nn

from ..utils.gconv_layers import GConv2d, GConvBlock, GDenseBlock


class DSF_CNN(nn.Module):
    def __init__(self, nr_orients):
        super().__init__()
        # input layers
        self.i1 = GConv2d(3, 10, 7, 1, nr_orients, padding=3)
        self.i2 = GConvBlock(10, 10, 7, nr_orients, nr_orients)
        self.p1 = nn.MaxPool2d((2, 2))
        # dense layers
        self.d1 = GDenseBlock(10, 16, [7, 5], [14, 6], 3, nr_orients, False)
        self.p2 = nn.MaxPool2d((2, 2))
        self.d2 = GDenseBlock(16, 32, [7, 5], [14, 6], 4, nr_orients, False)
        self.p3 = nn.MaxPool2d((2, 2))
        self.d3 = GDenseBlock(32, 32, [7, 5], [14, 6], 5, nr_orients, False)
        self.p4 = nn.MaxPool2d((2, 2))
        self.d4 = GDenseBlock(32, 32, [7, 5], [14, 6], 6, nr_orients, False)

    def forward(self, x):
        x1 = self.i2(self.i1(x))
        p1 = self.p1(x1)
        x2 = self.d1(p1)
        p2 = self.p2(x2)
        x3 = self.d2(p2)
        p3 = self.p3(x3)
        x4 = self.d3(p3)
        p4 = self.p4(x4)
        x5 = self.d4(p4)

        feats = [x1, x2, x3, x4, x5]

        return feats


def dsf_cnn_4(pretrained=False):
    """DSF-CNN with 4 filter orientations from

    https://arxiv.org/pdf/2004.03037.pdf

    """
    if pretrained == True:
        print("WARNING: No pre-trained model available for DSF-CNN!")
    return DSF_CNN(nr_orients=4)


def dsf_cnn_8(pretrained=False):
    """DSF-CNN with 8 filter orientations from

    https://arxiv.org/pdf/2004.03037.pdf

    """
    if pretrained == True:
        print("WARNING: No pre-trained model available for DSF-CNN!")
    return DSF_CNN(nr_orients=8)


def dsf_cnn_12(pretrained=False):
    """DSF-CNN with 12 filter orientations from

    https://arxiv.org/pdf/2004.03037.pdf

    """
    if pretrained == True:
        print("WARNING: No pre-trained model available for DSF-CNN!")
    return DSF_CNN(nr_orients=12)
