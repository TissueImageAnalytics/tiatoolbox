# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2020, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****


import torch.nn as nn
from collections import OrderedDict


class ModelDesc(nn.Module):
    """
    Compatible definition with existing TF definition of SCCNN
    """

    def __init__(self, input_ch=4, nr_types=4, **kwargs):
        super().__init__()

        has_bn = False

        def conv2d(in_ch, out_ch, ksize=3, stride=1, pad=0, bn=has_bn, act=nn.ReLU):
            layer_list = []
            layer_list.append(
                (
                    "conv",
                    nn.Conv2d(in_ch, out_ch, ksize, stride=1, padding=pad, bias=not bn),
                )
            )
            if bn:
                layer_list.append(("conv/bn", nn.BatchNorm2d(out_ch, eps=1e-5)))
            if act is not None:
                layer_list.append(("conv/act/", act()))
            return nn.Sequential(OrderedDict(layer_list))

        self.feature = nn.Sequential(
            OrderedDict(
                [
                    ("l1", conv2d(input_ch, 32, ksize=2, stride=1, bn=has_bn)),
                    ("pool1", nn.MaxPool2d(kernel_size=2, padding=0)),
                    ("l2", conv2d(32, 64, ksize=2, stride=1, bn=has_bn)),
                    ("pool2", nn.MaxPool2d(kernel_size=2, padding=0)),
                    ("l3", conv2d(64, 128, ksize=3, stride=1, bn=has_bn)),
                    ("pool3", nn.MaxPool2d(kernel_size=2, padding=0)),
                    ("l4", conv2d(128, 1024, ksize=5, stride=1, bn=has_bn)),
                    ("drop_out_1", nn.Dropout2d(p=0.5, inplace=True)),
                    ("l5", conv2d(1024, 512, ksize=1, stride=1, bn=has_bn)),
                    ("drop_out_2", nn.Dropout2d(p=0.5, inplace=True)),
                ]
            )
        )

        self.sc = conv2d(512, nr_types, ksize=1, bn=has_bn, act=None)

    def forward(self, imgs):
        """
        imgs should be in range of [0, 1.0]
        """
        batch_size = imgs.shape[0]

        feat = self.feature(imgs)
        out_logit = self.sc(feat)

        return out_logit
