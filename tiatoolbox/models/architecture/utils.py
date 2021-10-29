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
# The Original Code is Copyright (C) 2021, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Defines utlity layers and operators for models in tiatoolbox."""


import numpy as np
import torch
import torch.nn as nn


def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image with substracted amount.

    Args:
        x (torch.Tensor): Input images.
        cropping (list): The substracted amount to center crop
          on each axis.
        data_format (str): Denote if the input is of `NCHW` or `NHWC`
          layout.

    Returns:
        x (torch.Tensor): Center cropped images.

    """
    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(f"Unknown input format `{data_format}`")

    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


class UpSample2x(nn.Module):
    """A layer to scale input by a factor of 2.

    This layer uses Kronecker product underneath rather than the default
    pytorch interpolation.

    """

    def __init__(self):
        super().__init__()
        # correct way to create constant within module
        self.register_buffer(
            "unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32"))
        )
        self.unpool_mat.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        """Logic for using layers defined in init.

        Args:
            x (torch.Tensor): Input images, the tensor is in the shape of NCHW.

        Returns:
            ret (torch.Tensor): Input images upsampled by a factor of 2
                via nearest neighbour interpolation. The tensor is the shape
                as NCHW.

        """
        input_shape = list(x.shape)
        # un-squeeze is the same as expand_dims
        # permute is the same as transpose
        # view is the same as reshape
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
        return ret
