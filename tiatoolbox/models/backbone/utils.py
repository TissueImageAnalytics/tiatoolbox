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

import numpy as np
import torch
from typing import Union


def center_crop(
    img: Union[np.ndarray, torch.tensor],
    crop_shape: Union[np.ndarray, torch.tensor],
    data_format: str = "NCHW",
):
    """A function to center crop image with given crop shape.

    Args:
        img (ndarray, torch.tensor): input image, should be of 3 channels
        crop_shape (ndarray, torch.tensor): the substracted amount in the form of
            [substracted height, substracted width].
        data_format (str): choose either `NCHW` or `NHWC`

    """
    crop_t = crop_shape[0] // 2
    crop_b = crop_shape[0] - crop_t
    crop_l = crop_shape[1] // 2
    crop_r = crop_shape[1] - crop_l
    if data_format == "NCHW":
        img = img[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        img = img[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return img


def center_crop_to_shape(
    x: Union[np.ndarray, torch.tensor],
    y: Union[np.ndarray, torch.tensor],
    data_format: str = "NCHW",
):
    """A function to center crop image to shape.

    Centre crop `x` so that `x` has shape of `y` and `y` height and width must
    be smaller than `x` heigh width.

    Args:
        x (ndarray, torch.tensor): image to be cropped.
        y (ndarray, torch.tensor): reference image for getting cropping shape,
            should be of 3 channels.
        data_format: choose either `NCHW` or `NHWC`

    """

    """Centre crop x so that x has shape of y. y dims must be smaller than x dims.
    Args:
        x: input array
        y: array with desired shape.
    """
    if data_format == "NCHW":
        _, _, h1, w1 = x.shape
        _, _, h2, w2 = y.shape
    else:
        _, h1, w1, _ = x.shape
        _, h2, w2, _ = y.shape

    if h1 <= h2 or w1 <= w2:
        raise ValueError(
            " ".join(
                [
                    f"Height width of `x` is smaller than `y`",
                    f"{[h1, w1]} vs {[h2, w2]}",
                ]
            )
        )

    x_shape = x.shape
    y_shape = y.shape
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])

    return center_crop(x, crop_shape, data_format)
