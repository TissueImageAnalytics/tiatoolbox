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

"""Unit test package for HoVerNet."""

import torch

# import sys
# sys.path.append(".")

from tiatoolbox.models.backbone.utils import (
    center_crop,
    center_crop_to_shape,
)


def test_crop_operators():
    """Test for crop et. al. ."""
    sample = torch.rand((1, 3, 15, 15), dtype=torch.float32)
    output = center_crop(sample, [3, 3], data_format="NCHW")
    assert torch.sum(output - sample[:, :, 1:13, 1:13]) == 0, f"{output.shape}"

    sample = torch.rand((1, 15, 15, 3), dtype=torch.float32)
    output = center_crop(sample, [3, 3], data_format="NHWC")
    assert torch.sum(output - sample[:, 1:13, 1:13, :]) == 0, f"{output.shape}"

    # *
    x = torch.rand((1, 3, 15, 15), dtype=torch.float32)
    y = x[:, :, 6:9, 6:9]
    output = center_crop_to_shape(x, y, data_format="NCHW")
    assert torch.sum(output - y) == 0, f"{output.shape}"

    x = torch.rand((1, 15, 15, 3), dtype=torch.float32)
    y = x[:, 6:9, 6:9, :]
    output = center_crop_to_shape(x, y, data_format="NHWC")
    assert torch.sum(output - y) == 0, f"{output.shape}"
