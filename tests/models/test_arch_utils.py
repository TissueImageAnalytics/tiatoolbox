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

"""Unit test package for architecture utilities"""

import numpy as np
import pytest
import torch

from tiatoolbox.models.architecture.utils import UpSample2x, crop_op


def test_all():
    """Contains all tests for now."""
    layer = UpSample2x()
    sample = np.array([[1, 2], [3, 4]])[..., None]
    batch = torch.from_numpy(sample)[None]
    batch = batch.permute(0, 3, 1, 2).type(torch.float32)
    output = layer(batch).permute(0, 2, 3, 1)[0].numpy()
    _output = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ]
    )
    assert np.sum(_output - output) == 0

    #
    with pytest.raises(ValueError, match=r".*Unknown.*format.*"):
        crop_op(_output[None, :, :, None], [2, 2], "NHWCT")

    x = crop_op(_output[None, :, :, None], [2, 2], "NHWC")
    assert np.sum(x[0, :, :, 0] - sample) == 0
    x = crop_op(_output[None, None, :, :], [2, 2], "NCHW")
    assert np.sum(x[0, 0, :, :] - sample) == 0
