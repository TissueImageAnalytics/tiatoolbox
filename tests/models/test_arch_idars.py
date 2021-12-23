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
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Functional unit test package for IDARS."""

import torch

from tiatoolbox.models.architecture.idars import IDaRS


def test_functional():
    """Functional test for architectures."""
    # test forward
    samples = torch.rand(4, 3, 224, 224, dtype=torch.float32)
    model = IDaRS("resnet18")
    model(samples)

    model = IDaRS("resnet34")
    model(samples)

    # test preproc function
    img = torch.rand(224, 224, 3, dtype=torch.float32)
    img_ = IDaRS.preproc(img.numpy())
    assert tuple(img_.shape) == (224, 224, 3)
    img_ = IDaRS.preproc(img.numpy())
    assert tuple(img_.shape) == (224, 224, 3)
    # dummy to make runtime crash
    img_ = IDaRS.preproc(img.numpy() / 0.0)
    assert tuple(img_.shape) == (224, 224, 3)
