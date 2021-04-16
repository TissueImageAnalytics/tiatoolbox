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

from tiatoolbox.models.models_abc import Model_Base


class Model_Base(nn.Module):
    """Abstract base class for backbone models used in tiatoolbox."""

    def __init__(self, weight_init=False):
        """"""
        super().__init__()
        return

    @staticmethod
    def weight_init():
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        return
