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

"""Contains dataset functionality for use with models in tiatoolbox."""

<<<<<<< HEAD
from tiatoolbox.models.dataset.abc import ABCPatchDataset
=======
from tiatoolbox.models.dataset.abc import PatchDatasetABC
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
from tiatoolbox.models.dataset.classification import (
    PatchDataset,
    WSIPatchDataset,
    predefined_preproc_func,
)

<<<<<<< HEAD
from tiatoolbox.models.dataset.info import (
    ABCDatasetInfo,
    KatherPatchDataset
)
=======
from tiatoolbox.models.dataset.info import DatasetInfoABC, KatherPatchDataset
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
