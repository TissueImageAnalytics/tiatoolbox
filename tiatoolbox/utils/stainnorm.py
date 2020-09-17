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
# The Original Code is Copyright (C) 2006, Blender Foundation
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Stain normalisation utilities used in tiatoolbox"""
import numpy as np
from abc import ABC, abstractmethod


class ABCStainExtractor(ABC):
    """Abstract base class for stain extraction"""

    @staticmethod
    @abstractmethod
    def get_stain_matrix(img):
        """Estimate the stain matrix given an image.
        Args:
            img (ndarray): input image

        """


class RuifrokStainExtractor(ABCStainExtractor):
    """Normalize a patch colour to the target image using the method of:

    A.C. Ruifrok & D.A. Johnston 'Quantification of histochemical staining
    by color deconvolution'. Analytical and quantitative cytology and histology
    / the International Academy of Cytology and American Society of Cytology

    """

    @staticmethod
    def get_stain_matrix(_):
        """Uses pre-defined stain matrix. Top row corresponds
        to haematoxylin and bottom row corresponds to eosin

        """
        return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
