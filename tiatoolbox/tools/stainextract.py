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
# This file contains code inspired by StainTools
# [https://github.com/Peter554/StainTools] written by Peter Byfield.
#
# The Original Code is Copyright (C) 2020, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Stain normalisation utilities used in tiatoolbox"""
import numpy as np


class CustomExtractor:
    """Get the user-defined stain matrix

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield

    """

    def __init__(self, stain_matrix):
        self.stain_matrix = stain_matrix
        if self.stain_matrix.shape != (2, 3) and self.stain_matrix.shape != (3, 3):
            raise Exception("Stain matrix must be either (2,3) or (3,3)")

    def get_stain_matrix(self, _):
        return self.stain_matrix


class RuifrokExtractor:
    """Get the stain matrix as defined in:

    Ruifrok, Arnout C., and Dennis A. Johnston. "Quantification of
    histochemical staining by color deconvolution." Analytical and
    quantitative cytology and histology 23.4 (2001): 291-299.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield

    """

    @staticmethod
    def get_stain_matrix(_):
        return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])


class MacenkoExtractor:
    """Get the stain matrix as defined in:

    Macenko, Marc, et al. "A method for normalizing histology
    slides for quantitative analysis." 2009 IEEE International
    Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield

    """

    @staticmethod
    def get_stain_matrix(_):
        return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])


class VahadaneExtractor:
    """Get the stain matrix as defined in:

    Vahadane, Abhishek, et al. "Structure-preserving color normalization
    and sparse stain separation for histological images."
    IEEE transactions on medical imaging 35.8 (2016): 1962-1971.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield

    """

    @staticmethod
    def get_stain_matrix(_):
        return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
