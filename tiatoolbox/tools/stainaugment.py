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

"""Staing augmentation"""
import numpy as np
import copy
from tiatoolbox.tools.stainnorm import VahadaneNormaliser
from tiatoolbox.utils.misc import get_luminosity_tissue_mask


class StainAugmentation:
    """Stain augmentation using predefined stain matrix or stain extraction methods
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.


    Args:
        image:

    Examples:
         >>> from tiatoolbox.tools.stainaugment import StainAugmentaiton
    """

    def __init__(
        self,
        image,
        source_stain_matrix=None,
        sigma1=0.2,
        sigma2=0.2,
        augment_background=False,
    ):
        self.augment_background = augment_background
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.stain_normaliser = VahadaneNormaliser()
        if source_stain_matrix is None:
            self.stain_normaliser.fit(image)
            self.source_stain_matrix = self.stain_normaliser.target_stain_matrix
            self.source_concentrations = self.stain_normaliser.target_concentrations
        else:
            self.source_stain_matrix = source_stain_matrix
            self.source_concentrations = self.stain_normaliser.get_concentrations(
                image, source_stain_matrix
            )
        self.n_stains = self.source_concentrations.shape[1]
        self.tissue_mask = get_luminosity_tissue_mask(image, threshold=0.85).ravel()
        self.image_shape = image.shape

    def augment(self, seed=None):
        """Return an stain augmented image.

        Args:
            seed:
        Returns:
            I_augmented:
        """
        if seed is not None:
            np.random.seed(seed=seed)
        augmented_concentrations = copy.deepcopy(self.source_concentrations)
        for i in range(self.n_stains):
            alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
            beta = np.random.uniform(-self.sigma2, self.sigma2)
            if self.augment_background:
                augmented_concentrations[:, i] *= alpha
                augmented_concentrations[:, i] += beta
            else:
                augmented_concentrations[self.tissue_mask, i] *= alpha
                augmented_concentrations[self.tissue_mask, i] += beta

        I_augmented = 255 * np.exp(
            -1 * np.dot(augmented_concentrations, self.source_stain_matrix)
        )
        I_augmented = I_augmented.reshape(self.image_shape)
        I_augmented = np.clip(I_augmented, 0, 255)
        return np.uint8(I_augmented)
