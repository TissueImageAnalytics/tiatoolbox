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
# The Original Code is Copyright (C) 2021, TIACentre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Staing augmentation"""
import copy
import random

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

from tiatoolbox.tools.stainnorm import MacenkoNormaliser, VahadaneNormaliser
from tiatoolbox.utils.misc import get_luminosity_tissue_mask


class StainAugmentation(ImageOnlyTransform):
    """Stain augmentation using predefined stain matrix or stain extraction methods
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.


    Args:
        img:

    Examples:
         >>> from tiatoolbox.tools.stainaugment import StainAugmentaiton
    """

    def __init__(
        self,
        method: str = "vahadane",
        stain_matrix: np.ndarray = None,
        sigma1: float = 0.5,
        sigma2: float = 0.25,
        augment_background: bool = False,
        always_apply=False,
        p=0.5,
    ) -> np.ndarray:
        super(StainAugmentation, self).__init__(always_apply=always_apply, p=p)

        self.augment_background = augment_background
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.method = method
        self.stain_matrix = stain_matrix

        if self.method.lower() not in {"macenko", "vahadane"}:
            raise Exception(f"Invalid stain extractor method: {self.method}")

        if self.method.lower() == "macenko":
            self.stain_normaliser = MacenkoNormaliser()
        elif self.method.lower() == "vahadane":
            self.stain_normaliser = VahadaneNormaliser()

    def fit(self, img, threshold=0.85):
        if self.stain_matrix is None:
            self.stain_normaliser.fit(img)
            self.stain_matrix = self.stain_normaliser.stain_matrix_target
            self.source_concentrations = self.stain_normaliser.target_concentrations
        else:
            self.source_concentrations = self.stain_normaliser.get_concentrations(
                img, self.stain_matrix
            )
        self.n_stains = self.source_concentrations.shape[1]
        self.tissue_mask = get_luminosity_tissue_mask(img, threshold=threshold).ravel()
        self.img_shape = img.shape

    def augment(self):  # alpha, beta
        augmented_concentrations = copy.deepcopy(self.source_concentrations)
        for i in range(self.n_stains):
            if self.augment_background:
                augmented_concentrations[:, i] *= self.alpha
                augmented_concentrations[:, i] += self.beta
            else:
                augmented_concentrations[self.tissue_mask, i] *= self.alpha
                augmented_concentrations[self.tissue_mask, i] += self.beta

        img_augmented = 255 * np.exp(
            -1 * np.dot(augmented_concentrations, self.stain_matrix)
        )
        img_augmented = img_augmented.reshape(self.img_shape)
        img_augmented = np.clip(img_augmented, 0, 255)
        return np.uint8(img_augmented)

    def apply(self, img, **params):  # alpha=None, beta=None,
        """Fit the augmentor on the input image and return an augmented instance

        Args:
            img (::np.ndarray::): Input RGB image in the form of unit8 numpy array.
        Returns:
            img_augmented (::np.ndarray::): Stain augmented image with the same size
              and format as the input img.
        """
        self.fit(img, threshold=0.85)
        return self.augment()

    def get_params(self):
        self.alpha = random.uniform(1 - self.sigma1, 1 + self.sigma1)
        self.beta = random.uniform(-self.sigma2, self.sigma2)
        return {"alpha": self.alpha, "beta": self.beta}

    def get_params_dependent_on_targets(self, params):
        """Does nothing, added to resolve flake 8 error"""
        pass

    def get_transform_init_args_names(self):
        return ("method", "stain_matrix", "sigma1", "sigma2", "augment_background")
