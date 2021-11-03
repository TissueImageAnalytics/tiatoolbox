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


def stain_augment(
    image,
    method="vahadane",
    stain_matrix=None,
    alpha=None,
    beta=None,
    augment_background=False,
):
    if method.lower() == "macenko":
        stain_normaliser = MacenkoNormaliser()
    elif method.lower() == "vahadane":
        stain_normaliser = VahadaneNormaliser()
    else:
        raise Exception(f"Invalid stain extractor method: {method}")

    if stain_matrix is None:
        stain_normaliser.fit(image)
        stain_matrix = stain_normaliser.stain_matrix_target
        source_concentrations = stain_normaliser.target_concentrations
    else:
        source_concentrations = stain_normaliser.get_concentrations(image, stain_matrix)
    n_stains = source_concentrations.shape[1]
    tissue_mask = get_luminosity_tissue_mask(image, threshold=0.85).ravel()
    image_shape = image.shape

    augmented_concentrations = copy.deepcopy(source_concentrations)
    for i in range(n_stains):
        if augment_background:
            augmented_concentrations[:, i] *= alpha
            augmented_concentrations[:, i] += beta
        else:
            augmented_concentrations[tissue_mask, i] *= alpha
            augmented_concentrations[tissue_mask, i] += beta

    image_augmented = 255 * np.exp(-1 * np.dot(augmented_concentrations, stain_matrix))
    image_augmented = image_augmented.reshape(image_shape)
    image_augmented = np.clip(image_augmented, 0, 255)
    return np.uint8(image_augmented)


class StainAugmentation(ImageOnlyTransform):
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

    def apply(self, image, alpha=None, beta=None, **params):
        """Return an stain augmented image.

        Args:
            seed:
        Returns:
            image_augmented:
        """
        return stain_augment(
            image,
            method=self.method,
            stain_matrix=self.stain_matrix,
            augment_background=self.augment_background,
            alpha=alpha,
            beta=beta,
        )

    def get_params(self):
        alpha = random.uniform(1 - self.sigma1, 1 + self.sigma1)
        beta = random.uniform(-self.sigma2, self.sigma2)
        return {"alpha": alpha, "beta": beta}

    def get_transform_init_args_names(self):
        return ("method", "stain_matrix", "sigma1", "sigma2", "augment_background")
