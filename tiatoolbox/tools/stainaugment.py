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
    """Stain augmentation using predefined stain matrix or stain extraction methods.

    This stain augmentation class can be used in 'albumentations' augmentation pipelines
    as well as stand alone. There is an option to use predefined `stain_matrix` in the
    input which enables the `StainAugmentation` to generate augmented images faster or
    do stain normalization to a specific target `stain_matrix`. Having stain matrix
    beforhand, we don't need to do dictionary learning for stain matrix extraction,
    hence,speed up the stain augmentation/normalization process which makes it more
    appropriate for one-the-fly stain augmentation/normalization.

    Args:
        method (str): The method to use for stain matrix and stain concentration
            extraction. Can be either "vahadane" (default) or "macenko".
        stain_matrix (:class:`numpy.ndarray`): Pre-extracted stain matrix of a target
            image. This can be used for both on-the-fly stain normalization and faster
            stain augmentation. User can use tools in `tiatoolbox.tools.stainextract`
            to extract this information. If None (default), the stain matrix will be
            automatically extracted using the method specified by user.
        sigma1 (float): Controls the extent of the stain concentrations scale parameter
            (`alpha` belonging to [1-sigma1, 1+sigma1] range). Default is 0.5.
        sigma2 (float): Controls the extent of the stain concentrations shift parameter
            (`beta` belonging to [-sigma2, sigma2] range). Default is 0.25.
        augment_background (bool): Specifies whether to apply stain augmentation on the
            background or not. Default is False, which indicates that only tissue region
            will be stain augmented.
        always_apply (False): For use with 'albumentations' pipeline. Please refer to
            albumentations documentations for more information.
        p (0.5): For use with 'albumentations' pipeline which specifies the probability
            of using the augmentation in a 'albumentations' pipeline. . Please refer to
            albumentations documentations for more information.
    Attributes:
        stain_normaliser: Fitted stain normalization method.
        stain_matrix (:class:`numpy.ndarray`): extracted stain matrix from the image
        source_concentrations (:class:`numpy.ndarray`): extracted stain
            concentrations from the input image.
        n_stains (int): number of stain channels in the stain concentrations.
            Expected to be 2 for H&E stained images.
        tissue_mask (:class:`numpy.ndarray`): tissue region mask in the image.
    Examples:
        >>> '''Using the stain augmentor in the 'albumentations' pipeline'''
        >>> from tiatoolbox.tools.stainaugment import StainAugmentaiton
        >>> import albumentations as A
        >>> # Defining an examplar stain matrix as refrence
        >>> stain_matrix = np.array([[0.91633014, -0.20408072, -0.34451435],
        ...                [0.17669817, 0.92528011, 0.33561059]])
        >>> # Define albumentations pipeline
        >>> aug_pipline = A.Compose([
        ...                         A.RandomRotate90(),
        ...                         A.Flip(),
        ...                         StainAugmentaiton(stain_matrix=stain_matrix)
        ...                         ])
        >>> # apply the albumentations pipeline on an image (RGB numpy unit8 type)
        >>> img_aug = aug(image=img)['image']

        >>> '''Using the stain augmentor stand alone'''
        >>> from tiatoolbox.tools.stainaugment import StainAugmentaiton
        >>> # Defining an examplar stain matrix as refrence
        >>> stain_matrix = np.array([[0.91633014, -0.20408072, -0.34451435],
        ...                [0.17669817, 0.92528011, 0.33561059]])
        >>> # Instantiate the stain augmentor and fit it on an image
        >>> stain_augmentor = StainAugmentation(stain_matrix=stain_matrix)
        >>> stain_augmentor.fit(img)
        >>> # Now using the fitted `stain_augmentor` in a loop to generate
        >>> # several augmented instances from the same image.
        >>> for i in range(10):
        ...     img_aug = stain_augmentor.augment()
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

        self.alpha = None
        self.beta = None
        self.img_shape = None
        self.tissue_mask = None
        self.n_stains = None
        self.source_concentrations = None

    def fit(self, img, threshold=0.85):
        """Fit function to extract information needed for stain augmentation.

        The `fit` function uses either 'Macenko' or 'Vahadane' stain extraction methods
        to extract stain matrix and stain concentrations of the input image to be used
        in the `augment` function.

        Args:
            img (:class:`numpy.ndarray`): RGB image in the form of uint8 numpy array.
            threshold (float): The threshold value used to find tissue mask from the
                luminosity component of the image. The found `tissue_mask` will be used
                to filter out background area in stain augmentation process upon user
                setting `augment_background=False`.
        """
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

    def augment(self):
        """Return an augmented instance based on source stain concentrations.

        Stain concentrations of the source image are altered (scaled and shifted)
        based on the random alpha and beta paramters, and then an augmented image is
        reconstructed from the altered concentrations.
        All parameters needed for this part are calculated when calling fit() function.

        Returns:
            img_augmented (:class:`numpy.ndarray`): stain augmented image.
        """
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

    def apply(self, image, **params):  # alpha=None, beta=None,
        """Call the `fit` and `augment` functions to generate an stain augmented image.

        Args:
            img (:class:`numpy.ndarray`): Input RGB image in the form of unit8 numpy
                array.
        Returns:
            :class:`numpy.ndarray`: Stain augmented image with the same
                size and format as the input img.
        """
        self.fit(image, threshold=0.85)
        return self.augment()

    def get_params(self):
        """Returns randomly generated parameters based on input arguments."""
        self.alpha = random.uniform(1 - self.sigma1, 1 + self.sigma1)
        self.beta = random.uniform(-self.sigma2, self.sigma2)
        return {"alpha": self.alpha, "beta": self.beta}

    def get_params_dependent_on_targets(self, params):
        """Does nothing, added to resolve flake 8 error"""
        return {}

    def get_transform_init_args_names(self):
        """Return the argument names for albumentations use."""
        return ("method", "stain_matrix", "sigma1", "sigma2", "augment_background")
