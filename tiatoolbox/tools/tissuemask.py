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

"""Methods of masking tissue and background."""

from typing import Union, Tuple, Optional
from abc import ABC, abstractmethod
import warnings

import cv2
import numpy as np

from tiatoolbox.utils.misc import objective_power2mpp


class TissueMasker(ABC):
    def fit(self, image=None, **kwargs):
        """Fit the masker to the image and given key word parameters.

        Args:
            image (np.ndarray): RGB image, usually a WSI thumbnail.
            kwargs (dict): Generic key word arguments.
        """

    @abstractmethod
    def transform(self, image: np.ndarray):
        """Create and return a tissue mask.

        Args:
            thumbnail (np.ndarray): RGB image, usually a WSI thumbnail.

        Returns:
            np.ndarary: Map of semantic classes spatially over the WSI
                e.g. regions of tissue vs background.
        """
        return

    def fit_transform(self, image: np.ndarray, **kwargs):
        """Apply fit and transform in one step.

        Args:
            image (np.ndarray): Image to create mask from.
            kwargs (dict): Generic key word arguments passed to fit.
        """
        self.fit(image, **kwargs)
        return self.transform(image)


class OtsuTissueMasker(TissueMasker):
    """Tissue masker which uses Otsu's method to determine background."""

    def transform(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        mask, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return mask


class MorphologicalMasker(TissueMasker):
    """Tissue masker which uses a threshold and simple morphological operations.

    This method applies Otsu's threshold before a simple small region
    removal, followed by a morphological dilation. The kernel for the
    dilation is an ellipse of radius 64/mpp unless a value is given for
    kernel_size to :func:`fit`. MPP is estimated from objective power
    via func:`tiatoolbox.utils.misc.objective_power2mpp` if unknown. For
    small region removal, the area of the kernel is used as a threshold.
    """

    def __init__(self) -> None:
        super().__init__()
        self.kernel_size = np.array([1, 1])
        self.min_region_size = 3
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)

    def fit(self, image: np.ndarray = None, **kwargs) -> np.ndarray:
        """Fit the masker to the image and given key word parameters.

        At least one of mpp, power, or kernel_size is required.

        Args:
            thumbnail (np.ndarray): An RGB thumnail image.
            kwargs (dict): Key word arguements.
            kernel_size (int or tuple of int): Manual kernel size, optional.
            mpp (float or tuple of float): The microns per-pixel of
                the image. Used to calculate kernel_size, optional.
            power (float or tuple of float): The objective
                power of the image. Used to calculate kernel_size, optional.
        """
        mpp = kwargs.get("mpp")
        power = kwargs.get("power")
        kernel_size = kwargs.get("kernel_size")

        # Check kwargs
        if all(arg is None for arg in [mpp, power, kernel_size]):
            warnings.warn(
                "No mpp, power, or kernel_size given. "
                "Using default kernel size of 1. "
                "Masking may be inconsistent across slides of different resolutions."
            )
            kernel_size = self.kernel_size

        if sum(arg is not None for arg in [mpp, power, kernel_size]) > 1:
            raise ValueError("Only one of mpp, power, kernel_size can be given.")

        # Convert (objective) power to MPP
        if power is not None:
            mpp = objective_power2mpp(power)

        # Normalise args to be either None or a length 2 numpy array
        if mpp is not None:
            mpp = np.array(mpp)
            if mpp.size != 2:
                mpp = mpp.repeat(2)
            kernel_size = np.max([64 / mpp, [1, 1]], axis=0)
        if kernel_size is not None:
            kernel_size = np.array(kernel_size)
            if kernel_size.size != 2:
                kernel_size = kernel_size.repeat(2)

        self.kernel_size = kernel_size

        # Scale the kernel with MPP for consistent results across resolutions
        if mpp is not None:
            self.kernel_size = np.max([64 / mpp, [1, 1]], axis=0)

        # At this point _kernel_size should not be None
        if self.kernel_size is None or None in self.kernel_size:
            raise AssertionError("Internal _kernel_size is None")

        self.kernel_size = tuple(np.round(self.kernel_size).astype(int))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)
        self.min_region_size = np.sum(self.kernel)

    def transform(self, image: np.ndarray):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        _, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        sizes = stats[1:, -1]
        for i, size in enumerate(sizes):
            if size < self.min_region_size:
                mask[output == i + 1] = 0

        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel)

        return mask
