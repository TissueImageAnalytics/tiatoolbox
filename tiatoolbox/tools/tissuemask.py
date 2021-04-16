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

from abc import ABC, abstractmethod

import cv2
import numpy as np

from tiatoolbox.utils.misc import objective_power2mpp


class TissueMasker(ABC):
    """Base class for tissue maskers.

    Takes an image as in put and outputs a mask.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fitted = False

    @abstractmethod
    def fit(self, image: np.ndarray, mask=None) -> None:
        """Fit the masker to the image and given key word parameters.

        Args:
            image (np.ndarray): RGB image, usually a WSI thumbnail.
            mask (np.ndarray): Target/ground-truth mask.
        """
        self.fitted = True

    @abstractmethod
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Create and return a tissue mask.

        Args:
            thumbnail (np.ndarray): RGB image, usually a WSI thumbnail.

        Returns:
            np.ndarary: Map of semantic classes spatially over the WSI
                e.g. regions of tissue vs background.
        """
        if not self.fitted:
            raise Exception("Fit must be called before transform.")

    def fit_transform(self, image: np.ndarray, **fit_params) -> np.ndarray:
        """Apply :func:`fit` and :func:`transform` in one step.

        Sometimes it can be more optimal to perform both at the same
        time for a single sample. In this case the base implementation
        of :fun:`fit` followed by :func:`transform` can be overridden.

        Args:
            image (np.ndarray): Image to create mask from.
            fit_params (dict): Generic key word arguments passed to fit.
        """
        self.fit(image, **fit_params)
        return self.transform(image)


class OtsuTissueMasker(TissueMasker):
    """Tissue masker which uses Otsu's method to determine background."""

    def __init__(self) -> None:
        super().__init__()
        self.threshold = None

    def fit(self, image: np.ndarray, mask=None) -> None:
        # find Otsu's threshold
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        self.threshold, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        self.fitted = True

    def transform(self, image: np.ndarray) -> np.ndarray:
        super().transform(image)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        mask = gray < self.threshold

        return mask.astype(bool)


class MorphologicalMasker(TissueMasker):
    """Tissue masker which uses a threshold and simple morphological operations.

    This method applies Otsu's threshold before a simple small region
    removal, followed by a morphological dilation. The kernel for the
    dilation is an ellipse of radius 64/mpp unless a value is given for
    kernel_size to :func:`fit`. MPP is estimated from objective power
    via func:`tiatoolbox.utils.misc.objective_power2mpp` if a power
    argument is given instead of mpp to the initialiser.
    For small region removal, the minimum area size defaults to the area
    of the kernel.
    """

    def __init__(
        self, *, mpp=None, power=None, kernel_size=None, min_region_size=None
    ) -> None:
        """Initialise a morphological masker.

        Args:
            mpp (float or tuple of float):
                The microns per-pixel of the image to be masked. Used to
                calculate kernel_size a 64/mpp, optional.
            power (float or tuple of float):
                The objective power of the image to be masked. Used to
                calculate kernel_size as 64/objective_power2mpp(power),
                optional.
            kernel_size (int or tuple of int):
                Size of elliptical kernel in x and y, optional.
            min_region_size (int):
                Minimum region size in pixels to consider as foreground.
                Defaults to area of the kernel.
        """
        super().__init__()

        self.min_region_size = min_region_size
        self.threshold = None

        # Check for conflicting arguments
        if sum(arg is not None for arg in [mpp, power, kernel_size]) > 1:
            raise ValueError("Only one of mpp, power, kernel_size can be given.")

        # Default to kernel_size of (1, 1) if no arguments given
        if all(arg is None for arg in [mpp, power, kernel_size]):
            kernel_size = np.array([1, 1])

        # Convert (objective) power approximately to MPP to unify units
        if power is not None:
            mpp = objective_power2mpp(power)

        # Convert MPP to an integer kernel_size
        if mpp is not None:
            mpp = np.array(mpp)
            if mpp.size != 2:
                mpp = mpp.repeat(2)
            kernel_size = np.max([64 / mpp, [1, 1]], axis=0)

        # Ensure kernel_size is a length 2 numpy array
        kernel_size = np.array(kernel_size)
        if kernel_size.size != 2:
            kernel_size = kernel_size.repeat(2)

        # Sanity check: At this point _kernel_size should not be None
        if kernel_size is None or None in kernel_size:
            raise AssertionError("Internal _kernel_size is None")

        # Convert to an integer double/ pair
        self.kernel_size = tuple(np.round(kernel_size).astype(int))

        # Create structuring element for morphological operations
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)

        # Set min region size to kernel area if None
        if self.min_region_size is None:
            self.min_region_size = np.sum(self.kernel)

    def fit(self, image: np.ndarray = None, mask=None) -> None:
        # Find Otsu's threshold
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        self.threshold, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        self.fitted = True

    def transform(self, image: np.ndarray):
        super().transform(image)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        mask = (gray < self.threshold).astype(np.uint8)

        _, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        sizes = stats[1:, -1]
        for i, size in enumerate(sizes):
            if size < self.min_region_size:
                mask[output == i + 1] = 0

        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel)

        return mask.astype(bool)
