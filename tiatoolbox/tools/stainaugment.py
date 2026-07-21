"""Stain augmentation."""

from __future__ import annotations

import copy

import numpy as np

from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.utils.misc import get_luminosity_tissue_mask


class StainAugmentor:
    """Stain augmentation using predefined stain matrices or stain extraction methods.

    This class performs stain augmentation on RGB histology images by modifying
    their stain concentration values. It supports two stain extraction methods:
    *Vahadane* and *Macenko*. The augmentation can operate either by extracting a
    stain matrix from the input image or by using a user-provided stain matrix.

    Providing a pre-extracted `stain_matrix` allows faster augmentation and enables
    the augmentor to behave like a stain normalizer when the augmentation parameters
    (`sigma1`, `sigma2`) are set to zero. This avoids the need for dictionary
    learning during stain matrix extraction and is suitable for efficient
    on-the-fly stain augmentation or normalization.

    Args:
        method (str):
            Stain extraction method to use. Supported values are `"vahadane"`
            (default) and `"macenko"`.
        stain_matrix (numpy.ndarray):
            Optional pre-extracted stain matrix for a target image. When supplied,
            the augmentor uses this matrix directly for stain normalization or
            faster augmentation. If `None`, the stain matrix is extracted from the
            input image using the selected method.
        sigma1 (float):
            Controls the scaling of stain concentrations. The scale factor `alpha`
            is sampled from the range `[1 - sigma1, 1 + sigma1]`. Default is `0.4`.
        sigma2 (float):
            Controls the additive shift of stain concentrations. The shift `beta`
            is sampled from the range `[-sigma2, sigma2]`. Default is `0.2`.
        augment_background (bool):
            Whether to apply stain augmentation to background pixels. If `False`
            (default), augmentation is applied only to tissue regions determined
            using a luminosity-based tissue mask.
        p (float):
            Probability of applying stain augmentation. If a random draw exceeds
            `p`, the input image is returned unchanged. This provides a simple way
            to control how often augmentation occurs during training.

    Attributes:
        stain_normalizer:
            Internal stain normalization object used for matrix and concentration
            extraction.
        stain_matrix (numpy.ndarray):
            Extracted or user-provided stain matrix.
        source_concentrations (numpy.ndarray):
            Stain concentration values extracted from the input image.
        n_stains (int):
            Number of stain channels in the concentration matrix. Typically `2`
            for H&E images.
        tissue_mask (numpy.ndarray):
            Boolean mask indicating tissue regions when background augmentation
            is disabled.

    Examples:
        >>> from tiatoolbox.tools.stainaugment import StainAugmentor
        >>> import numpy as np
        >>>
        >>> stain_matrix = np.array([
        ...     [0.91633014, -0.20408072, -0.34451435],
        ...     [0.17669817,  0.92528011,  0.33561059],
        ... ])
        >>>
        >>> augmentor = StainAugmentor(stain_matrix=stain_matrix, p=0.5)
        >>> augmentor.fit(img)
        >>> img_aug = augmentor.augment()

    """

    def __init__(
        self,
        method: str = "vahadane",
        stain_matrix: np.ndarray | None = None,
        sigma1: float = 0.4,
        sigma2: float = 0.2,
        p: float = 1.0,
        *,
        augment_background: bool = False,
    ) -> None:
        """Initialize StainAugmentor object."""
        self.method = method.lower()
        if self.method not in {"macenko", "vahadane"}:
            msg = (
                f"Unsupported stain extractor method {self.method!r}. "
                f"Choose either 'vahadane' or 'macenko'."
            )
            raise ValueError(
                msg,
            )

        self.stain_matrix = stain_matrix
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.augment_background = augment_background
        self.p = p

        self.stain_normalizer = get_normalizer(self.method)
        self.rng = np.random.default_rng()

        # Populated during fit()
        self.alpha: float
        self.beta: float
        self.img_shape: tuple[int, ...]
        self.tissue_mask: np.ndarray
        self.n_stains: int
        self.source_concentrations: np.ndarray
        self._original_img: np.ndarray

    def fit(self, img: np.ndarray, threshold: float = 0.85) -> None:
        """Extract stain matrix and concentrations from the input image."""
        self._original_img = img  # store original image for probability logic

        if self.stain_matrix is None:
            self.stain_normalizer.fit(img)
            self.stain_matrix = self.stain_normalizer.stain_matrix_target
            self.source_concentrations = self.stain_normalizer.target_concentrations
        else:
            self.source_concentrations = self.stain_normalizer.get_concentrations(
                img, self.stain_matrix
            )

        self.n_stains = self.source_concentrations.shape[1]

        if not self.augment_background:
            self.tissue_mask = get_luminosity_tissue_mask(
                img, threshold=threshold
            ).ravel()

        self.img_shape = img.shape

    def augment(self) -> np.ndarray:
        """Generate a stain-augmented image, applied with probability p."""
        if self.rng.random() > self.p:
            return self._original_img

        augmented_concentrations = copy.deepcopy(self.source_concentrations)

        for i in range(self.n_stains):
            self._sample_params()

            if self.augment_background:
                augmented_concentrations[:, i] *= self.alpha
                augmented_concentrations[:, i] += self.beta
            else:
                augmented_concentrations[self.tissue_mask, i] *= self.alpha
                augmented_concentrations[self.tissue_mask, i] += self.beta

        self.stain_matrix = np.asarray(self.stain_matrix)

        img_augmented = 255 * np.exp(
            -np.dot(augmented_concentrations, self.stain_matrix)
        )
        img_augmented = img_augmented.reshape(self.img_shape)
        img_augmented = np.clip(img_augmented, 0, 255)

        return img_augmented.astype(np.uint8)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Convenience wrapper: fit + augment."""
        self.fit(img)
        return self.augment()

    def _sample_params(self) -> None:
        """Generate random alpha/beta parameters."""
        rng = np.random.default_rng()
        self.alpha = rng.uniform(1 - self.sigma1, 1 + self.sigma1)
        self.beta = rng.uniform(-self.sigma2, self.sigma2)
