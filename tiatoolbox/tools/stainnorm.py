"""Stain normalization classes."""

from __future__ import annotations

import cv2
import numpy as np

from tiatoolbox.tools.stainextract import (
    CustomExtractor,
    MacenkoExtractor,
    RuifrokExtractor,
    VahadaneExtractor,
)
from tiatoolbox.utils.exceptions import MethodNotSupportedError
from tiatoolbox.utils.misc import load_stain_matrix
from tiatoolbox.utils.transforms import od2rgb, rgb2od


class StainNormalizer:
    """Stain normalization base class.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Attributes:
        extractor (CustomExtractor, RuifrokExtractor):
            Method specific stain extractor.
        stain_matrix_target (:class:`numpy.ndarray`):
            Stain matrix of target.
        target_concentrations (:class:`numpy.ndarray`):
            Stain concentration matrix of target.
        maxC_target (:class:`numpy.ndarray`):
            99th percentile of each stain.
        stain_matrix_target_RGB (:class:`numpy.ndarray`):
            Target stain matrix in RGB.

    """

    def __init__(self: StainNormalizer) -> None:
        """Initialize :class:`StainNormalizer`."""
        self.extractor: (
            CustomExtractor | MacenkoExtractor | RuifrokExtractor | VahadaneExtractor
        )
        self.stain_matrix_target: np.ndarray
        self.target_concentrations: np.ndarray
        self.maxC_target = None
        self.stain_matrix_target_RGB: np.ndarray

    @staticmethod
    def get_concentrations(img: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        """Estimate concentration matrix given an image and stain matrix.

        Args:
            img (:class:`numpy.ndarray`):
                Input image.
            stain_matrix (:class:`numpy.ndarray`):
                Stain matrix for haematoxylin and eosin stains.

        Returns:
            numpy.ndarray:
                Stain concentrations of input image.

        """
        od = rgb2od(img).reshape((-1, 3))
        x, _, _, _ = np.linalg.lstsq(stain_matrix.T, od.T, rcond=-1)
        return x.T

    def fit(self: StainNormalizer, target: np.ndarray) -> None:
        """Fit to a target image.

        Args:
            target (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
              Target/reference image.

        """
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_concentrations = self.get_concentrations(
            target,
            self.stain_matrix_target,
        )
        self.maxC_target = np.percentile(
            self.target_concentrations,
            99,
            axis=0,
        ).reshape((1, 2))
        # useful to visualize.
        self.stain_matrix_target_RGB = od2rgb(self.stain_matrix_target)

    def transform(self: StainNormalizer, img: np.ndarray) -> np.ndarray:
        """Transform an image.

        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                RGB input source image.

        Returns:
            :class:`numpy.ndarray`:
                RGB stain normalized image.

        """
        stain_matrix_source = self.extractor.get_stain_matrix(img)
        source_concentrations = self.get_concentrations(img, stain_matrix_source)
        max_c_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= self.maxC_target / max_c_source
        trans = 255 * np.exp(
            -1 * np.dot(source_concentrations, self.stain_matrix_target),
        )

        # ensure between 0 and 255
        trans[trans > 255] = 255  # noqa: PLR2004
        trans[trans < 0] = 0

        return trans.reshape(img.shape).astype(np.uint8)


class CustomNormalizer(StainNormalizer):
    """Stain Normalization using a user-defined stain matrix.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Args:
        stain_matrix (:class:`numpy.ndarray`):
            User-defined stain matrix. Must be either 2x3 or 3x3.

    Examples:
        >>> from tiatoolbox.tools.stainnorm import CustomNormalizer
        >>> norm = CustomNormalizer(stain_matrix)
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(source_img)

    """

    def __init__(self: CustomNormalizer, stain_matrix: np.ndarray) -> None:
        """Initialize :class:`CustomNormalizer`."""
        super().__init__()

        self.extractor = CustomExtractor(stain_matrix)


class RuifrokNormalizer(StainNormalizer):
    """Ruifrok & Johnston stain normalizer.

    Normalize a patch to the stain appearance of the target image using
    the method of:

    Ruifrok, Arnout C., and Dennis A. Johnston. "Quantification of
    histochemical staining by color deconvolution." Analytical and
    quantitative cytology and histology 23.4 (2001): 291-299.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Examples:
        >>> from tiatoolbox.tools.stainnorm import RuifrokNormalizer
        >>> norm = RuifrokNormalizer()
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(source_img)

    """

    def __init__(self: RuifrokNormalizer) -> None:
        """Initialize :class:`RuifrokNormalizer`."""
        super().__init__()
        self.extractor = RuifrokExtractor()


class MacenkoNormalizer(StainNormalizer):
    """Macenko stain normalizer.

    Normalize a patch to the stain appearance of the target image using
    the method of:

    Macenko, Marc, et al. "A method for normalizing histology slides for
    quantitative analysis." 2009 IEEE International Symposium on
    Biomedical Imaging: From Nano to Macro. IEEE, 2009.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Examples:
        >>> from tiatoolbox.tools.stainnorm import MacenkoNormalizer
        >>> norm = MacenkoNormalizer()
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(source_img)

    """

    def __init__(self: MacenkoNormalizer) -> None:
        """Initialize :class:`MacenkoNormalizer`."""
        super().__init__()
        self.extractor = MacenkoExtractor()


class VahadaneNormalizer(StainNormalizer):
    """Vahadane stain normalizer.

    Normalize a patch to the stain appearance of the target image using
    the method of:

    Vahadane, Abhishek, et al. "Structure-preserving color normalization
    and sparse stain separation for histological images." IEEE
    transactions on medical imaging 35.8 (2016): 1962-1971.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Examples:
        >>> from tiatoolbox.tools.stainnorm import VahadaneNormalizer
        >>> norm = VahadaneNormalizer()
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(source_img)

    """

    def __init__(self: VahadaneNormalizer) -> None:
        """Initialize :class:`VahadaneNormalizer`."""
        super().__init__()
        self.extractor = VahadaneExtractor()


class ReinhardNormalizer(StainNormalizer):
    """Reinhard colour normalizer.

    Normalize a patch colour to the target image using the method of:

    Reinhard, Erik, et al. "Color transfer between images." IEEE
    Computer graphics and applications 21.5 (2001): 34-41.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Attributes:
        target_means (float):
            Mean of each LAB channel.
        target_stds (float):
            Standard deviation of each LAB channel.

    Examples:
        >>> from tiatoolbox.tools.stainnorm import ReinhardNormalizer
        >>> norm = ReinhardNormalizer()
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(src_img)

    """

    def __init__(self: ReinhardNormalizer) -> None:
        """Initialize :class:`ReinhardNormalizer`."""
        super().__init__()
        self.target_means: tuple[float, float, float]
        self.target_stds: tuple[float, float, float]

    def fit(self: ReinhardNormalizer, target: np.ndarray) -> None:
        """Fit to a target image.

        Args:
            target (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                Target image.

        """
        means, stds = self.get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def transform(self: ReinhardNormalizer, img: np.ndarray) -> np.ndarray:
        """Transform an image.

        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                Input image.

        Returns:
            :class:`numpy.ndarray` of type :class:`numpy.float`:
                Colour normalized RGB image.

        """
        chan1, chan2, chan3 = self.lab_split(img)
        means, stds = self.get_mean_std(img)
        norm1 = (
            (chan1 - means[0]) * (self.target_stds[0] / stds[0])
        ) + self.target_means[0]
        norm2 = (
            (chan2 - means[1]) * (self.target_stds[1] / stds[1])
        ) + self.target_means[1]
        norm3 = (
            (chan3 - means[2]) * (self.target_stds[2] / stds[2])
        ) + self.target_means[2]
        return self.merge_back(norm1, norm2, norm3)

    @staticmethod
    def lab_split(img: np.ndarray) -> tuple[float, float, float]:
        """Convert from RGB uint8 to LAB and split into channels.

        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                Input image.

        Returns:
            tuple:
                - :py:obj:`float`:
                    L channel in LAB colour space.
                - :py:obj:`float`:
                    A channel in LAB colour space.
                - :py:obj:`float`:
                    B channel in LAB colour space.

        """
        img = img.astype("uint8")  # ensure input image is uint8
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_float = img.astype(np.float32)
        chan1, chan2, chan3 = cv2.split(img_float)
        chan1 /= 2.55  # should now be in range [0,100]
        chan2 -= 128.0  # should now be in range [-127,127]
        chan3 -= 128.0  # should now be in range [-127,127]
        return chan1, chan2, chan3

    @staticmethod
    def merge_back(chan1: float, chan2: float, chan3: float) -> np.ndarray:
        """Take separate LAB channels and merge back to give RGB uint8.

        Args:
            chan1 (float):
                L channel.
            chan2 (float):
                A channel.
            chan3 (float):
                B channel.

        Returns:
            :class:`numpy.ndarray`:
                Merged image.

        """
        chan1 *= 2.55  # should now be in range [0,255]
        chan2 += 128.0  # should now be in range [0,255]
        chan3 += 128.0  # should now be in range [0,255]
        img = np.clip(cv2.merge((chan1, chan2, chan3)), 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    def get_mean_std(
        self: ReinhardNormalizer,
        img: np.ndarray,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Get mean and standard deviation of each channel.

        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                Input image.

        Returns:
            tuple:
                - :py:obj:`float` - Means:
                    Mean values for each RGB channel.
                - :py:obj:`float` - Standard deviations:
                    Standard deviation for each RGB channel.

        """
        img = img.astype("uint8")  # ensure input image is uint8
        chan1, chan2, chan3 = self.lab_split(img)
        m1, sd1 = cv2.meanStdDev(chan1)
        m2, sd2 = cv2.meanStdDev(chan2)
        m3, sd3 = cv2.meanStdDev(chan3)
        means = m1, m2, m3
        stds = sd1, sd2, sd3
        return means, stds


def get_normalizer(
    method_name: str,
    stain_matrix: np.ndarray | None = None,
) -> StainNormalizer:
    """Return a :class:`.StainNormalizer` with corresponding name.

    Args:
        method_name (str):
            Name of stain norm method, must be one of "reinhard",
            "custom", "ruifrok", "macenko" or "vahadane".
        stain_matrix (:class:`numpy.ndarray` or str or pathlib.Path):
            User-defined stain matrix. This must either be a numpy array
            or a path to either a .csv or .npy file. This is only
            utilised if using "custom" method name.

    Returns:
        StainNormalizer:
            An object with base :class:'.StainNormalizer' as base class.

    Examples:
        >>> from tiatoolbox.tools.stainnorm import get_normalizer
        >>> norm = get_normalizer('Reinhard')
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(source_img)

    """
    if method_name.lower() not in [
        "reinhard",
        "ruifrok",
        "macenko",
        "vahadane",
        "custom",
    ]:
        raise MethodNotSupportedError

    if stain_matrix is not None and method_name.lower() != "custom":
        msg = '`stain_matrix` is only defined when using `method_name`="custom".'
        raise ValueError(
            msg,
        )

    if method_name.lower() == "reinhard":
        return ReinhardNormalizer()
    if method_name.lower() == "ruifrok":
        return RuifrokNormalizer()
    if method_name.lower() == "macenko":
        return MacenkoNormalizer()
    if method_name.lower() == "vahadane":
        return VahadaneNormalizer()

    if stain_matrix is None:
        msg = '`stain_matrix` is None when using `method_name`="custom".'
        raise ValueError(
            msg,
        )
    return CustomNormalizer(load_stain_matrix(stain_matrix))
