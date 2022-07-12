"""Methods of masking tissue and background."""

from abc import ABC, abstractmethod

import cv2
import numpy as np
from skimage.filters import threshold_otsu

from tiatoolbox.utils.misc import objective_power2mpp


class TissueMasker(ABC):
    """Base class for tissue maskers.

    Takes an image as in put and outputs a mask.

    """

    def __init__(self) -> None:
        super().__init__()
        self.fitted = False

    @abstractmethod
    def fit(self, images: np.ndarray, masks=None) -> None:
        """Fit the masker to the images and parameters.

        Args:
            images (:class:`numpy.ndarray`):
                List of images, usually WSI thumbnails. Expected shape is
                NHWC (number images, height, width, channels).
            masks (:class:`numpy.ndarray`):
                Target/ground-truth masks. Expected shape is NHW (n
                images, height, width).

        """

    @abstractmethod
    def transform(self, images: np.ndarray) -> np.ndarray:
        """Create and return a tissue mask.

        Args:
            images (:class:`numpy.ndarray`):
                RGB image, usually a WSI thumbnail.

        Returns:
            :class:`numpy.ndarray`:
                Map of semantic classes spatially over the WSI
                e.g. regions of tissue vs background.

        """
        if not self.fitted:
            raise Exception("Fit must be called before transform.")

    def fit_transform(self, images: np.ndarray, **kwargs) -> np.ndarray:
        """Perform :func:`fit` then :func:`transform`.

        Sometimes it can be more optimal to perform both at the same
        time for a single sample. In this case the base implementation
        of :func:`fit` followed by :func:`transform` can be overridden.

        Args:
            images (:class:`numpy.ndarray`):
                Image to create mask from.
            **kwargs (dict):
                Other key word arguments passed to fit.
        """
        self.fit(images, **kwargs)
        return self.transform(images)


class OtsuTissueMasker(TissueMasker):
    """Tissue masker which uses Otsu's method to determine background.

    Otsu's method.

    Examples:
        >>> from tiatoolbox.tools.tissuemask import OtsuTissueMasker
        >>> masker = OtsuTissueMasker()
        >>> masker.fit(thumbnail)
        >>> masks = masker.transform([thumbnail])

        >>> from tiatoolbox.tools.tissuemask import OtsuTissueMasker
        >>> masker = OtsuTissueMasker()
        >>> masks = masker.fit_transform([thumbnail])

    """

    def __init__(self) -> None:
        super().__init__()
        self.threshold = None

    def fit(self, images: np.ndarray, masks=None) -> None:
        """Find a binary threshold using Otsu's method.

        Args:
            images (:class:`numpy.ndarray`):
                List of images with a length 4 shape (N, height, width,
                channels).
            masks (:class:`numpy.ndarray`):
                Unused here, for API consistency.

        """
        images_shape = np.shape(images)
        if len(images_shape) != 4:
            raise ValueError(
                "Expected 4 dimensional input shape (N, height, width, 3)"
                f" but received shape of {images_shape}."
            )

        # Convert RGB images to greyscale
        grey_images = [x[..., 0] for x in images]
        if images_shape[-1] == 3:
            grey_images = np.zeros(images_shape[:-1], dtype=np.uint8)
            for n, image in enumerate(images):
                grey_images[n] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        pixels = np.concatenate([np.array(grey).flatten() for grey in grey_images])

        # Find Otsu's threshold for all pixels
        self.threshold = threshold_otsu(pixels)

        self.fitted = True

    def transform(self, images: np.ndarray) -> np.ndarray:
        """Create masks using the threshold found during :func:`fit`.


        Args:
            images (:class:`numpy.ndarray`):
                List of images with a length 4 shape (N, height, width,
                channels).

        Returns:
            :class:`numpy.ndarray`:
                List of images with a length 4 shape (N, height, width,
                channels).

        """
        super().transform(images)

        masks = []
        for image in images:
            grey = image[..., 0]
            if len(image.shape) == 3 and image.shape[-1] == 3:
                grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mask = (grey < self.threshold).astype(bool)
            masks.append(mask)

        return [mask]


class MorphologicalMasker(OtsuTissueMasker):
    """Tissue masker which uses a threshold and simple morphological operations.

    This method applies Otsu's threshold before a simple small region
    removal, followed by a morphological dilation. The kernel for the
    dilation is an ellipse of radius 64/mpp unless a value is given for
    kernel_size. MPP is estimated from objective power via
    func:`tiatoolbox.utils.misc.objective_power2mpp` if a power argument
    is given instead of mpp to the initialiser.

    For small region removal, the minimum area size defaults to the area
    of the kernel. If no mpp, objective power, or kernel_size arguments
    are given then the kernel defaults to a size of 1x1.

    The scale of the morphological operations can also be manually
    specified with the `kernel_size` argument, for example if the
    automatic scale from mpp or objective power is too large or small.

    Examples:
        >>> from tiatoolbox.tools.tissuemask import MorphologicalMasker
        >>> from tiatoolbox.wsicore.wsireader import WSIReader
        >>> wsi = WSIReader.open("slide.svs")
        >>> thumbnail = wsi.slide_thumbnail(32, "mpp")
        >>> masker = MorphologicalMasker(mpp=32)
        >>> masks = masker.fit_transform([thumbnail])

    An example reading a thumbnail from a file where the objective power
    is known:

        >>> from tiatoolbox.tools.tissuemask import MorphologicalMasker
        >>> from tiatoolbox.utils.misc import imread
        >>> thumbnail = imread("thumbnail.png")
        >>> masker = MorphologicalMasker(power=1.25)
        >>> masks = masker.fit_transform([thumbnail])

    """

    def __init__(
        self, *, mpp=None, power=None, kernel_size=None, min_region_size=None
    ) -> None:
        """Initialise a morphological masker.

        Args:
            mpp (float or tuple(float)):
                The microns per-pixel of the image to be masked. Used to
                calculate kernel_size a 64/mpp, optional.
            power (float or tuple(float)):
                The objective power of the image to be masked. Used to
                calculate kernel_size as 64/objective_power2mpp(power),
                optional.
            kernel_size (int or tuple(int)):
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
            kernel_size = np.max([32 / mpp, [1, 1]], axis=0)

        # Ensure kernel_size is a length 2 numpy array
        kernel_size = np.array(kernel_size)
        if kernel_size.size != 2:
            kernel_size = kernel_size.repeat(2)

        # Convert to an integer double/ pair
        self.kernel_size = tuple(np.round(kernel_size).astype(int))

        # Create structuring element for morphological operations
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)

        # Set min region size to kernel area if None
        if self.min_region_size is None:
            self.min_region_size = np.sum(self.kernel)

    def transform(self, images: np.ndarray):
        """Create masks using the found threshold followed by morphological operations.


        Args:
            images (:class:`numpy.ndarray`):
                List of images with a length 4 shape (N, height, width,
                channels).

        Returns:
            :class:`numpy.ndarray`:
                List of images with a length 4 shape (N, height, width,
                channels).

        """
        super().transform(images)

        results = []
        for image in images:
            if len(image.shape) == 3 and image.shape[-1] == 3:
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

            results.append(mask.astype(bool))
        return results
