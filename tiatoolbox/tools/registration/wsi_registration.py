import numpy as np
from skimage import exposure, filters, morphology


def match_histograms(image_a, image_b, disk_size=3):
    """Image normalization function.

    This function performs histogram equalization to unify the
    appearance of an image pair.

    Args:
        fixed_img (:class:`numpy.ndarray`):
            A grayscale fixed image.
        moving_img (:class:`numpy.ndarray`):
            A grayscale moving image.

    Returns:
        :class:`numpy.ndarray`:
            A normalized grayscale fixed image.
        :class:`numpy.ndarray`:
            A normalized grayscale moving image.

    """

    image_a, moving_img = np.squeeze(image_a), np.squeeze(image_b)
    if len(image_a.shape) == 3 or len(image_b.shape) == 3:
        raise ValueError("The input images should be grayscale images.")

    moving_entropy, fixed_entropy = filters.rank.entropy(
        image_b, morphology.disk(disk_size)
    ), filters.rank.entropy(image_a, morphology.disk(disk_size))
    if np.mean(fixed_entropy) > np.mean(moving_entropy):
        image_b = exposure.match_histograms(image_b, image_a)
    else:
        image_a = exposure.match_histograms(image_a, image_b)

    return image_a, image_b
