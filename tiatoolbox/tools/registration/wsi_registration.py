import numpy as np
from skimage import exposure, filters, morphology


def match_histograms(image_a, image_b, disk_radius=3):
    """Image normalization function.

    This function performs histogram equalization to unify the
    appearance of an image pair.

    Args:
            image_a (:class:`numpy.ndarray`):
                    A grayscale image.
            image_b (:class:`numpy.ndarray`):
                    A grayscale image.
            disk_radius (int):
                    The radius of the disk-shaped footprint.

    Returns:
            :class:`numpy.ndarray`:
                    A normalized grayscale image.
            :class:`numpy.ndarray`:
                    A normalized grayscale image.

    """

    image_a, image_b = np.squeeze(image_a), np.squeeze(image_b)
    if len(image_a.shape) == 3 or len(image_b.shape) == 3:
        raise ValueError("The input images should be grayscale images.")

    entropy_a, entropy_b = filters.rank.entropy(
        image_a, morphology.disk(disk_radius)
    ), filters.rank.entropy(image_b, morphology.disk(disk_radius))
    if np.mean(entropy_a) > np.mean(entropy_b):
        image_b = exposure.match_histograms(image_b, image_a)
    else:
        image_a = exposure.match_histograms(image_a, image_b)

    return image_a, image_b
