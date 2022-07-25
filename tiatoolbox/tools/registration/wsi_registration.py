import numpy as np
from skimage import exposure, filters, morphology


def preprocess(fixed_img, moving_img):
    """This function performs normalization to unify the appearance of fixed
     and moving images

    Args:
            fixed_img (:class:`numpy.ndarray`):
                    A grayscale fixed image
            moving_img (:class:`numpy.ndarray`):
                    A grayscale moving image

    Returns:
            :class:`numpy.ndarray`:
                    A normalized grayscale fixed image
            :class:`numpy.ndarray`:
                    A normalized grayscale moving image

    """
    if len(fixed_img.shape) != 2 or len(moving_img.shape) != 2:
        raise ValueError("The input images should be grayscale images.")

    moving_entropy, fixed_entropy = filters.rank.entropy(
        moving_img, morphology.disk(3)
    ), filters.rank.entropy(fixed_img, morphology.disk(3))
    if np.mean(fixed_entropy) > np.mean(moving_entropy):
        moving_img = exposure.match_histograms(moving_img, fixed_img)
    else:
        fixed_img = exposure.match_histograms(fixed_img, moving_img)

    return fixed_img, moving_img
