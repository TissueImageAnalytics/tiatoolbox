"""Define Image transforms"""
import numpy as np
from PIL import Image


def background_composite(image, fill=255):
    """Image composite with specified background

    Args:
        image (ndarray, PIL.Image): input image
        fill (int): fill value for the background, default=255

    Returns:
        ndarray: image with background composite

    Examples:
        >>> from tiatoolbox.utils import transforms
        >>> from matplotlib import pyplot as plt
        >>> img_with_alpha = np.zeros((2000, 2000, 4)).astype('uint8')
        >>> img_with_alpha[:1000, :, 3] = 255 # edit alpha channel
        >>> img_back_composite = transforms
        ...     .background_composite(img_with_alpha)
        >>> plt.imshow(img_with_alpha)
        >>> plt.imshow(img_back_composite)

    """

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGBA")

    composite = Image.fromarray(np.full(list(image.size[::-1]) + [4], fill,
                                        dtype=np.uint8))
    composite.alpha_composite(image)
    composite = np.asarray(composite.convert("RGB"))
    return composite
