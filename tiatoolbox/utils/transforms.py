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

    """

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGBA")

    composite = Image.fromarray(np.full(list(image.size[::-1]) + [4], fill,
                                        dtype=np.uint8))
    composite.alpha_composite(image)
    composite = np.asarray(composite.convert("RGB"))
    return composite
