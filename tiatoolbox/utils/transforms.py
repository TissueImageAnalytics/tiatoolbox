"""Define Image transforms."""
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image

from tiatoolbox.utils.misc import parse_cv2_interpolaton, select_cv2_interpolation


def background_composite(image, fill=255, alpha=False):
    """Image composite with specified background.

    Args:
        image (ndarray or PIL.Image):
            Input image.
        fill (int):
            Fill value for the background, defaults to 255.
        alpha (bool):
            True if alpha channel is required.

    Returns:
        :class:`numpy.ndarray`:
            Image with background composite.

    Examples:
        >>> from tiatoolbox.utils import transforms
        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> img_with_alpha = np.zeros((2000, 2000, 4)).astype('uint8')
        >>> img_with_alpha[:1000, :, 3] = 255 # edit alpha channel
        >>> img_back_composite = transforms.background_composite(
        ...     img_with_alpha
        ... )
        >>> plt.imshow(img_with_alpha)
        >>> plt.imshow(img_back_composite)
        >>> plt.show()

    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGBA")

    composite = Image.fromarray(
        np.full(list(image.size[::-1]) + [4], fill, dtype=np.uint8)
    )
    composite.alpha_composite(image)
    if not alpha:
        return np.asarray(composite.convert("RGB"))

    return np.asarray(composite)


def imresize(img, scale_factor=None, output_size=None, interpolation="optimise"):
    """Resize input image.

    Args:
        img (:class:`numpy.ndarray`):
            Input image, assumed to be in `HxWxC` or `HxW` format.
        scale_factor (tuple(float)):
            Scaling factor to resize the input image.
        output_size (tuple(int)):
            Output image size, (width, height).
        interpolation (str or int):
            Interpolation method used to interpolate the image using
            `opencv interpolation flags
            <https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html>`_
            default='optimise', uses cv2.INTER_AREA for scale_factor
            <1.0 otherwise uses cv2.INTER_CUBIC.

    Returns:
        :class:`numpy.ndarray`: Resized image. The image may be of different `np.dtype`
            compared to the input image. However, the numeric precision is ensured.

    Examples:
        >>> from tiatoolbox.wsicore import wsireader
        >>> from tiatoolbox.utils import transforms
        >>> wsi = wsireader.WSIReader(input_path="./CMU-1.ndpi")
        >>> slide_thumbnail = wsi.slide_thumbnail()
        >>> # Resize the image to half size using scale_factor 0.5
        >>> transforms.imresize(slide_thumbnail, scale_factor=0.5)

    """
    if scale_factor is None and output_size is None:
        raise TypeError("One of scale_factor and output_size must be not None.")
    if scale_factor is not None:
        scale_factor = np.array(scale_factor)
        if scale_factor.size == 1:
            scale_factor = np.repeat(scale_factor, 2)

    # Handle None arguments
    if output_size is None:
        width = int(img.shape[1] * scale_factor[0])
        height = int(img.shape[0] * scale_factor[1])
        output_size = (width, height)

    if scale_factor is None:
        scale_factor = img.shape[:2][::-1] / np.array(output_size)

    # Return original if scale factor is 1
    if np.all(scale_factor == 1.0):
        return img

    # Get appropriate cv2 interpolation enum
    if interpolation == "optimise":
        interpolation = select_cv2_interpolation(scale_factor)

    # a list of (original type, converted type) tuple
    # all `converted type` are np.dtypes that cv2.resize
    # can work on out-of-the-box (anything else will cause
    # error). The `converted type` has been selected so that
    # they can maintain the numeric precision of the `original type`.
    dtype_mapping = [
        (np.bool, np.uint8),
        (np.int8, np.int16),
        (np.int16, np.int16),
        (np.int32, np.float32),
        (np.uint8, np.uint8),
        (np.uint16, np.uint16),
        (np.uint32, np.float32),
        (np.int64, np.float64),
        (np.uint64, np.float64),
        (np.float16, np.float32),
        (np.float32, np.float32),
        (np.float64, np.float64),
    ]
    source_dtypes = [v[0] for v in dtype_mapping]
    original_dtype = img.dtype
    if original_dtype not in source_dtypes:
        raise ValueError(
            f"Does not support resizing for array of dtype: {original_dtype}"
        )

    converted_dtype = dtype_mapping[source_dtypes.index(original_dtype)][1]
    img = img.astype(converted_dtype)

    interpolation = parse_cv2_interpolaton(interpolation)

    # Resize the image
    # Handle case for 1x1 images which cv2 v4.5.4 no longer handles
    if img.shape[0] == img.shape[1] == 1:
        return img.repeat(output_size[1], 0).repeat(output_size[0], 1)

    if len(img.shape) == 3 and img.shape[-1] > 4:
        img_channels = [
            cv2.resize(img[..., ch], tuple(output_size), interpolation=interpolation)[
                ..., None
            ]
            for ch in range(img.shape[-1])
        ]
        return np.concatenate(img_channels, axis=-1)

    return cv2.resize(img, tuple(output_size), interpolation=interpolation)


def rgb2od(img):
    r"""Convert from RGB to optical density (:math:`OD_{RGB}`) space.

    .. math::
        RGB = 255 * exp^{-1*OD_{RGB}}

    Args:
        img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
            RGB image.

    Returns:
        :class:`numpy.ndarray`:
            Optical density (OD) RGB image.

    Examples:
        >>> from tiatoolbox.utils import transforms, misc
        >>> rgb_img = misc.imread('path/to/image')
        >>> od_img = transforms.rgb2od(rgb_img)

    """
    mask = img == 0
    img[mask] = 1
    return np.maximum(-1 * np.log(img / 255), 1e-6)


def od2rgb(od):
    r"""Convert from optical density (:math:`OD_{RGB}`) to RGB.

    .. math::
        RGB = 255 * exp^{-1*OD_{RGB}}

    Args:
        od (:class:`numpy.ndarray`):
            Optical density (OD) RGB image.

    Returns:
        :class:`numpy.ndarray`:
            RGB Image.

    Examples:
        >>> from tiatoolbox.utils import transforms, misc
        >>> rgb_img = misc.imread('path/to/image')
        >>> od_img = transforms.rgb2od(rgb_img)
        >>> rgb_img = transforms.od2rgb(od_img)

    """
    od = np.maximum(od, 1e-6)
    return (255 * np.exp(-1 * od)).astype(np.uint8)


def bounds2locsize(bounds, origin="upper"):
    """Calculate the size of a tuple of bounds.

    Bounds are expected to be in the `(left, top, right, bottom)` or
    `(start_x, start_y, end_x, end_y)` format.

    Args:
        bounds (tuple(int)):
            A 4-tuple or length 4 array of bounds values in `(left, top,
            right, bottom)` format.
        origin (str):
            Upper (Top-left) or lower (bottom-left) origin.
            Defaults to upper.

    Returns:
        tuple:
            A 2-tuple containing integer 2-tuples for location and size:

            - :py:obj:`tuple` - location tuple
                - :py:obj:`int` - x
                - :py:obj:`int` - y

            - :py:obj:`size` - size tuple
                - :py:obj:`int` - width
                - :py:obj:`int` - height

    Examples:
        >>> from tiatoolbox.utils.transforms import bounds2locsize
        >>> bounds = (0, 0, 10, 10)
        >>> location, size = bounds2locsize(bounds)

        >>> from tiatoolbox.utils.transforms import bounds2locsize
        >>> _, size = bounds2locsize((12, 4, 24, 16))

    """
    left, top, right, bottom = bounds
    origin = origin.lower()
    if origin == "upper":
        return np.array([left, top]), np.array([right - left, bottom - top])
    if origin == "lower":
        return np.array([left, bottom]), np.array([right - left, top - bottom])
    raise ValueError("Invalid origin. Only 'upper' or 'lower' are valid.")


def locsize2bounds(location, size):
    """Convert a location and size to bounds.

    Args:
        location (tuple(int)):
            A 2-tuple or length 2 array of x,y coordinates.
        size (tuple(int)):
            A 2-tuple or length 2 array of width and height.

    Returns:
        tuple:
            A tuple of bounds:
            - :py:obj:`int` - left / start_x
            - :py:obj:`int` - top / start_y
            - :py:obj:`int` - right / end_x
            - :py:obj:`int` - bottom / end_y

    """
    return (
        location[0],
        location[1],
        location[0] + size[0],
        location[1] + size[1],
    )


def bounds2slices(
    bounds: Tuple[int, int, int, int],
    stride: Union[int, Tuple[int, int, Tuple[int, int]]] = 1,
) -> Tuple[slice]:
    """Convert bounds to slices.

    Create a tuple of slices for each start/stop pair in bounds.

    Arguments:
        bounds (tuple(int)):
            Iterable of integer bounds. Must be even in length with the
            first half as starting values and the second half as end
            values, e.g. (start_x, start_y, stop_x, stop_y).
        stride (int):
            Stride to apply when converting to slices.

    Returns:
        tuple of slice:
            Tuple of slices in image read order (y, x, channels).

    Example:
        >>> from tiatoolbox.utils.transforms import bounds2slices
        >>> import numpy as np
        >>> bounds = (5, 5, 10, 10)
        >>> array = np.ones((10, 10, 3))
        >>> slices = bounds2slices(bounds)
        >>> region = array[slices, ...]

    """
    if np.size(stride) not in [1, 2]:
        raise ValueError("Invalid stride shape.")
    if np.size(stride) == 1:
        stride = np.tile(stride, 4)
    elif np.size(stride) == 2:  # pragma: no cover
        stride = np.tile(stride, 2)

    start, stop = np.reshape(bounds, (2, -1)).astype(int)
    slice_array = np.stack([start[::-1], stop[::-1]], axis=1)
    return tuple(slice(*x, s) for x, s in zip(slice_array, stride))


def pad_bounds(
    bounds: Tuple[int, int, int, int],
    padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int]:
    """Add padding to bounds.

    Arguments:
        bounds (tuple(int)):
            Iterable of integer bounds. Must be even in length with the
            first half as starting values and the second half as end
            values, e.g. (start_x, start_y, stop_x, stop_y).
        padding (int):
            Padding to add to bounds.

    Examples:
        >>> pad_bounds((0, 0, 0, 0), 1)

    Returns:
        tuple of int:
            Tuple of bounds with padding to the edges.

    """
    if np.size(bounds) % 2 != 0:
        raise ValueError("Bounds must have an even number of elements.")
    ndims = np.size(bounds) // 2

    if np.size(padding) not in [1, 2, np.size(bounds)]:
        raise ValueError("Invalid number of padding elements.")

    if np.size(padding) == 1 or np.size(padding) == np.size(bounds):
        pass
    elif np.size(padding) == ndims:  # pragma: no cover
        padding = np.tile(padding, 2)

    signs = np.repeat([-1, 1], ndims)
    return np.add(bounds, padding * signs)
