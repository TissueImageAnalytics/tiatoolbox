"""Define Image transforms."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from tiatoolbox.utils.misc import parse_cv2_interpolaton, select_cv2_interpolation


def background_composite(
    image: np.ndarray | Image.Image,
    fill: int = 255,
    *,
    alpha: bool,
) -> np.ndarray:
    """Image composite with specified background.

    Args:
        image (ndarray or :class:`Image`):
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
        np.full([*list(image.size[::-1]), 4], fill, dtype=np.uint8),
    )
    composite.alpha_composite(image)
    if not alpha:
        return np.asarray(composite.convert("RGB"))

    return np.asarray(composite)


def _convert_scalar_to_width_height(array: np.ndarray) -> np.ndarray:
    """Converts scalar numpy array to specify width and height."""
    if array.size == 1:
        return np.repeat(array, 2)

    return array


def _get_scale_factor_array(
    scale_factor: float | tuple[float, float] | None,
) -> np.ndarray | None:
    """Converts scale factor to appropriate format required by imresize."""
    if scale_factor is not None:
        scale_factor_array = np.array(scale_factor, dtype=float)
        return _convert_scalar_to_width_height(scale_factor_array)
    return scale_factor


def _get_output_size_array(
    img: np.ndarray,
    output_size: int | tuple[int, int] | None,
    scale_factor_array: np.ndarray | None,
) -> np.ndarray:
    """Converts output size to appropriate format required by imresize."""
    # Handle None arguments
    if output_size is None and scale_factor_array is not None:
        width = int(img.shape[1] * scale_factor_array[0])
        height = int(img.shape[0] * scale_factor_array[1])
        return np.array((width, height))

    return _convert_scalar_to_width_height(np.array(output_size))


def imresize(
    img: np.ndarray,
    scale_factor: float | tuple[float, float] | None = None,
    output_size: int | tuple[int, int] | None = None,
    interpolation: str = "optimise",
) -> np.ndarray:
    """Resize input image.

    Args:
        img (:class:`numpy.ndarray`):
            Input image, assumed to be in `HxWxC` or `HxW` format.
        scale_factor (float or Tuple[float, float]):
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
        msg = "One of scale_factor and output_size must be not None."
        raise TypeError(msg)

    scale_factor_array = _get_scale_factor_array(scale_factor)
    output_size_array = _get_output_size_array(
        img=img,
        output_size=output_size,
        scale_factor_array=scale_factor_array,
    )

    if scale_factor is None:
        scale_factor_array = img.shape[:2][::-1] / np.array(output_size_array)

    # Return original if scale factor is 1
    if np.all(scale_factor_array == 1.0):  # noqa: PLR2004
        return img

    # Get appropriate cv2 interpolation enum
    if interpolation == "optimise":
        interpolation = select_cv2_interpolation(scale_factor_array)

    # a list of (original type, converted type) tuple
    # all `converted type` are np.dtypes that cv2.resize
    # can work on out-of-the-box (anything else will cause
    # error). The `converted type` has been selected so that
    # they can maintain the numeric precision of the `original type`.
    dtype_mapping = [
        (np.bool_, np.uint8),
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
        msg = f"Does not support resizing for array of dtype: {original_dtype}"
        raise ValueError(
            msg,
        )

    converted_dtype = dtype_mapping[source_dtypes.index(original_dtype)][1]
    img = img.astype(converted_dtype)

    cv2_interpolation = parse_cv2_interpolaton(interpolation)

    # Resize the image
    # Handle case for 1x1 images which cv2 v4.5.4 no longer handles
    if img.shape[0] == img.shape[1] == 1:
        return img.repeat(output_size_array[1], 0).repeat(output_size_array[0], 1)

    if len(img.shape) == 3 and img.shape[-1] > 4:  # noqa: PLR2004
        img_channels = [
            cv2.resize(
                src=img[..., ch],
                dsize=output_size_array,
                interpolation=cv2_interpolation,
            )[
                ...,
                None,
            ]
            for ch in range(img.shape[-1])
        ]
        return np.concatenate(img_channels, axis=-1)

    return cv2.resize(src=img, dsize=output_size_array, interpolation=cv2_interpolation)


def rgb2od(img: np.ndarray) -> np.ndarray:
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


def od2rgb(od: np.ndarray) -> np.ndarray:
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


def bounds2locsize(
    bounds: tuple[int, int, int, int] | np.ndarray,
    origin: str = "upper",
) -> tuple[np.ndarray, np.ndarray]:
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
        np.ndarray:
            A set of two arrays containing integer for location and size:

            - location array
                - x location
                - y location

            - size array
                - width
                - height

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
    msg = "Invalid origin. Only 'upper' or 'lower' are valid."
    raise ValueError(msg)


def locsize2bounds(
    location: tuple[int, int],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
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
    bounds: tuple[int, int, int, int] | np.ndarray,
    stride: int | tuple[int, int, tuple[int, int]] = 1,
) -> tuple[slice, ...]:
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
        msg = "Invalid stride shape."
        raise ValueError(msg)
    stride_array = np.tile(stride, 2)
    if np.size(stride) == 1:
        stride_array = np.tile(stride, 4)

    start, stop = np.reshape(bounds, (2, -1)).astype(int)
    slice_array = np.stack([start[::-1], stop[::-1]], axis=1)

    slices = []
    for x, s in zip(slice_array, stride_array):
        slices.append(slice(x[0], x[1], s))

    return tuple(slices)


def pad_bounds(
    bounds: tuple[int, int, int, int],
    padding: int | tuple[int, int] | tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
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
        msg = "Bounds must have an even number of elements."
        raise ValueError(msg)
    ndims = np.size(bounds) // 2

    if np.size(padding) not in [1, 2, np.size(bounds)]:
        msg = "Invalid number of padding elements."
        raise ValueError(msg)

    if np.size(padding) == 1 or np.size(padding) == np.size(bounds):
        pass
    elif np.size(padding) == ndims:  # pragma: no cover
        padding = np.tile(padding, 2)

    signs = np.repeat([-1, 1], ndims)
    return np.add(bounds, padding * signs)
