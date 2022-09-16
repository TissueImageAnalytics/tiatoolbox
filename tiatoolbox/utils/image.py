"""Miscellaneous utilities which operate on image data."""
import warnings
from typing import Tuple, Union

import numpy as np
from PIL import Image

from tiatoolbox.utils.misc import conv_out_size
from tiatoolbox.utils.transforms import (
    bounds2locsize,
    bounds2slices,
    imresize,
    locsize2bounds,
    pad_bounds,
)

PADDING_TO_BOUNDS = np.array([-1, -1, 1, 1])
"""
Constant array which when multiplied with padding and added to bounds,
applies the padding to the bounds.
"""
# Make this immutable / non-writable
PADDING_TO_BOUNDS.flags.writeable = False


def normalize_padding_size(padding):
    """Normalizes padding to be length 4 (left, top, right, bottom).

    Given a scalar value, this is assumed to apply to all sides and
    therefore repeated for each output (left, right, top, bottom). A
    length 2 input is assumed to apply the same padding to the
    left/right and top/bottom.

    Args:
        padding (int or tuple(int)):
            Padding to normalize.

    Raises:
        ValueError:
            Invalid input size of padding (e.g. length 3).
        ValueError:
            Invalid input shape of padding (e.g. 3 dimensional).

    Returns:
        :class:`numpy.ndarray`:
            Numpy array of length 4 with elements containing padding for
            left, top, right, bottom.

    """
    padding_shape = np.shape(padding)
    if len(padding_shape) > 1:
        raise ValueError(
            "Invalid input padding shape. Must be scalar or 1 dimensional."
        )
    padding_size = np.size(padding)
    if padding_size == 3:
        raise ValueError("Padding has invalid size 3. Valid sizes are 1, 2, or 4.")

    if padding_size == 1:
        return np.repeat(padding, 4)
    if padding_size == 2:
        return np.tile(padding, 2)
    return np.array(padding)


def find_padding(read_location, read_size, image_size):
    """Find the correct padding to add when reading a region of an image.

    Args:
      read_location (tuple(int)):
        The location of the region to read.
      read_size (tuple(int)):
        The size of the location to read.
      image_size (tuple(int)):
        The size of the image to read from.

    Returns:
        tuple:
            Tuple of padding to apply in the format expect by `np.pad`.
            i.e. `((before_x, after_x), (before_y, after_y))`.

    Examples:
        >>> from tiatoolbox.utils.image import find_padding
        >>> location, size = (-2, -2), (10, 10)
        >>> # Find padding needed to make the output (10, 10)
        >>> # if the image is only (5 , 5) and read at
        >>> # location (-2, -2).
        >>> find_padding(location, size, image_size=(5, 5))

    """
    read_location = np.array(read_location)
    read_size = np.array(read_size)
    image_size = np.array(image_size)

    before_padding = np.maximum(-read_location, 0)
    region_end = read_location + read_size
    after_padding = np.maximum(region_end - np.max([image_size, read_location], 0), 0)
    return np.stack([before_padding[::-1], after_padding[::-1]], axis=1)


def find_overlap(read_location, read_size, image_size):
    """Find the part of a region which overlaps the image area.

    Args:
      read_location (tuple(int)):
        The location of the region to read.
      read_size (tuple(int)):
        The size of the location to read.
      image_size (tuple(int)):
        The size of the image to read from.

    Returns:
        tuple:
            Bounds of the overlapping region.

    Examples:
        >>> from tiatoolbox.utils.image import find_overlap
        >>> loc, size = (-5, -5), (10, 10)
        >>> find_overlap(loc, size, (5, 5))

    """
    read_location = np.array(read_location)
    read_size = np.array(read_size)
    image_size = np.array(image_size)

    start = np.maximum(read_location, 0)
    region_end = read_location + read_size
    stop = np.minimum(region_end, image_size)

    # Concatenate start and stop to make a bounds array (left, top, right, bottom)
    return np.concatenate([start, stop])


def make_bounds_size_positive(bounds):
    """Make bounds have positive size and get horizontal/vertical flip flags.

    Bounds with a negative size in either direction with have the
    coordinates swapped (e.g. left and right or top and bottom swapped)
    and a respective horizontal or vertical flip flag set in the output
    to reflect the swaps which occurred.

    Args:
        bounds (:class:`numpy.ndarray`):
            Length 4 array of bounds.

    Returns:
        tuple:
            Three tuple containing positive bounds and flips:
            - :class:`numpy.ndarray` - Positive bounds
            - :py:obj:`bool` - Horizontal flip
            - :py:obj:`bool` - Vertical flip

    Examples:
        >>> from tiatoolbox.utils.image import make_bounds_size_positive
        >>> bounds = (10, 10, 0, 0)
        >>> positive_bounds, flipud, fliplr = make_bounds_size_positive(bounds)

    """
    fliplr, flipud = False, False
    _, (width, height) = bounds2locsize(bounds)
    if width >= 0 and height >= 0:
        return bounds, fliplr, flipud
    l, t, r, b = bounds
    if width < 0:
        l, r = r, l
        fliplr = True
    if height < 0:
        t, b = b, t
        flipud = True
    bounds = np.array([l, t, r, b])
    return (bounds, fliplr, flipud)


def crop_and_pad_edges(
    bounds: Tuple[int, int, int, int],
    max_dimensions: Tuple[int, int],
    region: np.ndarray,
    pad_mode: str = "constant",
    pad_constant_values: Union[int, Tuple] = 0,
) -> np.ndarray:
    """Apply padding to areas of a region which are outside max dimensions.

    Applies padding to areas of the image region which have coordinates
    less than zero or above the width and height in `max_dimensions`.
    Note that bounds and max_dimensions must be given for the same image
    pyramid level (or more generally resolution e.g. if interpolated
    between levels or working in other units).

    Note: This function is planned to be deprecated in the future when a
    transition from OpenSlide to tifffile as a dependency is complete.
    It is currently used to remove padding from OpenSlide regions before
    applying custom padding via :func:`numpy.pad`. This allows the
    behaviour when reading OpenSlide images to be consistent with other
    formats.

    Args:
        bounds (tuple(int)):
            Bounds of the image region.
        max_dimensions (tuple(int)):
            The maximum valid x and y values of the bounds, i.e. the
            width and height of the slide.
        region (:class:`numpy.ndarray`):
            The image region to be cropped and padded.
        pad_mode (str):
            The pad mode to use, see :func:`numpy.pad` for valid pad
            modes. Defaults to 'constant'. If set to "none" or None no
            padding is applied.
        pad_constant_values (int or tuple(int)):
            Constant value(s) to use when padding. Only used with
            pad_mode constant.

    Returns:
        :class:`numpy.ndarray`:
            The cropped and padded image.

    Examples:
        >>> from tiatoolbox.utils.image import crop_and_pad_edges
        >>> import numpy as np
        >>> region = np.ones((10, 10, 3))
        >>> padded_region = crop_and_pad_edges(
        ...   bounds=(-1, -1, 5, 5),
        ...   max_dimensions=(10, 10),
        ...   region=image,
        ...   pad_mode="constant",
        ...   pad_constant_values=0,
        ... )

    """
    loc, size = bounds2locsize(bounds)

    if np.min(max_dimensions) < 0:
        raise ValueError("Max dimensions must be >= 0.")

    if np.min(size) <= 0:
        raise ValueError("Bounds must have size (width and height) > 0.")

    padding = find_padding(loc, size, max_dimensions)
    if len(region.shape) > 2:
        padding = np.concatenate([padding, [[0, 0]]])

    # If no padding is required then return the original image unmodified
    if np.all(np.array(padding) == 0):
        return region

    overlap = find_overlap(loc, size, max_dimensions)
    overlap = np.maximum(overlap - np.tile(loc, 2), 0)

    # Add extra padding dimension for colour channels
    if len(region.shape) > 2:
        padding = padding + ((0, 0),)

    # Crop the region
    slices = bounds2slices(overlap)
    slices += (...,)
    crop = region[slices]

    # Return if pad_mode is None
    if pad_mode in ["none", None]:
        return crop

    # Pad the region and return
    if pad_mode == "constant":
        return np.pad(crop, padding, mode=pad_mode, constant_values=pad_constant_values)
    return np.pad(crop, padding, mode=pad_mode)


def safe_padded_read(
    image,
    bounds,
    stride=1,
    padding=0,
    pad_mode="constant",
    pad_constant_values=0,
    pad_kwargs=None,
):
    """Read a region of a numpy array with padding applied to edges.

    Safely 'read' regions, even outside the image bounds. Accepts
    integer bounds only.

    Regions outside the source image are padded using any of the pad
    modes available in :func:`numpy.pad`.

    Note that padding of the output is not guaranteed to be
    integer/pixel aligned if using a stride != 1.

    .. figure:: ../images/out_of_bounds_read.png
            :width: 512
            :alt: Illustration for reading a region with negative
                coordinates using zero padding and reflection padding.

    Args:
        image (:class:`numpy.ndarray` or :class:`glymur.Jp2k`):
            Input image to read from.
        bounds (tuple(int)):
            Bounds of the region in (left, top, right, bottom) format.
        stride (int or tuple(int)):
            Stride when reading from img. Defaults to 1. A tuple is
            interpreted as stride in x and y (axis 1 and 0
            respectively). Also applies to padding.
        padding (int or tuple(int)):
            Padding to apply to each bound. Default to 0.
        pad_mode (str):
            Method for padding when reading areas outside the input
            image. Default is constant (0 padding). Possible values are:
            constant, reflect, wrap, symmetric. See :func:`numpy.pad`
            for more.
        pad_constant_values (int, tuple(int)): Constant values to use
            when padding with constant pad mode. Passed to the
            :func:`numpy.pad` `constant_values` argument. Default is 0.
        pad_kwargs (dict):
            Arbitrary keyword arguments passed through to the padding
            function :func:`numpy.pad`.

    Returns:
        :class:`numpy.ndarray`:
            Padded image region.

    Raises:
        ValueError:
            Bounds must be integers.
        ValueError:
            Padding can't be negative.

    Examples:
        >>> bounds = (-5, -5, 5, 5)
        >>> safe_padded_read(img, bounds)

        >>> bounds = (-5, -5, 5, 5)
        >>> safe_padded_read(img, bounds, pad_mode="reflect")

        >>> bounds = (1, 1, 6, 6)
        >>> safe_padded_read(img, bounds, padding=2, pad_mode="reflect")

    """
    if pad_kwargs is None:
        pad_kwargs = {}
    if pad_mode == "constant" and "constant_values" not in pad_kwargs:
        pad_kwargs["constant_values"] = pad_constant_values

    padding = np.array(padding)
    # Ensure the bounds are integers.
    if not issubclass(np.array(bounds).dtype.type, (int, np.integer)):
        raise ValueError("Bounds must be integers.")

    if np.any(padding < 0):
        raise ValueError("Padding cannot be negative.")

    # Allow padding to be a 2-tuple in addition to an int or 4-tuple
    padding = normalize_padding_size(padding)

    # Ensure stride is a 2-tuple
    if np.size(stride) not in [1, 2]:
        raise ValueError("Stride must be of size 1 or 2.")
    if np.size(stride) == 1:
        stride = np.tile(stride, 2)
    x_stride, y_stride = stride

    # Check if the padded coords are outside the image bounds
    # (over the width/height or under 0)
    padded_bounds = bounds + (padding * np.array([-1, -1, 1, 1]))
    img_size = np.array(image.shape[:2][::-1])
    hw_limits = np.tile(img_size, 2)  # height/width limits
    zeros = np.zeros(hw_limits.shape)
    # If all original bounds are within the bounds
    padded_over = padded_bounds >= hw_limits
    padded_under = padded_bounds < zeros
    # If all padded coords are within the image then read normally
    if not any(padded_over | padded_under):
        l, t, r, b = padded_bounds
        return image[t:b:y_stride, l:r:x_stride, ...]
    # Else find the closest coordinates which are inside the image
    clamped_bounds = np.max([np.min([padded_bounds, hw_limits], axis=0), zeros], axis=0)
    clamped_bounds = np.round(clamped_bounds).astype(int)
    # Read the area within the image
    l, t, r, b = clamped_bounds
    region = image[t:b:y_stride, l:r:x_stride, ...]
    # Reduce bounds an img_size for the stride
    if not np.all(np.isin(stride, [None, 1])):
        # This if is not required but avoids unnecessary calculations
        bounds = conv_out_size(np.array(bounds), stride=np.tile(stride, 2))
        padded_bounds = bounds + (padding * np.array([-1, -1, 1, 1]))
        img_size = conv_out_size(img_size, stride=stride)

    # Return without padding if pad_mode is none
    if pad_mode in ["none", None]:
        return region

    # Find how much padding needs to be applied to fill the edge gaps
    before_padding = np.min([[0, 0], padded_bounds[2:]], axis=0)
    after_padding = np.max([img_size, padded_bounds[:2] - img_size], axis=0)
    edge_padding = padded_bounds - np.concatenate([before_padding, after_padding])
    edge_padding[:2] = np.min([edge_padding[:2], [0, 0]], axis=0)
    edge_padding[2:] = np.max([edge_padding[2:], [0, 0]], axis=0)
    edge_padding = np.abs(edge_padding)
    l, t, r, b = edge_padding
    pad_width = [(t, b), (l, r)]
    if len(region.shape) == 3:
        pad_width += [(0, 0)]
    # Pad the image region at the edges
    return np.pad(
        region,
        pad_width,
        mode=pad_mode,
        **pad_kwargs,
    )


def sub_pixel_read(  # noqa: CCR001
    image,
    bounds,
    output_size,
    padding=0,
    stride=1,
    interpolation="nearest",
    pad_at_baseline=False,
    interpolation_padding=2,
    read_func=None,
    pad_mode="constant",
    pad_constant_values=0,
    read_kwargs=None,
    pad_kwargs=None,
):
    """Read and resize an image region with sub-pixel bounds.

    Allows for reading of image regions with sub-pixel coordinates, and
    out of bounds reads with various padding and interpolation modes.

    .. figure:: ../images/sub_pixel_reads.png
            :width: 512
            :alt: Illustration for reading a region with fractional
                coordinates (sub-pixel).

    Args:
        image (:class:`numpy.ndarray`):
            Image to read from.
        bounds (tuple(float)):
            Bounds of the image to read in (left, top, right, bottom)
            format.
        output_size (tuple(int)):
            The desired output size.
        padding (int or tuple(int)):
            Amount of padding to apply to the image region in pixels.
            Defaults to 0.
        stride (int or tuple(int)):
            Stride when reading from img. Defaults to 1. A tuple is
            interpreted as stride in x and y (axis 1 and 0
            respectively).
        interpolation (str):
            Method of interpolation. Possible values are: nearest,
            linear, cubic, lanczos, area. Defaults to nearest.
        pad_at_baseline (bool):
            Apply padding in terms of baseline pixels. Defaults to
            False, meaning padding is added to the output image size in
            pixels.
        interpolation_padding (int):
            Padding to temporarily apply before rescaling to avoid
            border effects. Defaults to 2.
        read_func (collections.abc.Callable):
            Custom read function. Defaults to :func:`safe_padded_read`.
            A function which recieves two positional args of the image
            object and a set of integer bounds in addition to padding
            key word arguments for reading a pixel-aligned bounding
            region. This function should return a numpy array with 2 or
            3 dimensions. See examples for more.
        pad_mode (str):
            Method for padding when reading areas are outside the input
            image. Default is constant (0 padding). This is passed to
            `read_func` which defaults to :func:`safe_padded_read`. See
            :func:`safe_padded_read` for supported pad modes. Setting to
            "none" or None will result in no padding being applied.
        pad_constant_values (int, tuple(int)): Constant values to use
            when padding with constant pad mode. Passed to the
            :func:`numpy.pad` `constant_values` argument. Default is 0.
        **read_kwargs (dict):
            Arbitrary keyword arguments passed through to `read_func`.
        **pad_kwargs (dict):
            Arbitrary keyword arguments passed through to the padding
            function :func:`numpy.pad`.

    Returns:
        :class:`numpy.ndimage`:
            Output image region.

    Raises:
        ValueError:
            Invalid arguments.
        AssertionError:
            Internal errors, possibly due to invalid values.

    Examples:
        >>> # Simple read
        >>> bounds = (0, 0, 10.5, 10.5)
        >>> sub_pixel_read(image, bounds)

        >>> # Read with padding applied to bounds before reading:
        >>> bounds = (0, 0, 10.5, 10.5)
        >>> region = sub_pixel_read(
        ...     image,
        ...     bounds,
        ...     padding=2,
        ...     pad_mode="reflect",
        ... )

        >>> # Read with padding applied after reading:
        >>> bounds = (0, 0, 10.5, 10.5)
        >>> region = sub_pixel_read(image, bounds)
        >>> region = np.pad(region, padding=2, mode="reflect")

        >>> # Custom read function which generates a diagonal gradient:
        >>> bounds = (0, 0, 10.5, 10.5)
        >>> def gradient(_, b, **kw):
        ...     width, height = (b[2] - b[0], b[3] - b[1])
        ...     return np.mgrid[:height, :width].sum(0)
        >>> sub_pixel_read(bounds, read_func=gradient)

        >>> # Custom read function which gets pixel data from a custom object:
        >>> bounds = (0, 0, 10, 10)
        >>> def openslide_read(image, bounds, **kwargs):
        ...     # Note that bounds may contain negative integers
        ...     left, top, right, bottom = bounds
        ...     size = (right - left, bottom - top)
        ...     pil_img = image.read_region((left, top), level=0, size=size)
        ...     return np.array(pil_img.convert("RGB"))
        >>> sub_pixel_read(bounds, read_func=openslide_read)

    """
    # Handle inputs
    if pad_kwargs is None:
        pad_kwargs = {}
    if read_kwargs is None:
        read_kwargs = {}
    if interpolation is None:
        interpolation = "none"

    if pad_mode == "constant" and "constant_values" not in pad_kwargs:
        pad_kwargs["constant_values"] = pad_constant_values

    if 0 in bounds2locsize(bounds)[1]:
        raise ValueError("Bounds must have non-zero size")

    # Normalize padding
    padding = normalize_padding_size(padding)

    # Check the bounds are valid or have a negative size
    # The left/start_x and top/start_y values should usually be smaller
    # than the right/end_x and bottom/end_y values.
    bounds, fliplr, flipud = make_bounds_size_positive(bounds)
    if fliplr or flipud:
        warnings.warn("Bounds have a negative size, output will be flipped.")

    if isinstance(image, Image.Image):
        image = np.array(image)

    # Normalize none pad_mode to None
    if pad_mode.lower() == "none":
        pad_mode = None

    # Initialise variables
    image_size = np.flip(image.shape[:2])
    scaling = np.array([1, 1])
    _, bounds_size = bounds2locsize(bounds)
    if output_size is not None and interpolation != "none":
        scaling = np.array(output_size) / bounds_size / stride
    read_bounds = bounds
    if pad_mode is None:
        output_size = np.round(
            bounds2locsize(find_overlap(*bounds2locsize(bounds), image_size))[1]
            * scaling
        ).astype(int)

    overlap_bounds = find_overlap(*bounds2locsize(bounds), image_size=image_size)
    if pad_mode is None:
        read_bounds = overlap_bounds
    pad_width = np.zeros((2, 2), int)

    baseline_padding = padding
    if not pad_at_baseline:
        baseline_padding = padding * np.tile(scaling, 2)

    # Check the padded bounds do not have zero size
    _, padded_bounds_size = bounds2locsize(pad_bounds(bounds, baseline_padding))
    if 0 in padded_bounds_size:
        raise ValueError("Bounds have zero size after padding.")

    read_bounds = pad_bounds(read_bounds, interpolation_padding + baseline_padding)
    # 0 Expand to integers and find residuals
    start, end = np.reshape(read_bounds, (2, -1))
    int_read_bounds = np.concatenate(
        [
            np.floor(start),
            np.ceil(end),
        ]
    )
    residuals = np.abs(int_read_bounds - read_bounds)
    read_bounds = int_read_bounds
    valid_int_bounds = find_overlap(
        *bounds2locsize(int_read_bounds), image_size
    ).astype(int)

    # 1 Read the region
    _, valid_int_size = bounds2locsize(valid_int_bounds)
    if read_func is None:
        region = image[bounds2slices(valid_int_bounds, stride=stride)]
    else:
        region = read_func(image, valid_int_bounds, stride, **read_kwargs)
        if region is None or 0 in region.shape:
            raise ValueError("Read region is empty or None.")
        region_size = region.shape[:2][::-1]
        if not np.array_equal(region_size, valid_int_size):
            raise ValueError("Read function returned a region of incorrect size.")

    # 1.5 Pad the region
    pad_width = find_padding(*bounds2locsize(read_bounds), image_size=image_size)
    if pad_mode is None:
        pad_width -= find_padding(*bounds2locsize(overlap_bounds), image_size)
    # Apply stride to padding
    pad_width = pad_width / stride
    # Add 0 padding to channels if required
    if len(image.shape) > 2:
        pad_width = np.concatenate([pad_width, [(0, 0)]])
    # 1.7 Do the padding
    if pad_mode == "constant":
        region = np.pad(
            region,
            pad_width.astype(int),
            mode=pad_mode or "constant",
            **pad_kwargs,
        )
    else:
        region = np.pad(region, pad_width.astype(int), mode=pad_mode or "constant")
    # 2 Re-scaling
    if output_size is not None and interpolation != "none":
        region = imresize(region, scale_factor=scaling, interpolation=interpolation)
    # 3 Trim interpolation padding
    region_size = np.flip(region.shape[:2])
    trimming = bounds2slices(
        np.round(
            pad_bounds(
                locsize2bounds((0, 0), region_size),
                (-(interpolation_padding + residuals) * np.tile(scaling, 2)),
            )
        )
    )
    region = region[trimming + (...,)]
    region_size = region.shape[:2][::-1]
    # 4 Ensure output is the correct size
    if output_size is not None and interpolation != "none":
        total_padding_per_axis = padding.reshape(2, 2).sum(axis=0)
        if pad_at_baseline:
            output_size = np.round(
                np.add(output_size, total_padding_per_axis * scaling)
            ).astype(int)
        else:
            output_size = np.add(output_size, total_padding_per_axis)
        if not np.array_equal(region_size, output_size):
            region = imresize(
                region, output_size=tuple(output_size), interpolation=interpolation
            )
    # 5 Apply flips to account for negative bounds
    if fliplr:
        region = np.flipud(region)
    if flipud:
        region = np.fliplr(region)
    return region  # noqa: R504
