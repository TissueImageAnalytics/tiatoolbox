# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2020, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Miscellaneous utilities which operate on image data."""
import warnings

import numpy as np
import cv2
from PIL import Image

from tiatoolbox.utils.transforms import bounds2locsize
from tiatoolbox.utils.misc import conv_out_size


def safe_padded_read(
    img, bounds, stride=1, padding=0, pad_mode="constant", **pad_kwargs
):
    """Read a region of a numpy array with padding applied to edges.

    Safely 'read' regions, even outside of the image bounds. Accepts
    integer bounds only.

    Regions outside of the source image are padded using
    any of the pad modes available in :func:`numpy.pad`.

    .. figure:: images/out_of_bounds_read.png
            :width: 512
            :alt: Illustration for reading a region with negative
                coordinates using zero padding and reflection padding.

    Args:
        img (:class:`numpy.ndarray` or :class:`glymur.Jp2k`):
            Input image to read from.
        bounds (tuple(int)):
            Bounds of the region in (left, top,
            right, bottom) format.
        stride (int, tuple(int)):
            Stride when reading from img. Defaults to 1. A tuple is
            interpreted as stride in x and y (axis 1 and 0 respectively).
            Also applies to padding.
        padding (int, tuple(int)):
            Padding to apply to each bound. Default to 0.
        pad_mode (str):
            Method for padding when reading areas outside of
            the input image. Default is constant (0 padding). Possible
            values are: constant, reflect, wrap, symmetric. See
            :func:`numpy.pad` for more.
        **pad_kwargs (dict):
            Arbitrary keyword arguments passed through to the
            padding function :func:`numpy.pad`.

    Returns:
        np.ndarray: Padded image region.

    Raises:
        ValueError: Bounds must be integers.
        ValueError: Padding can't be negative.

    Examples:
        >>> bounds = (-5, -5, 5, 5)
        >>> safe_padded_read(img, bounds)

        >>> bounds = (-5, -5, 5, 5)
        >>> safe_padded_read(img, bounds, pad_mode="reflect")

        >>> bounds = (1, 1, 6, 6)
        >>> safe_padded_read(img, bounds, padding=2 pad_mode="reflect")
    """
    padding = np.array(padding)
    # Ensure the bounds are integers.
    if not issubclass(np.array(bounds).dtype.type, (int, np.integer)):
        raise ValueError("Bounds must be integers.")

    if np.any(padding < 0):
        raise ValueError("Padding cannot be negative.")

    # Allow padding to be a 2-tuple in addition to an int or 4-tuple
    if np.size(padding) not in [1, 2, 4]:
        raise ValueError("Padding must be of size 1, 2 or 4.")
    if np.size(padding) == 2:
        padding = np.tile(padding, 2)

    # Ensure stride is a 2-tuple
    if np.size(stride) not in [1, 2]:
        raise ValueError("Stride must be of size 1 or 2.")
    if np.size(stride) == 1:
        stride = np.tile(stride, 2)
    x_stride, y_stride = stride

    # Check if the padded coords outside of the image bounds
    # (over the width/height or under 0)
    padded_bounds = bounds + (padding * np.array([-1, -1, 1, 1]))
    img_size = np.array(img.shape[:2][::-1])
    hw_limits = np.tile(img_size, 2)  # height/width limits
    zeros = np.zeros(hw_limits.shape)
    over = padded_bounds >= hw_limits
    under = padded_bounds < zeros
    # If all coords are within the image then read normally
    if not any(over | under):
        l, t, r, b = padded_bounds
        return img[t:b:y_stride, l:r:x_stride, ...]
    # Else find the closest coordinates which are inside the image
    clamped_bounds = np.max([np.min([padded_bounds, hw_limits], axis=0), zeros], axis=0)
    clamped_bounds = np.round(clamped_bounds).astype(int)
    # Read the area within the image
    l, t, r, b = clamped_bounds
    region = img[t:b:y_stride, l:r:x_stride, ...]
    # Reduce bounds an img_size for the stride
    if not np.all(np.isin(stride, [None, 1])):
        # This if is not required but avoids unnecessary calculations
        bounds = conv_out_size(np.array(bounds), stride=np.tile(stride, 2))
        padded_bounds = bounds + (padding * np.array([-1, -1, 1, 1]))
        img_size = conv_out_size(img_size, stride=stride)
    # Find how much padding needs to be applied to fill the edge gaps
    # edge_padding = np.abs(padded_bounds - clamped_bounds)
    edge_padding = padded_bounds - np.array(
        [
            *np.min([[0, 0], padded_bounds[2:]], axis=0),
            *np.max([img_size, padded_bounds[:2] - img_size], axis=0),
        ]
    )
    edge_padding[:2] = np.min([edge_padding[:2], [0, 0]], axis=0)
    edge_padding[2:] = np.max([edge_padding[2:], [0, 0]], axis=0)
    edge_padding = np.abs(edge_padding)
    l, t, r, b = edge_padding
    pad_width = [(t, b), (l, r)]
    if len(region.shape) == 3:
        pad_width += [(0, 0)]
    # Pad the image region at the edges
    region = np.pad(region, pad_width, mode=pad_mode, **pad_kwargs)
    return region


def sub_pixel_read(
    image,
    bounds,
    output_size,
    padding=0,
    stride=1,
    interpolation="nearest",
    pad_for_interpolation=True,
    pad_at_baseline=False,
    read_func=None,
    pad_mode="constant",
    **read_kwargs,
):
    """Read and resize an image region with sub-pixel bounds.

    Allows for reading of image regions with sub-pixel coordinates, and
    out of bounds reads with various padding and interpolation modes.

    .. figure:: images/sub_pixel_reads.png
            :width: 512
            :alt: Illustration for reading a region with fractional
                coordinates (sub-pixel).

    Args:
        image (:class:`numpy.ndarray`):
            Image to read from.
        bounds (tuple(float)):
            Bounds of the image to read in
            (left, top, right, bottom) format.
        output_size (tuple(int)):
            The desired output size.
        padding (int, tuple(int)):
            Amount of padding to apply to the image region in pixels.
            Defaults to 0.
        stride (int, tuple(int)):
            Stride when reading from img. Defaults to 1. A tuple is
            interpreted as stride in x and y (axis 1 and 0 respectively).
        interpolation (str):
            Method of interpolation. Possible values are: nearest,
            linear, cubic, lanczos. Defaults to nearest.
        pad_at_baseline (bool):
            Apply padding in terms of baseline
            pixels. Defaults to False, meaning padding is added to the
            output image size in pixels.
        pad_for_interpolation (bool):
            Add padding before scaling in
            order to avoid border effects from interpolation. This
            padding is removed after scaling/resampling. Defaults to
            True.
        read_func (collections.abc.Callable):
            Custom read function. Defaults to
            :func:`safe_padded_read`. A function which recieves
            two positional args of the image object and a set of
            integer bounds in addition to padding key word arguments
            for reading a pixel-aligned bounding region. This function
            should return a numpy array with 2 or 3 dimensions. See
            examples for more.
        pad_mode (str):
            Method for padding when reading areas outside of
            the input image. Default is constant (0 padding). This is
            passed to `read_func` which defaults to
            :func:`safe_padded_read`. See :func:`safe_padded_read`
            for supported pad modes.
        **read_kwargs (dict):
            Arbitrary keyword arguments passed through to `read_func`.

    Return:
        np.ndimage: Output image region.

    Raises:
        ValueError: Invalid arguments.
        AssertionError: Internal errors, possibly due to invalid values.

    Examples:

        Simple read:

        >>> bounds = (0, 0, 10.5, 10.5)
        >>> sub_pixel_read(image, bounds)

        Read with padding applied to bounds before reading:

        >>> bounds = (0, 0, 10.5, 10.5)
        >>> region = sub_pixel_read(
        ...     image,
        ...     bounds,
        ...     padding=2,
        ...     pad_mode="reflect",
        ... )

        Read with padding applied after reading:

        >>> bounds = (0, 0, 10.5, 10.5)
        >>> region = sub_pixel_read(image, bounds)
        >>> reguin = np.pad(region, padding=2, mode="reflect")

        Read with no temporary padding to account for interpolation mode:

        >>> bounds = (0, 0, 10.5, 10.5)
        >>> region = sub_pixel_read(
        ...     image,
        ...     bounds,
        ...     interpolation="cubic",
        ...     pad_for_interpolation=False,
        ...     padding=2,
        ...     pad_mode="reflect",
        ... )

        Custom read function which generates a diagonal gradient:

        >>> bounds = (0, 0, 10.5, 10.5)
        >>> def gradient(_, b, **kw):
        ...     width, height = (b[2] - b[0], b[3] - b[1])
        ...     return np.mgrid[:height, :width].sum(0)
        >>> sub_pixel_read(bounds, read_func=gradient)

        Custom read function which gets pixel data from a custom object:

        >>> bounds = (0, 0, 10, 10)
        >>> def openslide_read(image, bounds, **kwargs):
        ...     # Note that bounds may contain negative integers
        ...     left, top, right, bottom = bounds
        ...     size = (right - left, bottom - top)
        ...     pil_img = image.read_region((left, top), level=0, size=size)
        ...     return np.array(pil_img.convert("RGB"))
        >>> sub_pixel_read(bounds, read_func=openslide_read)


    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    bounds = np.array(bounds)
    _, bounds_size = bounds2locsize(bounds)
    if np.size(stride) == 2:
        stride = np.tile(stride, 2)
    bounds_size = bounds_size / stride
    if 0 in bounds_size:
        raise AssertionError("Bounds must have non-zero size in each dimension")
    scale_factor = output_size / bounds_size

    # Set interpolation variables.
    inter_padding = 0
    inter_cv2 = cv2.INTER_CUBIC
    if interpolation == "nearest":
        inter_cv2 = cv2.INTER_NEAREST
    elif interpolation == "linear":
        inter_cv2 = cv2.INTER_LINEAR
    elif interpolation == "cubic":
        inter_cv2 = cv2.INTER_CUBIC
        inter_padding = 1
    elif interpolation == "lanczos":
        inter_cv2 = cv2.INTER_LANCZOS4
        inter_padding = 2
    else:
        raise ValueError("Invalid interpolation mode.")
    if pad_for_interpolation is False:
        inter_padding = 0
    # Make interpolation padding a length 4 array.
    inter_padding = np.repeat(inter_padding, 4)

    # Normalise desired padding to be a length 4 array.
    if np.size(padding) == 1:
        desired_padding = np.repeat(padding, 4)
    elif np.size(padding) == 2:
        desired_padding = np.tile(padding, 2)
    else:
        desired_padding = np.array(padding)

    # Divide padding by scale factor to get padding to add to bounds.
    if pad_at_baseline:
        bounds_padding = desired_padding
        output_padding = desired_padding * np.tile(scale_factor, 2)
    else:
        bounds_padding = desired_padding / np.tile(scale_factor, 2)
        output_padding = desired_padding

    padded_output_size = np.round(output_size + output_padding.reshape(2, 2).sum(0))

    # Find the pixel-aligned indexes to read the image at
    PADDING_TO_BOUNDS = np.array([-1, -1, 1, 1])
    padded_bounds = bounds + (bounds_padding * PADDING_TO_BOUNDS)
    pixel_aligned_bounds = padded_bounds.copy()
    pixel_aligned_bounds[:2] = np.floor(pixel_aligned_bounds[:2])
    pixel_aligned_bounds[2:] = np.ceil(pixel_aligned_bounds[2:])
    pixel_aligned_bounds = pixel_aligned_bounds.astype(int)
    # Add interpolation padding to integer bounds so that read_func
    # doesn't need to handle anything except the bounds.
    int_bounds = pixel_aligned_bounds + (inter_padding * PADDING_TO_BOUNDS)
    # Keep the difference between pixel-aligned and original coordinates
    residuals = padded_bounds - int_bounds

    # The left/start_x and top/start_y values should usaully be smaller
    # than the right/end_x and bottom/end_y values.
    if not np.all(int_bounds[:2] < int_bounds[2:]):
        warnings.warn("Read: Bounds have negative size.")

    # Ensure the pixel-aligned + interpolation padded bounds are integers.
    if int_bounds.dtype != int:
        raise AssertionError("Bounds must be integers.")

    # If no read function is given, use the default.
    if read_func is None:
        read_func = safe_padded_read

    # Perform the pixel-aligned read.
    region = read_func(
        image, int_bounds, pad_mode=pad_mode, stride=stride, **read_kwargs
    )

    if not np.all(np.array(region.shape[:2]) > 0):
        raise AssertionError("Region should not be empty.")

    # Find the size which the region should be scaled to.
    scaled_size = np.array(region.shape[:2][::-1]) * scale_factor
    scaled_size = tuple(scaled_size.astype(int))
    # Resize/scale the region
    scaled_region = cv2.resize(region, scaled_size, interpolation=inter_cv2)

    # Remove the interpolation and pixel alignment padding.
    scaled_residuals = residuals * np.tile(scale_factor, 2)
    resized_indexes = scaled_residuals.copy()

    # Complex rounding to make output size consistent with requested size.
    # Can swap for simple np.round if a 1-off error is acceptable.
    while not np.all(resized_indexes % 1.0 == 0):
        # Find the non-zero fracitonal part furthest from an integer value
        to_round = np.abs(np.abs(resized_indexes % 1.0) - 0.5)
        to_round[to_round == 0] = np.inf
        i = np.argmin(to_round)
        # Swap floor/ceil if the value negative
        sign = 1 if resized_indexes[i] >= 0 else -1
        up_down = [np.ceil, np.floor][::sign]
        # Check if removing the current amount would make the image too small
        _, dim = bounds2locsize(resized_indexes + np.array([0, 0, *scaled_size]))[i % 2]
        target_dim = padded_output_size[i % 2]
        # If so, round so that less is cropped
        if dim < target_dim:
            resized_indexes[i] = up_down[1](resized_indexes[i])
        # Else, remove it
        else:
            resized_indexes[i] = up_down[0](resized_indexes[i])

    resized_indexes = resized_indexes.astype(int)
    resized_indexes += np.array([0, 0, *scaled_size])
    l, t, r, b = resized_indexes
    result = scaled_region[t:b, l:r, ...]
    result_size = np.array(result.shape[:2][::-1])

    if not np.all(np.abs(result_size - padded_output_size) <= 1):
        raise AssertionError("Output size should not differ from requested size.")

    return result
