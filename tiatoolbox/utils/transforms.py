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

"""Define Image transforms."""
import numpy as np
from PIL import Image
import cv2

from tiatoolbox import utils


def background_composite(image, fill=255, alpha=False):
    """Image composite with specified background.

    Args:
        image (ndarray or PIL.Image): input image
        fill (int): fill value for the background, default=255
        alpha (bool): True if alpha channel is required

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
        composite = np.asarray(composite.convert("RGB"))
    else:
        composite = np.asarray(composite)

    return composite


def imresize(img, scale_factor=None, output_size=None, interpolation="optimise"):
    """Resize input image.

    Args:
        img (:class:`numpy.ndarray`): input image
        scale_factor (tuple(float)): scaling factor to resize the input image
        output_size (tuple(int)): output image size, (width, height)
        interpolation (str or int): interpolation method used to interpolate the image
         using `opencv interpolation flags
         <https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html>`
         __ default='optimise', uses cv2.INTER_AREA for scale_factor
         <1.0 otherwise uses cv2.INTER_CUBIC

    Returns:
        :class:`numpy.ndarray`: resized image

    Examples:
        >>> from tiatoolbox.wsicore import wsireader
        >>> from tiatoolbox.utils import transforms
        >>> wsi = wsireader.WSIReader(input_path="./CMU-1.ndpi")
        >>> slide_thumbnail = wsi.slide_thumbnail()
        >>> # Resize the image to half size using scale_factor 0.5
        >>> transforms.imresize(slide_thumbnail, scale_factor=0.5)

    """
    # Handle None arguments
    if output_size is None:
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        output_size = (width, height)

    if scale_factor is None:
        scale_factor = img.shape[:2][::-1] / np.array(output_size)

    # Return original if scale factor is 1
    if np.all(scale_factor == 1.0):
        return img

    # Get appropriate cv2 interpolation enum
    if interpolation == "optimise":
        if np.any(scale_factor > 1.0):
            interpolation = "cubic"
        else:
            interpolation = "area"
    interpolation = utils.misc.parse_cv2_interpolaton(interpolation)

    # Resize the image
    return cv2.resize(img, tuple(output_size), interpolation=interpolation)


def convert_RGB2OD(img):
    """Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).

    Args:
        img (:class:`numpy.ndarray` of type :class:`numpy.uint8`): Image RGB

    Returns:
        :class:`numpy.ndarray`: Optical denisty RGB image.

    Examples:
        >>> from tiatoolbox.utils import transforms
        >>> # rgb_img: RGB image
        >>> od_img = transforms.convert_RGB2OD(rgb_img)

    """
    mask = img == 0
    img[mask] = 1
    return np.maximum(-1 * np.log(img / 255), 1e-6)


def convert_OD2RGB(OD):
    """Convert from optical density (OD_RGB) to RGB.
    RGB = 255 * exp(-1*OD_RGB)

    Args:
        OD (:class:`numpy.ndarray`): Optical denisty RGB image

    Returns:
        numpy.ndarray: Image RGB

    Examples:
        >>> from tiatoolbox.utils import transforms
        >>> # od_img: optical density image
        >>> rgb_img = transforms.convert_OD2RGB(od_img)

    """
    OD = np.maximum(OD, 1e-6)
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def bounds2locsize(bounds, origin="upper"):
    """Calculate the size of a tuple of bounds.

    Bounds are expected to be in the (left, top, right, bottom) /
    (start_x, start_y, end_x, end_y) format.

    Args:
        bounds (tuple(int)): A 4-tuple or length 4 array of bounds
            values in (left, top, right, bottom) format.
        origin (str): Upper (Top-left) or lower (bottom-left) origin.
            Defaults to upper.

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
        location (tuple(int)): A 2-tuple or length 2 array of x,y
         coordinates.
        size (tuple(int)): A 2-tuple or length 2 array of width and
         height.

    Returns:
        tuple: A tuple of bounds:
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
