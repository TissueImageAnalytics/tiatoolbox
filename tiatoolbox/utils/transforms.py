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


def background_composite(image, fill=255, alpha=False):
    """Image composite with specified background.

    Args:
        image (ndarray, PIL.Image): input image
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


def imresize(img, scale_factor, interpolation=cv2.INTER_CUBIC):
    """Resize input image.

    Args:
        img (ndarray): input image
        scale_factor (float): scaling factor to resize the input image
        interpolation (int): interpolation, default=cv2.INTER_CUBIC

    Returns:
        ndarray: resized image

    Examples:
        >>> from tiatoolbox.dataloader import wsireader
        >>> from tiatoolbox.utils import transforms
        >>> wsi = wsireader.WSIReader(input_path="./CMU-1.ndpi")
        >>> slide_thumbnail = wsi.slide_thumbnail()
        >>> # Resize the image to half size using scale_factor 0.5
        >>> transforms.imresize(slide_thumbnail, scale_factor=0.5)

    """
    # Estimate new dimension
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    # Resize image
    resized_img = cv2.resize(img, dim, interpolation=interpolation)

    return resized_img


def convert_RGB2OD(img):
    """Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).

    Args:
        img (ndarray uint8): Image RGB

    Returns:
        ndarray: Optical denisty RGB image.

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
        OD (ndrray): Optical denisty RGB image

    Returns:
        ndarray uint8: Image RGB

    Examples:
        >>> from tiatoolbox.utils import transforms
        >>> # od_img: optical density image
        >>> rgb_img = transforms.convert_OD2RGB(od_img)

    """
    OD = np.maximum(OD, 1e-6)
    return (255 * np.exp(-1 * OD)).astype(np.uint8)
