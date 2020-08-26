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
# The Original Code is Copyright (C) 2006, Blender Foundation
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Define Image transforms"""
import numpy as np
from PIL import Image
import cv2


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

    composite = Image.fromarray(
        np.full(list(image.size[::-1]) + [4], fill, dtype=np.uint8)
    )
    composite.alpha_composite(image)
    composite = np.asarray(composite.convert("RGB"))
    return composite


def imresize(img, scale_factor, interpolation=cv2.INTER_CUBIC):
    """Resize input image

    Args:
        img (ndarray): input image
        scale_factor (float): scaling factor to resize the input image
        interpolation (int): interpolation, default=cv2.INTER_CUBIC

    Returns:
        ndarray: resized image

    Examples:
            >>> from tiatoolbox.dataloader import wsireader
            >>> from tiatoolbox.utils import transforms
            >>> wsi_obj = wsireader.WSIReader(input_dir="./",
            ...     file_name="CMU-1.ndpi")
            >>> slide_thumbnail = wsi_obj.slide_thumbnail()
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
