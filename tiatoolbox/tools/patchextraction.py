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

"""This file defines patch extraction methods for deep learning models."""
from abc import ABC
from tiatoolbox.utils.exceptions import MethodNotSupported
from tiatoolbox.utils.misc import imread

import math
import numpy as np


class PatchExtractor(ABC):
    """
    Class for extracting and merging patches in standard and whole-slide images.

    Args:
        img_patch_h(int): input image patch height.
        img_patch_w(int): input image patch width.
        pad_y(int): symmetric padding y-axis.
        pad_x(int): symmetric padding x-axis.

    """

    def __init__(self, img_patch_h, img_patch_w, pad_y=0, pad_x=0):
        self.img_patch_h = img_patch_h
        self.img_patch_w = img_patch_w
        self.pad_y = pad_y
        self.pad_x = pad_x

    @staticmethod
    def __get_last_steps(image_dim, label_patch_dim, stride):
        """Get the last location for patch extraction in a specific direction.

        Args:
            image_dim: 1D size of image
            label_patch_dim: 1D size of patches
            stride: 1D size of stride for patch extraction

        Returns:
            last_step: the final location for patch extraction
        """
        nr_step = math.ceil((image_dim - label_patch_dim) / stride)
        last_step = (nr_step + 1) * stride
        return int(last_step)

    def extract_patches(
        self,
        input_img,
        labels=None,
        save_output=False,
        save_path=None,
        save_name=None,
    ):
        """Extract patches from an image using locations provided by labels data.

        Args:
            input_img (str, ndarray): input image.
            labels (str, ndarray):
            save_output (bool): whether to save extracted patches
            save_path (str, pathlib.Path): path to save patches (only
              if save_output = True).
            save_name (str): filename for saving patches (only if save_output = True)

        Returns:
            img_patches (ndarray): extracted image patches of size NxHxWxD.
        """

        raise NotImplementedError

    def merge_patches(self, patches):
        """Merge the patch-level results to get the overall image-level prediction.

        Args:
            patches: patch-level predictions

        Returns:
            image: merged prediction
        """

        raise NotImplementedError


class FixedWindowPatchExtractor(PatchExtractor):
    """Extract and merge patches using fixed sized windows for images and labels.

    Args:
        img_patch_h: input image patch height
        img_patch_w: input image patch width
        stride_h: stride in horizontal direction for patch extraction
        stride_w: stride in vertical direction for patch extraction
    """

    def __init__(
        self, img_patch_h, img_patch_w, pad_y=0, pad_x=0, stride_h=1, stride_w=1
    ):
        super().__init__(
            img_patch_h=img_patch_h, img_patch_w=img_patch_w, pad_y=pad_y, pad_x=pad_x
        )
        self.stride_h = stride_h
        self.stride_w = stride_w

        raise NotImplementedError

    def extract_patches(
        self,
        input_img,
        labels=None,
        save_output=False,
        save_path=None,
        save_name=None,
    ):
        raise NotImplementedError

    def merge_patches(self, patches):
        raise NotImplementedError


class VariableWindowPatchExtractor(PatchExtractor):
    """Extract and merge patches using variable sized windows for images and labels.

    Args:
        img_patch_h: input image patch height
        img_patch_w: input image patch width
        stride_h: stride in horizontal direction for patch extraction
        stride_w: stride in vertical direction for patch extraction
        label_patch_h: network output label height
        label_patch_w: network output label width
    """

    def __init__(
        self,
        img_patch_h,
        img_patch_w,
        pad_y=0,
        pad_x=0,
        stride_h=1,
        stride_w=1,
        label_patch_h=None,
        label_patch_w=None,
    ):
        super().__init__(
            img_patch_h=img_patch_h, img_patch_w=img_patch_w, pad_y=pad_y, pad_x=pad_x
        )
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.label_patch_h = label_patch_h
        self.label_patch_w = label_patch_w

        raise NotImplementedError

    def extract_patches(
        self,
        input_img,
        labels=None,
        save_output=False,
        save_path=None,
        save_name=None,
    ):
        raise NotImplementedError

    def merge_patches(self, patches):
        raise NotImplementedError


class PointsPatchExtractor(PatchExtractor):
    """Extracting patches with specified points as a centre.

    Args:
        img_patch_h: input image patch height
        img_patch_w: input image patch width
    """

    def __init__(
        self,
        img_patch_h,
        img_patch_w,
        pad_y=0,
        pad_x=0,
        num_examples_per_patch=9,  # Square Root of Num of Examples must be odd
    ):
        super().__init__(
            img_patch_h=img_patch_h, img_patch_w=img_patch_w, pad_y=pad_y, pad_x=pad_x
        )

        self.num_examples_per_patch = num_examples_per_patch

    def extract_patches(
        self, input_img, labels=None, save_output=False, save_path=None, save_name=None
    ):
        if isinstance(labels, np.ndarray):
            labels = labels
        else:
            raise Exception("Please input correct csv, json path or csv data")

        if input_img == str:
            image = imread(input_img)
        elif isinstance(input_img, np.ndarray):
            image = input_img
        else:
            raise Exception("Please input correct image path or numpy array")

        patch_h = self.img_patch_h
        patch_w = self.img_patch_w

        image = np.lib.pad(
            image,
            ((self.pad_y, self.pad_y), (self.pad_x, self.pad_x), (0, 0)),
            "symmetric",
        )

        num_patches_img = len(labels) * self.num_examples_per_patch
        img_patches = np.zeros(
            (num_patches_img, patch_h, patch_w, image.shape[2]), dtype=image.dtype
        )
        labels = [None] * num_patches_img
        class_id = [None] * num_patches_img

        cell_tot = 1
        iter_tot = 0
        for row in labels:
            patch_label = row[0]
            cell_location = np.array([row[2], row[1]], dtype=np.int)
            cell_location[0] = (
                cell_location[0] + self.pad_y - 1
            )  # Python index starts from 0
            cell_location[1] = (
                cell_location[1] + self.pad_x - 1
            )  # Python index starts from 0
            if self.num_examples_per_patch > 1:
                root_num_examples = np.sqrt(self.num_examples_per_patch)
                start_location = -int(root_num_examples / 2)
                end_location = int(root_num_examples + start_location)
            else:
                start_location = 0
                end_location = 1

            for h in range(start_location, end_location):
                for w in range(start_location, end_location):
                    start_h = cell_location[0] - h - int((patch_h - 1) / 2)
                    start_w = cell_location[1] - w - int((patch_w - 1) / 2)
                    end_h = start_h + patch_h
                    end_w = start_w + patch_w
                    labels[iter_tot] = patch_label
                    class_id[iter_tot] = cell_tot
                    img_patches[iter_tot, :, :, :] = image[start_h:end_h, start_w:end_w]
                    iter_tot += 1

            cell_tot += 1
        return img_patches, labels, class_id

    def merge_patches(self, patches=None):
        raise MethodNotSupported(
            message="Merge patches not supported for " "PointsPatchExtractor"
        )


def get_patch_extractor(
    method_name, img_patch_h=224, img_patch_w=224, input_points=None, pad_y=0, pad_x=0
):
    """Return a patch extractor object as requested.
    Args:
        method_name (str): name of patch extraction method, must be one of
                            "point", "fixedwindow", "variablwindow".
        img_patch_h(int): desired image patch height, default=224.
        img_patch_w(int): desired image patch width, default=224.
        input_points(pd.dataframe, pathlib.Path): pandas dataframe with x, y, l,
          columns or path to csv/json containing input points and labels for patch
          extraction using points defined by x, y and l(labels).
        pad_y(int): symmetric padding y-axis, default=0.
        pad_x(int): symmetric padding x-axis, default=0.
    Return:
        PatchExtractor : an object with base 'PatchExtractor' as base class.
    Examples:
        >>> from tiatoolbox.tools.patchextraction import get_patch_extractor
        >>> # PointsPatchExtractor with default values
        >>> patch_extract = get_patch_extractor('point')

    """
    if method_name.lower() == "point":
        patch_extractor = PointsPatchExtractor(
            img_patch_h=img_patch_h,
            img_patch_w=img_patch_w,
            pad_y=pad_y,
            pad_x=pad_x,
            input_points=input_points,
        )
    elif method_name.lower() == "fixedwindow":
        patch_extractor = FixedWindowPatchExtractor(
            img_patch_h=img_patch_h, img_patch_w=img_patch_w, pad_y=pad_y, pad_x=pad_x
        )
    elif method_name.lower() == "variablewindow":
        patch_extractor = VariableWindowPatchExtractor(
            img_patch_h=img_patch_h, img_patch_w=img_patch_w, pad_y=pad_y, pad_x=pad_x
        )
    else:
        raise MethodNotSupported

    return patch_extractor
