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
from tiatoolbox.dataloader.wsireader import OpenSlideWSIReader, OmnyxJP2WSIReader
from tiatoolbox.utils.exceptions import MethodNotSupported
from tiatoolbox.utils.misc import imread, split_path_name_ext
from tiatoolbox.utils.exceptions import FileNotSupported

# import math
import numpy as np
import pathlib


class PatchExtractor(ABC):
    """
    Class for extracting and merging patches in standard and whole-slide images.

    Args:
        input_image(str, pathlib.Path, ndarray): input image for patch extraction.
        img_patch_h(int): input image patch height.
        img_patch_w(int): input image patch width.
        pad_y(int): symmetric padding y-axis.
        pad_x(int): symmetric padding x-axis.

    Attributes:
        input_image(ndarray, WSIReader): input image for patch extraction.
          input_image type is ndarray for an image tile whereas :obj:`WSIReader`
          for an WSI.
        img_patch_h(int): input image patch height.
        img_patch_w(int): input image patch width.
        pad_y(int): symmetric padding y-axis.
        pad_x(int): symmetric padding x-axis.
        n(int): current state of the iterator.

    """

    def __init__(self, input_image, img_patch_h, img_patch_w, pad_y=0, pad_x=0):
        self.img_patch_h = img_patch_h
        self.img_patch_w = img_patch_w
        self.pad_y = pad_y
        self.pad_x = pad_x
        self.n = 0
        self.input_image = input_image_for_patch_extraction(input_image)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        raise NotImplementedError

    # @staticmethod
    # def __get_last_steps(image_dim, label_patch_dim, stride):
    #     """Get the last location for patch extraction in a specific direction.
    #
    #     Args:
    #         image_dim: 1D size of image
    #         label_patch_dim: 1D size of patches
    #         stride: 1D size of stride for patch extraction
    #
    #     Returns:
    #         last_step: the final location for patch extraction
    #     """
    #     nr_step = math.ceil((image_dim - label_patch_dim) / stride)
    #     last_step = (nr_step + 1) * stride
    #     return int(last_step)

    # def extract_patches(self):
    #     """Extract patches from an image using locations provided by labels data.
    #
    #     Args:
    #         labels (str, ndarray):
    #
    #     Returns:
    #         img_patches (ndarray): extracted image patches of size NxHxWxD.
    #     """
    #     raise NotImplementedError

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
        stride_h(int): stride in horizontal direction for patch extraction.
        stride_w(int): stride in vertical direction for patch extraction.
    """

    def __init__(
        self,
        input_image,
        img_patch_h,
        img_patch_w,
        pad_y=0,
        pad_x=0,
        stride_h=1,
        stride_w=1,
    ):
        super().__init__(
            input_image=input_image,
            img_patch_h=img_patch_h,
            img_patch_w=img_patch_w,
            pad_y=pad_y,
            pad_x=pad_x,
        )
        self.stride_h = stride_h
        self.stride_w = stride_w

    # def extract_patches(self, input_img, labels=None):
    #     raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    def merge_patches(self, patches):
        raise NotImplementedError


class VariableWindowPatchExtractor(PatchExtractor):
    """Extract and merge patches using variable sized windows for images and labels.

    Args:
        stride_h: stride in horizontal direction for patch extraction
        stride_w: stride in vertical direction for patch extraction
        label_patch_h: network output label height
        label_patch_w: network output label width
    """

    def __init__(
        self,
        input_image,
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
            input_image=input_image,
            img_patch_h=img_patch_h,
            img_patch_w=img_patch_w,
            pad_y=pad_y,
            pad_x=pad_x,
        )
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.label_patch_h = label_patch_h
        self.label_patch_w = label_patch_w

    # def extract_patches(self, input_img, labels=None):
    #     raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    def merge_patches(self, patches):
        raise NotImplementedError


class PointsPatchExtractor(PatchExtractor):
    """Extracting patches with specified points as a centre.

    Args:
        labels(ndarray, pd.DataFrame, str, pathlib.Path): contains location and/or
         type of patch. Input can be path to a csv or json file.
        num_examples_per_patch(int): Number of examples per patch for ensemble
         classification, default=9 (centre of patch and all the eight neighbours as
         centre).

    Attributes:
        labels(pd.DataFrame): A table containing location and/or type of patch.
        num_examples_per_patch(int): Number of examples per patch for ensemble
         classification.

    """

    def __init__(
        self,
        input_image,
        labels,
        img_patch_h=224,
        img_patch_w=224,
        pad_y=0,
        pad_x=0,
        num_examples_per_patch=9,
    ):
        super().__init__(
            input_image=input_image,
            img_patch_h=img_patch_h,
            img_patch_w=img_patch_w,
            pad_y=pad_y,
            pad_x=pad_x,
        )

        self.num_examples_per_patch = num_examples_per_patch
        self.labels = labels

    def __next__(self):
        raise NotImplementedError

    # def extract_patches(self, input_img, labels=None):
    #     if not isinstance(labels, np.ndarray):
    #         raise Exception("Please input correct csv, json path or csv data")
    #
    #
    #     patch_h = self.img_patch_h
    #     patch_w = self.img_patch_w
    #
    #     image = np.lib.pad(
    #         image,
    #         ((self.pad_y, self.pad_y), (self.pad_x, self.pad_x), (0, 0)),
    #         "symmetric",
    #     )
    #
    #     num_patches_img = len(labels) * self.num_examples_per_patch
    #     img_patches = np.zeros(
    #         (num_patches_img, patch_h, patch_w, image.shape[2]), dtype=image.dtype
    #     )
    #     labels = [None] * num_patches_img
    #     class_id = [None] * num_patches_img
    #
    #     cell_tot = 1
    #     iter_tot = 0
    #     for row in labels:
    #         patch_label = row[0]
    #         cell_location = np.array([row[2], row[1]], dtype=np.int)
    #         cell_location[0] = (
    #             cell_location[0] + self.pad_y - 1
    #         )  # Python index starts from 0
    #         cell_location[1] = (
    #             cell_location[1] + self.pad_x - 1
    #         )  # Python index starts from 0
    #         if self.num_examples_per_patch > 1:
    #             root_num_examples = np.sqrt(self.num_examples_per_patch)
    #             start_location = -int(root_num_examples / 2)
    #             end_location = int(root_num_examples + start_location)
    #         else:
    #             start_location = 0
    #             end_location = 1
    #
    #         for h in range(start_location, end_location):
    #             for w in range(start_location, end_location):
    #                 start_h = cell_location[0] - h - int((patch_h - 1) / 2)
    #                 start_w = cell_location[1] - w - int((patch_w - 1) / 2)
    #                 end_h = start_h + patch_h
    #                 end_w = start_w + patch_w
    #                 labels[iter_tot] = patch_label
    #                 class_id[iter_tot] = cell_tot
    #              img_patches[iter_tot, :, :, :] = image[start_h:end_h, start_w:end_w]
    #                 iter_tot += 1
    #
    #         cell_tot += 1
    #     return img_patches, labels, class_id

    def merge_patches(self, patches=None):
        """Merge patch is not supported by :obj:`PointsPatchExtractor`.
        Calling this function for :obj:`PointsPatchExtractor` will raise an error. This
        overrides the merge_patches function in the base class :obj:`PatchExtractor`

        """
        raise MethodNotSupported(
            message="Merge patches not supported for PointsPatchExtractor"
        )


def input_image_for_patch_extraction(input_image):
    """Sets the correct value for PatchExtraction input_image attribute."""
    if not isinstance(input_image, np.ndarray):

        if isinstance(input_image, pathlib.Path):
            input_image = str(input_image)

        if isinstance(input_image, str):
            _, _, suffix = split_path_name_ext(input_image)

            if suffix in (".jpg", ".png"):
                input_image = imread(input_image)

            elif suffix in (".svs", ".ndpi", ".mrxs"):
                input_image = OpenSlideWSIReader(input_image)

            elif suffix == ".jp2":
                input_image = OmnyxJP2WSIReader(input_image)

            else:
                raise FileNotSupported("Filetype not supported.")
        else:
            raise FileNotSupported("Please input correct image path or numpy array")

    return input_image


def get_patch_extractor(method_name, **kwargs):
    """Return a patch extractor object as requested.

    Args:
        method_name (str): name of patch extraction method, must be one of "point",
          "fixedwindow", "variablewindow".
        **kwargs: Keyword arguments passed to :obj:`PatchExtractor`.

    Return:
        PatchExtractor: an object with base :obj:`PatchExtractor` as base class.

    Examples:
        >>> from tiatoolbox.tools.patchextraction import get_patch_extractor
        >>> # PointsPatchExtractor with default values
        >>> patch_extract = get_patch_extractor(
        ...  'point', img_patch_h=200, img_patch_w=200)

    """
    if method_name.lower() == "point":
        patch_extractor = PointsPatchExtractor(**kwargs)
    elif method_name.lower() == "fixedwindow":
        patch_extractor = FixedWindowPatchExtractor(**kwargs)
    elif method_name.lower() == "variablewindow":
        patch_extractor = VariableWindowPatchExtractor(**kwargs)
    else:
        raise MethodNotSupported

    return patch_extractor
