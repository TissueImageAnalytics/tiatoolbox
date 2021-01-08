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
from pathlib import Path
import numpy as np
import pandas as pd

from tiatoolbox.dataloader.wsireader import get_wsireader
from tiatoolbox.utils.exceptions import MethodNotSupported, FileNotSupported
from tiatoolbox.utils.misc import split_path_name_ext

# import math
# import pathlib


class PatchExtractor(ABC):
    """
    Class for extracting and merging patches in standard and whole-slide images.

    Args:
        input_img(str, pathlib.Path, ndarray): input image for patch extraction.
        patch_size(Tuple of int): patch size tuple (width, height).
        pad_size(Tuple of int): symmetric padding in (x, y) direction.

    Attributes:
        input_img(ndarray, WSIReader): input image for patch extraction.
          input_image type is ndarray for an image tile whereas :obj:`WSIReader`
          for an WSI.
        patch_size(Tuple of int): patch size tuple (width, height).
        pad_size(Tuple of int): symmetric padding in (x, y) direction.
        n(int): current state of the iterator.

    """

    def __init__(self, input_img, patch_size, pad_size=0):
        self.patch_size = patch_size
        self.pad_size = pad_size
        self.n = 0
        self.wsi = get_wsireader(input_img=input_img)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        raise NotImplementedError

    def __getitem__(self, item):
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
        stride(Tuple of int): stride in (x, y) direction for patch extraction.

    Attributes:
        stride(Tuple of int): stride in (x, y) direction for patch extraction.

    """

    def __init__(
        self,
        input_img,
        patch_size,
        pad_size=0,
        stride=1,
    ):
        super().__init__(
            input_img=input_img,
            patch_size=patch_size,
            pad_size=pad_size,
        )
        self.stride = stride

    # def extract_patches(self, input_img, labels=None):
    #     raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    def merge_patches(self, patches):
        raise NotImplementedError


class VariableWindowPatchExtractor(PatchExtractor):
    """Extract and merge patches using variable sized windows for images and labels.

    Args:
        stride(Tuple of int): stride in (x, y) direction for patch extraction.
        label_patch_size(Tuple of int): network output label (width, height).

    Attributes:
        stride(Tuple of int): stride in (x, y) direction for patch extraction.
        label_patch_size(Tuple of int): network output label (width, height).

    """

    def __init__(
        self,
        input_img,
        patch_size,
        pad_size=0,
        stride=1,
        label_patch_size=None,
    ):
        super().__init__(
            input_img=input_img,
            patch_size=patch_size,
            pad_size=pad_size,
        )
        self.stride_h = stride
        self.label_patch_size = label_patch_size

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
        input_img,
        labels,
        patch_size=224,
        pad_size=0,
        num_examples_per_patch=9,
    ):
        super().__init__(
            input_img=input_img,
            patch_size=patch_size,
            pad_size=pad_size,
        )

        self.num_examples_per_patch = num_examples_per_patch
        self.labels = read_point_annotations(input_table=labels)

    def __next__(self):
        n = self.n

        if n >= self.labels.shape[0]:
            raise StopIteration
        self.n = n + 1
        return self[n]

    def __getitem__(self, item):
        if type(item) is not int:
            raise TypeError("Index should be an integer.")

        if item >= self.labels.shape[0]:
            raise IndexError

        x, y, _ = self.labels.values[item, :]

        x = x - int((self.patch_size[1] - 1) / 2)
        y = y - int((self.patch_size[0] - 1) / 2)

        data = self.wsi.read_rect(location=(x, y), size=self.patch_size)

        return data

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


def read_point_annotations(input_table):
    """Read annotations as pandas DataFrame.

    Args:
        input_table (str or pathlib.Path or ndarray): path to csv, npy or json or
         an ndarray. first column in the table represents x position, second column
         represents y position. The third column represents the class. If the table has
         headers, the header should be x, y & class.

    Returns:
        pd.DataFrame: DataFrame with x, y location and class

    Examples:
        >>> from tiatoolbox.tools.patchextraction import read_point_annotations
        >>> labels = read_point_annotations('./annotations.csv')

    """
    if isinstance(input_table, (str, Path)):
        _, _, suffix = split_path_name_ext(input_table)

        if suffix == ".npy":
            out_table = np.load(input_table)
            out_table = pd.DataFrame(out_table, columns=["x", "y", "class"])

        elif suffix == ".csv":
            out_table = pd.read_csv(input_table)
            if "x" not in out_table.columns:
                out_table = pd.read_csv(
                    input_table, header=None, names=["x", "y", "class"]
                )

        elif suffix == ".json":
            out_table = pd.read_json(input_table)

        else:
            raise FileNotSupported("Filetype not supported.")

    elif isinstance(input_table, np.ndarray):
        out_table = pd.DataFrame(input_table, columns=["x", "y", "class"])

    elif isinstance(input_table, pd.DataFrame):
        out_table = input_table

    else:
        raise TypeError("Please input correct image path or an ndarray image.")

    return out_table


def get_patch_extractor(method_name, **kwargs):
    """Return a patch extractor object as requested.

    Args:
        method_name (str): name of patch extraction method, must be one of "point",
          "fixedwindow", "variablewindow".
        **kwargs: Keyword arguments passed to :obj:`PatchExtractor`.

    Returns:
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
