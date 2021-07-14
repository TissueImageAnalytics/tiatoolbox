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
# The Original Code is Copyright (C) 2021, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****


import os
import pathlib

import numpy as np
import torch

from tiatoolbox.utils.misc import imread


class ABCPatchDataset(torch.utils.data.Dataset):
    """Defines abstract base class for patch dataset.

    Attributes:
        return_labels (bool, False): `__getitem__` will return both the img and
        its label. If `label_list` is `None`, `None` is returned

        preproc_func: Preprocessing function used to transform the input data. If
        supplied, then torch.Compose will be used on the input preproc_list.
        preproc_list is a list of torchvision transforms for preprocessing the image.
        The transforms will be applied in the order that they are given in the list.
        https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(self, preproc_func=None):
        super().__init__()
        self.preproc = preproc_func
        self.data_is_npy_alike = False
        self.input_list = []
        self.label_list = []

    def _check_input_integrity(self, mode):
        """Perform check to make sure variables received during init are valid.

        These check include:
            - Input is of a singular data type, such as a list of paths.
            - If it is list of images, all images are of the same height and width
        """
        if mode == "patch":
            self.data_is_npy_alike = False

            # If input is a list - can contain a list of images or a list of image paths
            if isinstance(self.input_list, list):
                is_all_path_list = all(
                    isinstance(v, (pathlib.Path, str)) for v in self.input_list
                )
                is_all_npy_list = all(
                    isinstance(v, np.ndarray) for v in self.input_list
                )
                if not (is_all_path_list or is_all_npy_list):
                    raise ValueError(
                        "Input must be either a list/array of images "
                        "or a list of valid image paths."
                    )

                shape_list = []
                # When a list of paths is provided
                if is_all_path_list:
                    if any(not os.path.exists(v) for v in self.input_list):
                        # at least one of the paths are invalid
                        raise ValueError(
                            "Input must be either a list/array of images "
                            "or a list of valid image paths."
                        )
                    # Preload test for sanity check
                    shape_list = [self.load_img(v).shape for v in self.input_list]
                    self.data_is_npy_alike = False
                else:
                    shape_list = [v.shape for v in self.input_list]
                    self.data_is_npy_alike = True

                if any(len(v) != 3 for v in shape_list):
                    raise ValueError("Each sample must be an array of the form HWC.")

                max_shape = np.max(shape_list, axis=0)
                if (shape_list - max_shape[None]).sum() != 0:
                    raise ValueError("Images must have the same dimensions.")

            # If input is a numpy array
            elif isinstance(self.input_list, np.ndarray):
                # Check that input array is numerical
                if not np.issubdtype(self.input_list.dtype, np.number):
                    # ndarray of mixed data types
                    raise ValueError("Provided input array is non-numerical.")
                # N H W C | N C H W
                if len(self.input_list.shape) != 4:
                    raise ValueError(
                        "Input must be an array of images of the form NHWC. This can "
                        "be achieved by converting a list of images to a numpy array. "
                        " eg., np.array([img1, img2])."
                    )
                self.data_is_npy_alike = True

            else:
                raise ValueError(
                    "Input must be either a list/array of images "
                    "or a list of valid paths to image."
                )

        else:
            if not isinstance(self.input_list, (list, np.ndarray)):
                raise ValueError("input_list should be a list of patch coordinates")

    @staticmethod
    def load_img(path):
        """Load an image from a provided path.

        Args:
            path (str): Path to an image file.

        """
        path = pathlib.Path(path)
        if path.suffix in (".npy", ".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            patch = imread(path, as_uint8=False)
        else:
            raise ValueError(f"Can not load data of `{path.suffix}`")
        return patch

    @property
    def preproc(self):
        """Get preprocessing function."""
        return self.preproc_func

    @preproc.setter
    def preproc(self, func):
        """Setter for preprocessing function.

        Set the `preproc_func` to this `func` if it is not None.
        Else the `preproc_func` is reset to return source image.

        `func` must behave in the following manner:

        >>> transformed_img = func(img)

        """
        self.preproc_func = func if func is not None else lambda x: x

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        raise NotImplementedError
