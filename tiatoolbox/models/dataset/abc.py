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


from abc import ABC, abstractmethod
import os
import pathlib

import numpy as np
import torch

from tiatoolbox.utils.misc import imread


class PatchDatasetABC(ABC, torch.utils.data.Dataset):
    """Defines abstract base class for patch dataset.

    Attributes:
        return_labels (bool, False): `__getitem__` will return both the img and
          its label. If `labels` is `None`, `None` is returned
        preproc_func: Preprocessing function used to transform the input data. If
          supplied, then torch.Compose will be used on the input preprocs.
          preprocs is a list of torchvision transforms for preprocessing the
          image. The transforms will be applied in the order that they are given in
          the list. For more information, use the following link:
          https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(
        self,
    ):
        super().__init__()
        self._preproc = self.preproc
        self.data_is_npy_alike = False
        self.inputs = []
        self.labels = []

    def _check_input_integrity(self, mode):
        """Perform check to make sure variables received during init are valid.

        These checks include:
            - Input is of a singular data type, such as a list of paths.
            - If it is list of images, all images are of the same height and width

        """
        if mode == "patch":
            self.data_is_npy_alike = False

            # If input is a list - can contain a list of images or a list of image paths
            if isinstance(self.inputs, list):
                is_all_paths = all(
                    isinstance(v, (pathlib.Path, str)) for v in self.inputs
                )
                is_all_npys = all(isinstance(v, np.ndarray) for v in self.inputs)
                if not (is_all_paths or is_all_npys):
                    raise ValueError(
                        "Input must be either a list/array of images "
                        "or a list of valid image paths."
                    )

                shapes = []
                # When a list of paths is provided
                if is_all_paths:
                    if any(not os.path.exists(v) for v in self.inputs):
                        # at least one of the paths are invalid
                        raise ValueError(
                            "Input must be either a list/array of images "
                            "or a list of valid image paths."
                        )
                    # Preload test for sanity check
                    shapes = [self.load_img(v).shape for v in self.inputs]
                    self.data_is_npy_alike = False
                else:
                    shapes = [v.shape for v in self.inputs]
                    self.data_is_npy_alike = True

                if any(len(v) != 3 for v in shapes):
                    raise ValueError("Each sample must be an array of the form HWC.")

                max_shape = np.max(shapes, axis=0)
                if (shapes - max_shape[None]).sum() != 0:
                    raise ValueError("Images must have the same dimensions.")

            # If input is a numpy array
            elif isinstance(self.inputs, np.ndarray):
                # Check that input array is numerical
                if not np.issubdtype(self.inputs.dtype, np.number):
                    # ndarray of mixed data types
                    raise ValueError("Provided input array is non-numerical.")
                # N H W C | N C H W
                if len(self.inputs.shape) != 4:
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
            if not isinstance(self.inputs, (list, np.ndarray)):
                raise ValueError("inputs should be a list of patch coordinates")

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

    @staticmethod
    def preproc(image):
        """Define the pre-processing of this class of loader."""
        return image

    @property
    def preproc_func(self):
        """Return the current pre-processing function of this instance.

        The returned function is expected to behave as follows:
        >>> transformed_img = func(img)

        """
        return self._preproc

    @preproc_func.setter
    def preproc_func(self, func):
        """Set the pre-processing function for this instance.

        If `func=None`, the method will default to `self.preproc`. Otherwise,
          `func` is expected to be callable and behave as follows:
        >>> transformed_img = func(img)

        """
        if func is None:
            self._preproc = self.preproc
        elif callable(func):
            self._preproc = func
        else:
            raise ValueError(f"{func} is not callable!")

    def __len__(self):
        return len(self.inputs)

    @abstractmethod
    def __getitem__(self, idx):
        ...
