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


<<<<<<< HEAD
=======
from abc import ABC, abstractmethod
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
import os
import pathlib

import numpy as np
import torch

from tiatoolbox.utils.misc import imread


<<<<<<< HEAD
class ABCPatchDataset(torch.utils.data.Dataset):
=======
class PatchDatasetABC(ABC, torch.utils.data.Dataset):
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
    """Defines abstract base class for patch dataset.

    Attributes:
        return_labels (bool, False): `__getitem__` will return both the img and
<<<<<<< HEAD
        its label. If `label_list` is `None`, `None` is returned

        preproc_func: Preprocessing function used to transform the input data. If
        supplied, then torch.Compose will be used on the input preproc_list.
        preproc_list is a list of torchvision transforms for preprocessing the image.
=======
        its label. If `labels` is `None`, `None` is returned

        preproc_func: Preprocessing function used to transform the input data. If
        supplied, then torch.Compose will be used on the input preprocs.
        preprocs is a list of torchvision transforms for preprocessing the image.
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
        The transforms will be applied in the order that they are given in the list.
        https://pytorch.org/vision/stable/transforms.html.

    """

<<<<<<< HEAD
    def __init__(self, preproc_func=None):
        super().__init__()
        self.preproc = preproc_func
        self.data_is_npy_alike = False
        self.input_list = []
        self.label_list = []
=======
    def __init__(
        self,
    ):
        super().__init__()
        self._preproc = self.preproc
        self.data_is_npy_alike = False
        self.inputs = []
        self.labels = []
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18

    def _check_input_integrity(self, mode):
        """Perform check to make sure variables received during init are valid.

        These check include:
            - Input is of a singular data type, such as a list of paths.
            - If it is list of images, all images are of the same height and width
        """
        if mode == "patch":
            self.data_is_npy_alike = False

            # If input is a list - can contain a list of images or a list of image paths
<<<<<<< HEAD
            if isinstance(self.input_list, list):
                is_all_path_list = all(
                    isinstance(v, (pathlib.Path, str)) for v in self.input_list
                )
                is_all_npy_list = all(
                    isinstance(v, np.ndarray) for v in self.input_list
                )
                if not (is_all_path_list or is_all_npy_list):
=======
            if isinstance(self.inputs, list):
                is_all_paths = all(
                    isinstance(v, (pathlib.Path, str)) for v in self.inputs
                )
                is_all_npys = all(isinstance(v, np.ndarray) for v in self.inputs)
                if not (is_all_paths or is_all_npys):
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
                    raise ValueError(
                        "Input must be either a list/array of images "
                        "or a list of valid image paths."
                    )

<<<<<<< HEAD
                shape_list = []
                # When a list of paths is provided
                if is_all_path_list:
                    if any(not os.path.exists(v) for v in self.input_list):
=======
                shapes = []
                # When a list of paths is provided
                if is_all_paths:
                    if any(not os.path.exists(v) for v in self.inputs):
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
                        # at least one of the paths are invalid
                        raise ValueError(
                            "Input must be either a list/array of images "
                            "or a list of valid image paths."
                        )
                    # Preload test for sanity check
<<<<<<< HEAD
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
=======
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
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
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
<<<<<<< HEAD
            if not isinstance(self.input_list, (list, np.ndarray)):
                raise ValueError("input_list should be a list of patch coordinates")
=======
            if not isinstance(self.inputs, (list, np.ndarray)):
                raise ValueError("inputs should be a list of patch coordinates")
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18

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

<<<<<<< HEAD
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
=======
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
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
