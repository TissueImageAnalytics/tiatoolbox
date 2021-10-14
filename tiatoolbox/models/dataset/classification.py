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
import warnings

import cv2
import numpy as np
import PIL
import torchvision.transforms as transforms

from tiatoolbox.models.dataset import abc
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore.wsimeta import WSIMeta
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, get_wsireader


class _TorchPreprocCaller:
    """Wrapper for applying PyTorch transforms.

    Args:
        preprocs (list): List of torchvision transforms for preprocessing the image.
          The transforms will be applied in the order that they are given in the
          list. For more informaion, visit the following link:
          https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(self, preprocs):
        self.func = transforms.Compose(preprocs)

    def __call__(self, img):
        img = PIL.Image.fromarray(img)
        img = self.func(img)
        img = img.permute(1, 2, 0)
        return img


def predefined_preproc_func(dataset_name):
    """Get the preprocessing information used for the pretrained model.

    Args:
        dataset_name (str): Dataset name used to determine what preprocessing was used.
    Returns:
        preproc_func (_TorchPreprocCaller): Preprocessing function for transforming
          the input data.

    """
    preproc_dict = {
        "kather100k": [
            transforms.ToTensor(),
        ]
    }
    if dataset_name not in preproc_dict:
        raise ValueError(
            f"Predefined preprocessing for dataset `{dataset_name}` does not exist."
        )

    preprocs = preproc_dict[dataset_name]
    preproc_func = _TorchPreprocCaller(preprocs)
    return preproc_func


class PatchDataset(abc.PatchDatasetABC):
    """Defines a simple patch dataset, which inherits from the
      torch.utils.data.Dataset class.

    Attributes:
        inputs: Either a list of patches, where each patch is a ndarray or a list of
          valid path with its extension be
          (".jpg", ".jpeg", ".tif", ".tiff", ".png") pointing to an image.
        labels: List of label for sample at the same index in `inputs`.
          Default is `None`.
        preproc_func: Preprocessing function used to transform the input data.

    Examples:
        >>> # an user defined preproc func and expected behavior
        >>> preproc_func = lambda img: img/2  # reduce intensity by half
        >>> transformed_img = preproc_func(img)
        >>> # create a dataset to get patches preprocessed by the above function
        >>> ds = PatchDataset(
        ...     inputs=['/A/B/C/img1.png', '/A/B/C/img2.png'],
        ...     preproc_func=preproc_func
        ... )

    """

    def __init__(self, inputs, labels=None):
        super().__init__()

        self.data_is_npy_alike = False

        self.inputs = inputs
        self.labels = labels

        # perform check on the input
        self._check_input_integrity(mode="patch")

    def __getitem__(self, idx):
        patch = self.inputs[idx]

        # Mode 0 is list of paths
        if not self.data_is_npy_alike:
            patch = self.load_img(patch)

        # Apply preprocessing to selected patch
        patch = self._preproc(patch)

        data = {
            "image": patch,
        }
        if self.labels is not None:
            data["label"] = self.labels[idx]
            return data

        return data


class WSIPatchDataset(abc.PatchDatasetABC):
    """Defines a WSI-level patch dataset.

    Attributes:
        reader (:class:`.WSIReader`): an WSI Reader or Virtual Reader
          for reading pyramidal image or large tile in pyramidal way.
        inputs: List of coordinates to read from the `reader`,
          each coordinate is of the form [start_x, start_y, end_x, end_y].
        patch_size: a tuple(int, int) or ndarray of shape (2,).
          Expected size to read from `reader` at requested `resolution`
          and `units`. Expected to be (height, width).
        lv0_patch_size: a tuple (int, int) or ndarray of shape (2,).
          `patch_size` at level 0 in `reader` at requested `resolution`
          and `units`. Expected to be (height, width).
        resolution: check (:class:`.WSIReader`) for details.
        units: check (:class:`.WSIReader`) for details.
        preproc_func: Preprocessing function used to transform the input data.
          If supplied, then torch.Compose will be used on the input preprocs.
          preprocs is a list of torchvision transforms for preprocessing the
          image. The transforms will be applied in the order that they are
          given in the list. For more information, visit the following link:
          https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(
        self,
        img_path,
        mode="wsi",
        mask_path=None,
        patch_size=None,
        stride_size=None,
        resolution=None,
        units=None,
        auto_get_mask=True,
    ):
        """Create a WSI-level patch dataset.
        Args:
            mode (str): can be either `wsi` or `tile` to denote the image to read is
              either a whole-slide image or a large image tile.
            img_path (:obj:`str` or :obj:`pathlib.Path`): valid to pyramidal
              whole-slide image or large tile to read.
            mask_path (:obj:`str` or :obj:`pathlib.Path`): valid mask image.
            patch_size: a tuple (int, int) or ndarray of shape (2,).
              Expected shape to read from `reader` at requested `resolution` and
              `units`. Expected to be positive and of (height, width). Note, this
              is not at `resolution` coordinate space.
            stride_size: a tuple (int, int) or ndarray of shape (2,).
              Expected stride shape to read at requested `resolution` and `units`.
              Expected to be positive and of (height, width). Note, this is not at
              level 0.
            resolution: check (:class:`.WSIReader`) for details. When `mode='tile'`,
              value is fixed to be `resolution=1.0` and `units='baseline'`
              units: check (:class:`.WSIReader`) for details.
            preproc_func: Preprocessing function used to transform the input data.

        Examples:
            >>> # an user defined preproc func and expected behavior
            >>> preproc_func = lambda img: img/2  # reduce intensity by half
            >>> transformed_img = preproc_func(img)
            >>> # create a dataset to get patches from WSI with above
            >>> # preprocessing function
            >>> ds = WSIPatchDataset(
            ...     img_path='/A/B/C/wsi.svs',
            ...     mode="wsi",
            ...     patch_size=[512, 512],
            ...     stride_size=[256, 256],
            ...     auto_get_mask=False,
            ...     preproc_func=preproc_func
            ... )

        """
        super().__init__()

        # Is there a generic func for path test in toolbox?
        if not os.path.isfile(img_path):
            raise ValueError("`img_path` must be a valid file path.")
        if mode not in ["wsi", "tile"]:
            raise ValueError(f"`{mode}` is not supported.")
        patch_size = np.array(patch_size)
        stride_size = np.array(stride_size)

        if (
            not np.issubdtype(patch_size.dtype, np.integer)
            or np.size(patch_size) > 2
            or np.any(patch_size < 0)
        ):
            raise ValueError(f"Invalid `patch_size` value {patch_size}.")
        if (
            not np.issubdtype(stride_size.dtype, np.integer)
            or np.size(stride_size) > 2
            or np.any(stride_size < 0)
        ):
            raise ValueError(f"Invalid `stride_size` value {stride_size}.")

        img_path = pathlib.Path(img_path)
        if mode == "wsi":
            self.reader = get_wsireader(img_path)
        else:
            warnings.warn(
                (
                    "WSIPatchDataset only reads image tile at "
                    '`units="baseline"` and `resolution=1.0`.'
                )
            )
            units = "baseline"
            resolution = 1.0
            img = imread(img_path)
            # initialise metadata for VirtualWSIReader.
            # here, we simulate a whole-slide image, but with a single level.
            # ! should we expose this so that use can provide their metadat ?
            metadata = WSIMeta(
                mpp=np.array([1.0, 1.0]),
                objective_power=10,
                slide_dimensions=np.array(img.shape[:2][::-1]),
                level_downsamples=[1.0],
                level_dimensions=[np.array(img.shape[:2][::-1])],
            )
            # hack value such that read if mask is provided is through
            # 'mpp' or 'power' as varying 'baseline' is locked atm
            units = "mpp"
            resolution = 1.0
            self.reader = VirtualWSIReader(
                img,
                metadata,
            )

        # may decouple into misc ?
        # the scaling factor will scale base level to requested read resolution/units
        wsi_shape = self.reader.slide_dimensions(resolution=resolution, units=units)

        # use all patches, as long as it overlaps source image
        self.inputs = PatchExtractor.get_coordinates(
            image_shape=wsi_shape,
            patch_input_shape=patch_size[::-1],
            stride_shape=stride_size[::-1],
            input_within_bound=False,
        )

        mask_reader = None
        if mask_path is not None:
            if not os.path.isfile(mask_path):
                raise ValueError("`mask_path` must be a valid file path.")
            mask = imread(mask_path)  # assume to be gray
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = np.array(mask > 0, dtype=np.uint8)

            mask_reader = VirtualWSIReader(mask)
            mask_reader.info = self.reader.info
        elif auto_get_mask and mode == "wsi" and mask_path is None:
            # if no mask provided and `wsi` mode, generate basic tissue
            # mask on the fly
            mask_reader = self.reader.tissue_mask(resolution=1.25, units="power")
            # ? will this mess up  ?
            mask_reader.info = self.reader.info

        if mask_reader is not None:
            selected = PatchExtractor.filter_coordinates(
                mask_reader,  # must be at the same resolution
                self.inputs,  # must already be at requested resolution
                resolution=resolution,
                units=units,
            )
            self.inputs = self.inputs[selected]

        if len(self.inputs) == 0:
            raise ValueError("No coordinate remain after tiling!")

        self.patch_size = patch_size
        self.resolution = resolution
        self.units = units

        # Perform check on the input
        self._check_input_integrity(mode="wsi")

    def __getitem__(self, idx):
        coords = self.inputs[idx]
        # Read image patch from the whole-slide image
        patch = self.reader.read_bounds(
            coords,
            resolution=self.resolution,
            units=self.units,
            pad_constant_values=255,
            coord_space="resolution",
        )

        # Apply preprocessing to selected patch
        patch = self._preproc(patch)

        data = {"image": patch, "coords": np.array(coords)}
        return data
