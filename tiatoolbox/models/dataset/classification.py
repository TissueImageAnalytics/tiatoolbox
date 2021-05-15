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
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.wsicore.wsimeta import WSIMeta

import numpy as np
import PIL
import torch
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

from tiatoolbox import rcParam
from tiatoolbox.models.dataset import abc
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, get_wsireader
from tiatoolbox.utils.misc import download_data, grab_files_from_dir, imread, unzip_data
from tiatoolbox.utils.transforms import imresize


class _TorchPreprocCaller:
    """Wrapper for applying PyTorch transforms.

    Args:
        preproc_list (list): List of torchvision transforms for preprocessing the image.
            The transforms will be applied in the order that they are
            given in the list. https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(self, preproc_list):
        self.func = transforms.Compose(preproc_list)

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
            "Predefined preprocessing for" "dataset `%s` does not exist." % dataset_name
        )

    preproc_list = preproc_dict[dataset_name]
    preproc_func = _TorchPreprocCaller(preproc_list)
    return preproc_func


class PatchDataset(abc.__ABCPatchDataset):
    """Defines a simple patch dataset, which inherits
    from the torch.utils.data.Dataset class.

    Attributes:
        input_list: Either a list of patches, where each patch is a ndarray or a list of
        valid path with its extension be (".jpg", ".jpeg", ".tif", ".tiff", ".png")
        pointing to an image.

        label_list: List of label for sample at the same index in `input_list` .
        Default is `None`.

        return_labels (bool, False): `__getitem__` will return both the img
        and its label. If `label_list` is `None`, `None` is returned.

        preproc_func: Preprocessing function used to transform the input data. If
        supplied, then torch.Compose will be used on the input preproc_list.
        preproc_list is a list of torchvision transforms for preprocessing the image.
        The transforms will be applied in the order that they are given in the list.
        https://pytorch.org/vision/stable/transforms.html.

    Examples:
        >>> from tiatoolbox.models.data import Patch_Dataset
        >>> mean = [0.485, 0.456, 0.406]
        >>> std = [0.229, 0.224, 0.225]
        >>> preproc_list =
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ]
        >>> ds = Patch_Dataset('/path/to/data/', preproc_list=preproc_list)

    """

    def __init__(self, input_list, label_list=None, preproc_func=None):
        super().__init__(preproc_func=preproc_func)

        self.data_is_npy_alike = False

        self.input_list = input_list
        self.label_list = label_list

        # perform check on the input
        self.check_input_integrity(mode="patch")

    def __getitem__(self, idx):
        patch = self.input_list[idx]

        # Mode 0 is list of paths
        if not self.data_is_npy_alike:
            patch = self.load_img(patch)

        # Apply preprocessing to selected patch
        patch = self.preproc_func(patch)

        data = {
            "image": patch,
        }
        if self.label_list is not None:
            data["label"] = self.label_list[idx]
            return data
        return data


class WSIPatchDataset(abc.__ABCPatchDataset):
    """Defines a WSI-level patch dataset.

    Attributes:
        reader (:class:`.WSIReader`): an WSI Reader or Virtual Reader
        for reading pyramidal image or large tile in pyramidal way.

        input_list: List of coordinates to read from the `reader`,
        each coordinate is of the form [start_x, start_y, end_x, end_y].

        patch_shape: a tuple(int, int) or ndarray of shape (2,).
        Expected shape to read from `reader` at requested `resolution` and `units`.
        Expected to be (height, width).

        lv0_patch_shape: a tuple(int, int) or ndarray of shape (2,).
        Shape of `patch_shape` at level 0 in `reader` at requested
        `resolution` and `units`. Expected to be (height, width).

        resolution: check (:class:`.WSIReader`) for details.
        units: check (:class:`.WSIReader`) for details.

        preproc_func: Preprocessing function used to transform the input data. If
        supplied, then torch.Compose will be used on the input preproc_list.
        preproc_list is a list of torchvision transforms for preprocessing the image.
        The transforms will be applied in the order that they are given in the list.
        https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(
        self,
        wsi_path,
        mode="wsi",
        mask_path=None,
        preproc_func=None,
        patch_shape=None,
        stride_shape=None,
        resolution=None,
        units=None,
    ):
        """Create a WSI-level patch dataset.

        Args:
            mode (str): can be either `wsi` or `tile` to denote the image to read is
            pyramidal or a large tile.

            wsi_path (:obj:`str` or :obj:`pathlib.Path`): valid to pyramidal image
            or large tile to read.

            mask_path (:obj:`str` or :obj:`pathlib.Path`): valid mask image.

            patch_shape: a tuple (int, int) or ndarray of shape (2,).
            Expected shape to read from `reader` at requested `resolution` and `units`.
            Expected to be (height, width). Note, this is not at level 0.

            stride_shape: a tuple (int, int) or ndarray of shape (2,).
            Expected stride shape to read at requested `resolution` and `units`.
            Expected to be (height, width). Note, this is not at level 0.

            resolution: check (:class:`.WSIReader`) for details. When `mode='tile'`,
            value is fixed to be `resolution=1.0` and `units='baseline'`
            units: check (:class:`.WSIReader`) for details.

            preproc_func: Preprocessing function used to transform the input data.
            If supplied, then torch.Compose will be used on the input preproc_list.
            preproc_list is a list of torchvision transforms for preprocessing the
            image. The transforms will be applied in the order that they are given
            in the list. https://pytorch.org/vision/stable/transforms.html.

        """
        super().__init__(preproc_func=preproc_func)
        # Is there a generic func for path test in toolbox?
        if not os.path.isfile(wsi_path):
            raise ValueError("`wsi_path` must be a valid file path.")
        if mode not in ["wsi", "tile"]:
            raise ValueError("`%s` is not supported." % mode)
        patch_shape = np.array(patch_shape)
        stride_shape = np.array(stride_shape)
        # ! dont do the checking for patch at this stage and let
        # ! get coordinates deal with it? EXTRREMELY UGLY
        # print(patch_shape)
        if not np.issubdtype(patch_shape.dtype, np.integer) or \
                np.size(patch_shape) > 2 or np.any(patch_shape < 0):
            raise ValueError('Invalid `patch_shape` value %s.' % patch_shape)
        if not np.issubdtype(stride_shape.dtype, np.integer) or \
                np.size(stride_shape) > 2 or np.any(stride_shape < 0):
            raise ValueError('Invalid `stride_shape` value %s.' % stride_shape)
        if np.any(stride_shape < 1):
            raise ValueError('`stride_shape` value %s must > 1.' % stride_shape)

        # ! We must do conversion else wsireader will error out
        wsi_path = pathlib.Path(wsi_path)
        if mode == "wsi":
            self.reader = get_wsireader(wsi_path)
        else:
            warnings.warn((
                "WSIPatchDataset only read tile at "
                '`units="baseline"` and `resolution=1.0`.'
            ))
            # overwriting for later read
            # units = 'mpp'
            # resolution = 1.0
            units = "baseline"
            resolution = 1.0
            img = imread(wsi_path)
            metadata = WSIMeta(
                mpp=np.array([0.25, 0.25]),
                slide_dimensions=np.array(img.shape[:2][::-1]),
                level_downsamples=[1.0],
                level_dimensions=[np.array(img.shape[:2][::-1])],
            )
            # any value for mpp is fine, but the read
            # resolution for mask later must match
            # ? alignement, XY or YX ? Default to XY
            # ? to match OpenSlide for now
            self.reader = VirtualWSIReader(
                img,
                metadata,
            )

        # may decouple into misc ?
        # the scaling factor will scale base level to requested read resolution/units
        _, _, _, _, lv0_patch_shape = self.reader.find_read_rect_params(
            location=(0, 0),
            size=patch_shape,
            resolution=resolution,
            units=units,
        )
        wsi_metadata = self.reader.info
        scale = lv0_patch_shape / patch_shape
        stride_shape = patch_shape if stride_shape is None else stride_shape
        lv0_pyramid_shape = wsi_metadata.slide_dimensions
        lv0_stride_shape = (stride_shape * scale).astype(np.int32)
        # lv0 topleft coordinates
        self.input_list = PatchExtractor.get_coordinates(
            lv0_pyramid_shape[::-1], lv0_patch_shape, lv0_stride_shape
        )

        if len(self.input_list) == 0:
            raise ValueError('No coordinate remain after tiling!')

        if mask_path is not None:
            # ? extension checking
            if not os.path.isfile(wsi_path):
                raise ValueError("`mask_path` must be a valid file path.")

            mask = imread(mask_path)
            mask_reader = VirtualWSIReader(mask)
            mask_reader.attach_to_reader(self.reader.info)
            # * now filter coordinate basing on the mask
            # scaling factor between mask lv0 and source reader lv0
            scale = mask_reader.info.level_downsamples[0]
            scaled_input_list = self.input_list / scale
            sel = PatchExtractor.filter_coordinates(
                mask_reader,  # must be at the same resolution
                scaled_input_list,  # must be at the same resolution
                resolution=resolution,
                units=units,
            )
            self.input_list = self.input_list[sel]

        if len(self.input_list) == 0:
            raise ValueError('No coordinate remain after tiling!')

        self.patch_shape = patch_shape
        self.lv0_patch_shape = lv0_patch_shape
        self.resolution = resolution
        self.units = units

        # Perform check on the input
        self.check_input_integrity(mode="wsi")

    def __getitem__(self, idx):
        lv0_coords = self.input_list[idx]
        # Read image patch from the whole-slide image
        patch = self.reader.read_bounds(
            lv0_coords, resolution=self.resolution, units=self.units
        )
        # ! due to internal scaling, there will be rounding error and wont match
        # ! the requested size at requested read resolution
        # ! hence must apply rescale again
        patch = imresize(img=patch, output_size=self.patch_shape)

        # Apply preprocessing to selected patch
        patch = self.preproc_func(patch)

        # ? how to enforce return check?
        data = {"image": patch, "coords": np.array(lv0_coords)}
        return data
