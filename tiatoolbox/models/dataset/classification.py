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


class PatchDataset(abc.ABCPatchDataset):
    """Defines a simple patch dataset, which inherits
    from the torch.utils.data.Dataset class.

    Attributes:
        input_list: Either a list of patches, where each patch is a ndarray or a list of
        valid path with its extension be (".jpg", ".jpeg", ".tif", ".tiff", ".png")
        pointing to an image.

        label_list: List of label for sample at the same index in `input_list` .
        Default is `None`.

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


class WSIPatchDataset(abc.ABCPatchDataset):
    """Defines a WSI-level patch dataset.

    Attributes:
        reader (:class:`.WSIReader`): an WSI Reader or Virtual Reader
        for reading pyramidal image or large tile in pyramidal way.

        input_list: List of coordinates to read from the `reader`,
        each coordinate is of the form [start_x, start_y, end_x, end_y].

        patch_size: a tuple(int, int) or ndarray of shape (2,).
        Expected size to read from `reader` at requested `resolution` and `units`.
        Expected to be (height, width).

        lv0_patch_size: a tuple (int, int) or ndarray of shape (2,).
        `patch_size` at level 0 in `reader` at requested `resolution`
        and `units`. Expected to be (height, width).

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
        img_path,
        mode="wsi",
        mask_path=None,
        preproc_func=None,
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
            Expected shape to read from `reader` at requested `resolution` and `units`.
            Expected to be (height, width). Note, this is not at level 0.

            stride_size: a tuple (int, int) or ndarray of shape (2,).
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
        if not os.path.isfile(img_path):
            raise ValueError("`img_path` must be a valid file path.")
        if mode not in ["wsi", "tile"]:
            raise ValueError("`%s` is not supported." % mode)
        patch_size = np.array(patch_size)
        stride_size = np.array(stride_size)

        if (
            not np.issubdtype(patch_size.dtype, np.integer)
            or np.size(patch_size) > 2
            or np.any(patch_size < 0)
        ):
            raise ValueError("Invalid `patch_size` value %s." % patch_size)
        if (
            not np.issubdtype(stride_size.dtype, np.integer)
            or np.size(stride_size) > 2
            or np.any(stride_size < 0)
        ):
            raise ValueError("Invalid `stride_size` value %s." % stride_size)
        if np.any(stride_size < 1):
            raise ValueError("`stride_size` value %s must be > 1." % stride_size)

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

        self.input_list = PatchExtractor.get_coordinates(
            wsi_shape, patch_size[::-1], stride_size[::-1]
        )

        if len(self.input_list) == 0:
            raise ValueError("No coordinate remain after tiling!")

        mask_reader = None
        if mask_path is not None:
            if not os.path.isfile(mask_path):
                raise ValueError("`mask_path` must be a valid file path.")
            mask = imread(mask_path)
            mask_reader = VirtualWSIReader(mask)
            mask_reader.attach_to_reader(self.reader.info)
        elif auto_get_mask and mode == "wsi" and mask_path is None:
            # if no mask provided and `wsi` mode, generate basic tissue
            # mask on the fly
            mask_reader = self.reader.tissue_mask(resolution=1.25, units="power")
            # ? will this mess up ?
            mask_reader.attach_to_reader(self.reader.info)

        # ! should update the WSIReader such that sync read can be done on
        # ! with `baseline` input as well
        if mask_reader is not None and units in ['baseline', 'level']:
            raise ValueError("Mask can't be used with `resolution=%s`" % units)

        if mask_reader is not None:
            selected = PatchExtractor.filter_coordinates(
                mask_reader,  # must be at the same resolution
                self.input_list,  # must already be at requested resolution
                resolution=resolution,
                units=units,
            )
            self.input_list = self.input_list[selected]

        if len(self.input_list) == 0:
            raise ValueError("No coordinate remain after tiling!")

        self.patch_size = patch_size
        self.resolution = resolution
        self.units = units

        # Perform check on the input
        self.check_input_integrity(mode="wsi")

    def __getitem__(self, idx):
        coords = self.input_list[idx]
        # Read image patch from the whole-slide image
        patch = self.reader.read_bounds(
            coords,
            resolution=self.resolution,
            units=self.units,
            pad_constant_values=255,
            location_is_at_requested=True,
        )

        # Apply preprocessing to selected patch
        patch = self.preproc_func(patch)

        data = {"image": patch, "coords": np.array(coords)}
        return data
