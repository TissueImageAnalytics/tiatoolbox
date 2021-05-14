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
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.wsicore.wsimeta import WSIMeta

import numpy as np
import PIL
import torch
import torchvision.transforms as transforms

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
        and its label. If `label_list` is `None`, `None` is returned

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
        self.data_check(mode="patch")

        if self.label_list is None:
            self.label_list = [np.nan for i in range(len(self.input_list))]

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
    """Defines a WSI-level patch dataset."""

    def __init__(
        self,
        wsi_file,
        label,
        mask,
        mode="wsi",
        return_labels=False,
        preproc_func=None,
        # may want a set or sthg
        patch_shape=None,  # at requested read resolution, not wrt to lv0
        stride_shape=None,  # at requested read resolution, not wrt to lv0
        resolution=None,
        units=None,
    ):
        super().__init__(return_labels=return_labels, preproc_func=preproc_func)

        if not os.path.isfile(wsi_file):
            raise ValueError("Input must be a valid file path.")

        self.label = label  # assume a single label for the whole-slide image
        if mode == "wsi":
            self.wsi_reader = get_wsireader(wsi_file)
        else:
            self.wsi_reader = VirtualWSIReader(wsi_file)

        # ! this is level 0 HW, currently dont have any method to querry
        # ! max HW at request read resolution
        # may need to ask John if this does what I think it is
        # dummy read to retrieve scaling factor for lv0

        # !!! this entire portion should be done outside of this,
        # !!! may be in predict_engine just as hovernet
        # the scaling factor will scale base level to requested read resolution/units
        _, _, _, _, lv0_patch_shape = self.wsi_reader.find_read_rect_params(
            location=(0, 0),
            size=patch_shape,
            resolution=resolution,
            units=units,
        )
        scale = lv0_patch_shape / patch_shape
        stride_shape = patch_shape if stride_shape is None else stride_shape
        lv0_pyramid_shape = self.wsi_reader.info.slide_dimensions
        lv0_stride_shape = (stride_shape * scale).astype(np.int32)
        # lv0 topleft coordinates
        # self.input_list = get_patch_coords_info(lv0_pyramid_shape, lv0_stride_shape)
        self.input_list = PatchExtractor.get_coordinates(
            lv0_pyramid_shape, lv0_patch_shape, lv0_stride_shape
        )
        self.input_list = PatchExtractor.mask_coordinates(self.input_list, mask=mask)
        # ! TODO: apply filtering basing on masks
        self.patch_shape = patch_shape
        self.lv0_patch_shape = lv0_patch_shape
        self.read_kwargs = dict(
            resolution=resolution,
            units=units,
        )

        # Perform check on the input
        self.check_input_integrity(mode="wsi")

        # !
        # if self.label_list is None:
        #     self.label_list = [np.nan for i in range(len(self.input_list))]

    def __getitem__(self, idx):
        lv0_top_left = self.input_list[idx]
        lv0_bot_right = lv0_top_left + self.lv0_patch_shape
        lv0_coords = lv0_top_left.tolist() + lv0_bot_right.tolist()
        # Read image patch from the whole-slide image
        patch = self.wsi_reader.read_bounds(
            lv0_coords,
            **self.read_kwargs,
        )
        # ! due to internal scaling, there will be rounding error and wont match
        # ! the requested size at requested read resolution
        # ! hence must apply rescale again
        patch = imresize(img=patch, output_size=self.patch_shape)

        # Apply preprocessing to selected patch
        patch = self.preproc_func(patch)

        data = {"image": patch, "coords": np.array(lv0_coords)}
        if self.label is not None:
            data["label"] = self.label  # assume same label per WSI
            return data

        return data


class KatherPatchDataset(abc.__ABCPatchDataset):
    """Define a dataset class specifically for the Kather dataset, obtain from [URL].

    Attributes:
        save_dir_path (str or None): Path to directory containing the Kather dataset,
                 assumed to be as is after extracted. If the argument is `None`,
                 the dataset will be downloaded and extracted into the
                 'run_dir/download/Kather'.

        preproc_list: List of preprocessing to be applied. If not provided, by default
                      the following are applied in sequential order.

    """

    def __init__(
        self,
        save_dir_path=None,
        return_labels=False,
        preproc_func=None,
    ):
        super().__init__(return_labels=return_labels, preproc_func=preproc_func)

        self.data_is_npy_alike = False

        label_code_list = [
            "01_TUMOR",
            "02_STROMA",
            "03_COMPLEX",
            "04_LYMPHO",
            "05_DEBRIS",
            "06_MUCOSA",
            "07_ADIPOSE",
            "08_EMPTY",
        ]

        if save_dir_path is None:
            save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "dataset/")
            if not os.path.exists(save_dir_path):
                save_zip_path = os.path.join(save_dir_path, "Kather.zip")
                url = (
                    "https://zenodo.org/record/53169/files/"
                    "Kather_texture_2016_image_tiles_5000.zip"
                )
                download_data(url, save_zip_path)
                unzip_data(save_zip_path, save_dir_path)
            save_dir_path = os.path.join(
                save_dir_path, "Kather_texture_2016_image_tiles_5000/"
            )
        elif not os.path.exists(save_dir_path):
            raise ValueError("Dataset does not exist at `%s`" % save_dir_path)

        # What will happen if downloaded data get corrupted?
        all_path_list = []
        for label_id, label_code in enumerate(label_code_list):
            path_list = grab_files_from_dir(
                "%s/%s/" % (save_dir_path, label_code), file_types="*.tif"
            )
            path_list = [[v, label_id] for v in path_list]
            path_list.sort()
            all_path_list.extend(path_list)
        input_list, label_list = list(zip(*all_path_list))

        self.input_list = input_list
        self.label_list = label_list
        self.classes = label_code_list

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
        if self.return_labels:
            data["label"] = self.label_list[idx]
            return data
        return data
