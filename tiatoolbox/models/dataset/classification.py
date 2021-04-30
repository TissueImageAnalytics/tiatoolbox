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

import glob  # ! to be deprecated later
import os
import torch
import pathlib
import requests
import PIL
import numpy as np
import torchvision.transforms as transforms

from tiatoolbox.utils.misc import grab_files_from_dir, imread, download_and_unzip_data


def _preproc_info(pretrained):
    """Get the preprocessing information used for the pretrained model.

    Args:
        pretrained (str): xx.

    """
    preproc_dict = {
        "kather": [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    }

    preproc_list = preproc_dict[pretrained]

    return preproc_list


class Patch_Dataset(torch.utils.data.Dataset):
    """Defines a simple patch dataset, which inherits
    from the torch.utils.data.Dataset class.

    Attributes:
        img_list: Either a list of patches, where each patch is a ndarray or a list of valid path
                 with its extension be (".jpg", ".jpeg", ".tif", ".tiff", ".png") pointing to
                 an image.

        label_list: List of label for sample at the same index in `img_list` .
                 Default is `None`

        return_label (bool, False): __getitem__ will return both the img and its label.
                If `label_list` is `None`, `None` is returned

        preproc: preprocessing function used to transform the input data. If supplied, then
                 torch.Compose will be used on the input preproc_list. preproc_list is a
                 list of torchvision transforms for preprocessing the image.
                 The transforms will be applied in the order that they are
                 given in the list. https://pytorch.org/vision/stable/transforms.html.

    Examples:
        >>> from tiatoolbox.models.data import Patch_Dataset
        >>> preproc_list = [transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ]
        >>> ds = Patch_Dataset('/path/to/data/', preproc_list=preproc_list)

    """

    def __init__(
        self, img_list, label_list=None, return_label=False, preproc_list=None
    ):
        super().__init__()

        if preproc_list is None:
            self.preproc = lambda x: x
        else:
            self.preproc = lambda x: self.preprocess_one_image(x, preproc_list)

        self.data_mode = -1

        # perform check on the input

        # if input is a list - can contain a list of images or a list of image paths
        if isinstance(img_list, list):
            all_path_list = [isinstance(v, str) for v in img_list]
            all_npy_list = [isinstance(v, np.ndarray) for v in img_list]
            if not (any(all_path_list) or any(all_npy_list)):
                raise ValueError(
                    "Input must be either a list/array of images "
                    "or a list of valid image paths."
                )

            shape_list = []
            # when a list of paths is provided
            if any(all_path_list):
                if any([not os.path.exists(v) for v in img_list]):
                    # at least one of the paths are invalid
                    raise ValueError(
                        "At least one of the provided image paths does not exist."
                    )
                # preload test for sanity check
                try:
                    shape_list = [self.__load_img(v).shape for v in img_list]
                except:
                    raise ValueError(
                        "At least one of the provided image paths is invalid. "
                        "Check to make sure the supplied paths correspond to image files. Supported "
                        "image formats include: `.npy`, `.jpg`, `.jpeg`, `.tif`, `.tiff` or `.png`."
                    )
                self.data_mode = 0
            else:
                shape_list = [v.shape for v in img_list]
                self.data_mode = 1

            max_shape = np.max(shape_list, axis=0)
            # how will this behave for mixed channel ?
            if (shape_list - max_shape[None]).sum() != 0:
                raise ValueError("Images must have the same dimensions")

        # if input is a numpy array
        elif isinstance(img_list, np.ndarray):
            # check that input array is numerical
            if not np.issubdtype(img_list.dtype, np.number):
                # ndarray of mixed data types
                raise ValueError("Provided input array is non-numerical.")
            # N H W C | N C H W
            if len(img_list.shape) != 4:
                raise ValueError(
                    "Input must be an array of images of the form NHWC. This can "
                    "be achieved by converting a list of images to a numpy array. "
                    " eg., np.array([img1, img2])."
                )
            self.data_mode = 2

        else:
            raise ValueError(
                "Input must be either a list/array of images "
                "or a list of valid paths to image."
            )

        if label_list is None:
            label_list = [None for i in range(len(img_list))]

        self.img_list = img_list
        self.label_list = label_list
        self.return_label = return_label

        return

    def __load_img(self, path):
        """Load an image from a provided path.

        Args:
            path (str): path to an image file.

        """
        path = pathlib.Path(path)
        if path.suffix == ".npy":
            patch = np.load(path)
        elif path.suffix in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            patch = imread(path)

        return patch

    @staticmethod
    def preprocess_one_image(patch, preproc_list=None):
        """Apply preprocessing to a single input image patch.

        Args:
            patch (ndarray): input image patch.
            preproc_list (list): list of torchvision transforms for preprocessing the image.
                          The transforms will be applied in the order that they are
                          given in the list. https://pytorch.org/vision/stable/transforms.html.

        Returns:
            patch (ndarray): preprocessed image patch.

        """
        if preproc_list is not None:
            # using torchvision pipeline demands input is pillow format
            patch = PIL.Image.fromarray(patch)
            trans = transforms.Compose(preproc_list)
            patch = trans(patch)
            patch = patch.permute(1, 2, 0)
        return patch

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        patch = self.img_list[idx]

        if self.data_mode == 0:  # mode 0 is list of paths
            patch = self.__load_img(patch)

        # apply preprocessing to selected patch
        patch = self.preproc(patch)

        if self.return_label:
            return patch, self.label_list[idx]

        return patch


class Kather_Patch_Dataset(Patch_Dataset):
    """Define a dataset class specifically for the Kather dataset, obtain from [URL]

    Attributes: # ! TODO: support for path object ?
        root_dir (str or None): path to directory containing the Kather dataset,
                 assumed to be as is after extracted. If the argument is `None`,
                 the dataset will be downloaded and extracted into the
                 'run_dir/download/Kather'

        preproc_list: list of preprocessing to be applied. If not provided, by default
                      the following are applied in sequential order

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    """

    def __init__(self, root_dir=None, return_label=True, preproc_list=None):
        if preproc_list is None:
            preproc_info = _preproc_info("kather")

        self.return_label = return_label
        self.preproc = lambda x: self.preprocess_one_image(x, preproc_list)
        self.data_mode = 0

        label_code_list = [
            "01_TUMOR ",
            "02_STROMA",
            "03_COMPLEX",
            "04_LYMPHO",
            "05_DEBRIS",
            "06_MUCOSA",
            "07_ADIPOSE",
            "08_EMPTY",
        ]

        if root_dir is None or not os.path.isdir(root_dir):
            print("Dataset directory not provided / invalid.")
            url = "https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip"
            unzip_dir = "data_dump/kather"  #! currently a hack - sort proper location!
            download_and_unzip_data(url, unzip_dir)

            #! TODO below is a hack - need to return the root location when unzipping the file
            root_dir = unzip_dir + "/Kather_texture_2016_image_tiles_5000"

        all_path_list = []
        for label_id, label_code in enumerate(label_code_list):
            path_list = grab_files_from_dir(
                "%s/%s/" % (root_dir, label_code), file_types="*.tif"
            )
            path_list = [[v, label_id] for v in path_list]
            path_list.sort()
            all_path_list.extend(path_list)
        img_list, label_list = list(zip(*all_path_list))

        self.img_list = img_list
        self.label_list = label_list
        self.classes = label_code_list
