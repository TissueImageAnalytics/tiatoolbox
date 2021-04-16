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


import torch
import pathlib
import PIL
import numpy as np
import torchvision.transforms as transforms

from tiatoolbox.utils.misc import grab_files_from_dir, imread


class Patch_Dataset(torch.utils.data.Dataset):
    """Defines a simple patch dataset, which inherits
    from the torch.utils.data.Dataset class.

    Attributes:
        patch_info: Either a list of patches, where each patch is a ndarray or a directory
                to a list of patches. If a directory is provided, then patches should
                be either a standard image file (.png, .jpg etc) or a .npy file.
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

    def __init__(self, patch_info, labels=None, preproc_list=None):
        super().__init__()

        if preproc_list is None:
            self.preproc = lambda x: x
        else:
            self.preproc = lambda x: self.preprocess_one_image(x, preproc_list)

        if isinstance(patch_info, list):
            self.patch_info = patch_info
        else:
            file_types = ("*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.png", "*.npy")
            self.patch_info = grab_files_from_dir(
                input_path=patch_info, file_types=file_types
            )

        return

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
            patch = transforms.Compose(preproc_list)
            patch = patch.permute(1, 2, 0)
        return patch

    def __len__(self):
        return len(self.patch_info)

    def __getitem__(self, idx):
        patch = self.patch_list[idx]

        if isinstance(patch, (str, pathlib.Path)):
            if patch.suffix == "npy":
                patch = np.load(patch)
            else:
                patch = imread(patch)

        # apply preprocessing to selected patch
        patch = self.preproc(patch)
        return patch