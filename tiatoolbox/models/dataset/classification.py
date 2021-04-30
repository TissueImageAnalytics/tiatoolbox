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

import glob # ! to be deprecated later
import os
import torch
import pathlib
import PIL
import numpy as np
import torchvision.transforms as transforms

from tiatoolbox.utils.misc import grab_files_from_dir, imread


# ! Internal usage only ? If that is, dont need a func but ___var instead
def preproc_info(pretrained):
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

    # TODO: @Simon what is labels ?
    def __init__(self, img_list, label_list=None, return_label=False, preproc_list=None):
        super().__init__()

        if preproc_list is None:
            self.preproc = lambda x: x
        else:
            self.preproc = lambda x: self.preprocess_one_image(x, preproc_list)

        # !!! May want to decouple this portion into an util later
        # Input Integrity checking
        if not (any([not isinstance(v, str) for v in img_list]) \
            and any([not isinstance(v, np.ndarray) for v in img_list])):
            # mixed data types
            raise ValueError("Input must be either a list/array of images" 
                                "or a list of valid paths to image.")
    
        if isinstance(img_list, list): # a list of path
            if any([not os.path.exists(v) for v in img_list]):
                # one or many members are not valid paths
                raise ValueError("Input must be either a list/array of images" 
                                 "or a list of valid paths to image.")
            # preload test for sanity check
            try:
                shape_list = [self.__load_img(v).shape for v in img_list]
            except:
                raise ValueError("Invalid path") # may want to raise actual error

            shape_list = np.array(shape_list)
            max_shape = np.max(shape_list, axis=0)
            if (shape_list - max_shape[None]).sum() != 0:
                raise ValueError("A path point to an image of differen shape.")
        elif not np.issubdtype(img_list, np.number):
            # mixed data types
            raise ValueError("Input must be either a list/array of images" 
                            "or a list of valid paths to image.")
        else: # numeric np.ndarray
            if len(img_list.shape) != 4: # NHWC
                raise ValueError("Input must be a list/array of images of form NHWC")
        
        if label_list is None:
            label_list = [None for i in range(len(img_list))]

        self.img_list = img_list
        self.label_list = label_list
        self.return_label = return_label
        self.classes = np.unique(label_list).tolist() # TODO comment for attribute access ?
        return

    def __load_img(self, path):
        path = pathlib.Path(path)
        if patch.suffix == ".npy":
            patch = np.load(patch)
        elif patch.suffix in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            patch = imread(patch)
        else:
            raise ValueError("Input must be either a list/array of images" 
                             "or a list of valid paths to image.")
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

        if isinstance(patch, (str, pathlib.Path)):
            if patch.suffix == "npy":
                patch = np.load(patch)
            else:
                patch = imread(patch)

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
    def __init__(self, root_dir=None, return_label=False, preproc_list=None):
        if preproc_list is None:
            preproc_info = preproc_info['kather']

        self.preproc = lambda x: self.preprocess_one_image(x, preproc_list)

        label_code_list = [
            '01_TUMOR ', '02_STROMA',  '03_COMPLEX'  '04_LYMPHO',  
            '05_DEBRIS', '06_MUCOSA',  '07_ADIPOSE'  '08_EMPTY',
        ]

        # ! TODO: @Simon sync with download protocol in model
        if root_dir is None:
            assert False

        all_path_list = []
        for label_id, label_code in enumerate(label_code_list):
            # TODO: switch to tia internal func `grab_files_from_dir`
            # however, need to upgrade it with os.walk / regex instead of basic glob
            # because glob cant read path with special characters!
            path_list = glob.glob('%s/%s/*.tif' % (root_dir, label_code))            
            path_list = [[v, label_id] for v in path_list]
            path_list.sort()
            all_path_list.extend(path_list)
        img_list, label_list = list(zip(*all_path_list))

        self.img_list = img_list
        self.label_list = label_list 
        self.classes = label_code_list