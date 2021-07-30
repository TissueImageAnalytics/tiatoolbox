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
from abc import ABC, abstractmethod

from tiatoolbox import rcParam
from tiatoolbox.utils.misc import download_data, grab_files_from_dir, unzip_data


<<<<<<< HEAD
class ABCDatasetInfo(ABC):
=======
class DatasetInfoABC(ABC):
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
    """Define an abstract class for holding a dataset information.

    Enforcing such that following attributes must always be defined by the subclass.

    Attributes
<<<<<<< HEAD
        input_list (list): A list of paths where each path points to a sample image.
        label_list (list): A list of `int` where each is the label of the sample at
            the same index.
        label_name (dict): A dict indicates the possible associate name of each label
=======
        inputs (list): A list of paths where each path points to a sample image.
        labels (list): A list of `int` where each is the label of the sample at
            the same index.
        label_names (dict): A dict indicates the possible associate name of each label
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
            value.

    """

    @property
    @abstractmethod
<<<<<<< HEAD
    def input_list(self):
=======
    def inputs(self):
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
        raise NotImplementedError

    @property
    @abstractmethod
<<<<<<< HEAD
    def label_list(self):
=======
    def labels(self):
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
        raise NotImplementedError

    @property
    @abstractmethod
<<<<<<< HEAD
    def label_name(self):
        raise NotImplementedError


class KatherPatchDataset(ABCDatasetInfo):
=======
    def label_names(self):
        raise NotImplementedError


class KatherPatchDataset(DatasetInfoABC):
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
    """Define a class for holding the Kather dataset information.

    Args:
        save_dir_path (str or None): Path to directory containing the Kather dataset,
            assumed to be as is after extracted. If the argument is `None`,
            the dataset will be downloaded and extracted into the
            'run_dir/download/Kather'.

    Attributes
<<<<<<< HEAD
        input_list (list): A list of paths where each path points to a sample image.
        label_list (list): A list of `int` where each is the label of the sample at
            the same index.
        label_name (dict): A dict indicates the possible associate name of each label
=======
        inputs (list): A list of paths where each path points to a sample image.
        labels (list): A list of `int` where each is the label of the sample at
            the same index.
        label_names (dict): A dict indicates the possible associate name of each label
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
            value.

    """

    # We predefine to follow enforcement, actual initialization in init
<<<<<<< HEAD
    input_list = None
    label_list = None
    label_name = None
=======
    inputs = None
    labels = None
    label_names = None
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18

    def __init__(
        self,
        save_dir_path=None,
    ):
<<<<<<< HEAD
        label_code_list = [
=======
        label_names = [
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
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
        # bring outside to prevent case where download fail
        if not os.path.exists(save_dir_path):
            raise ValueError("Dataset does not exist at `%s`" % save_dir_path)

        # What will happen if downloaded data get corrupted?
<<<<<<< HEAD
        label_name = {}
        all_path_list = []
        for label_id, label_code in enumerate(label_code_list):
            path_list = grab_files_from_dir(
                f"{save_dir_path}/{label_code}/", file_types="*.tif"
            )
            path_list = [[v, label_id] for v in path_list]
            path_list.sort()
            all_path_list.extend(path_list)
            label_name[label_id] = label_code
        input_list, label_list = list(zip(*all_path_list))

        self.label_name = label_name
        self.input_list = list(input_list)  # type casting to list
        self.label_list = list(label_list)  # type casting to list
=======
        uid_name_map = {}
        all_paths = []
        for label_id, label_name in enumerate(label_names):
            paths = grab_files_from_dir(
                f"{save_dir_path}/{label_name}/", file_types="*.tif"
            )
            paths = [[v, label_id] for v in paths]
            paths.sort()
            all_paths.extend(paths)
            uid_name_map[label_id] = label_name
        inputs, labels = list(zip(*all_paths))

        self.label_names = uid_name_map
        self.inputs = list(inputs)  # type casting to list
        self.labels = list(labels)  # type casting to list
>>>>>>> 26d0a2006ca0fee58bb4b5901592a52aa2e2ae18
