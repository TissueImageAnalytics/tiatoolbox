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


class DatasetInfoABC(ABC):
    """Define an abstract class for holding dataset information.

    Enforcing such that following attributes must always be defined by the subclass.

    Attributes:
        inputs (list): A list of paths where each path points to a sample image.
        labels (list): A list of `int` where each is the label of the sample at
          the same index.
        label_names (dict): A dict indicates the possible associate name of each
          label value.

    """

    @property
    @abstractmethod
    def inputs(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def label_names(self):
        raise NotImplementedError


class KatherPatchDataset(DatasetInfoABC):
    """Define a class for holding the Kather dataset information.

    Args:
        save_dir_path (str or None): Path to directory containing the Kather
          dataset. This is assumed to be the same form after the data is initially
          downloaded. If the argument is `None`, the dataset will be downloaded
          and extracted into the 'run_dir/download/Kather'.
    Attributes
        inputs (list): A list of paths where each path points to a sample image.
        labels (list): A list of `int` where each is the label of the sample at
          the same index.
        label_names (dict): A dict indicates the possible associate name of each
          label value.

    """

    # We predefine to follow enforcement, actual initialization in init
    inputs = None
    labels = None
    label_names = None

    def __init__(
        self,
        save_dir_path=None,
    ):
        label_names = [
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
