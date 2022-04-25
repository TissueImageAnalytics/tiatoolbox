import os
from abc import ABC, abstractmethod
from pathlib import Path

from tiatoolbox import rcParam
from tiatoolbox.utils.misc import download_data, grab_files_from_dir, unzip_data


class DatasetInfoABC(ABC):
    """Define an abstract class for holding dataset information.

    Enforcing such that following attributes must always be defined by
    the subclass.

    Attributes:
        inputs (list):
            A list of paths where each path points to a sample image.
            labels (list): A list of `int` where each is the label of
            the sample at the same index.
        label_names (dict):
            A dict indicates the possible associate name of each label
            value.

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
        save_dir_path (str or None):
            Path to directory containing the Kather dataset. This is
            assumed to be the same form after the data is initially
            downloaded. If the argument is `None`, the dataset will be
            downloaded and extracted into the 'run_dir/download/Kather'.

    Attributes:
        inputs (list):
            A list of paths where each path points to a sample image.
        labels (list):
            A list of `int` where each value corresponds to the label of
            the sample at the same index.
        label_names (dict):
            A dict mapping each unique label value to the associated
            class name as a string.

    """

    # We pre-define to follow enforcement, actual initialization in init
    inputs = None
    labels = None
    label_names = None

    def __init__(
        self,
        save_dir_path=None,
    ):
        label_names = [
            "BACK",
            "NORM",
            "DEB",
            "TUM",
            "ADI",
            "MUC",
            "MUS",
            "STR",
            "LYM",
        ]

        if save_dir_path is None:  # pragma: no cover
            save_dir_path = Path(rcParam["TIATOOLBOX_HOME"], "dataset")
            if not os.path.exists(save_dir_path):
                save_zip_path = os.path.join(save_dir_path, "Kather.zip")
                url = (
                    "https://tiatoolbox.dcs.warwick.ac.uk/datasets"
                    "/kather100k-train-nonorm-subset-20k.zip"
                )
                download_data(url, save_zip_path)
                unzip_data(save_zip_path, save_dir_path)
            save_dir_path = Path(save_dir_path, "kather100k-validation")
        # bring outside to prevent case where download fail
        save_dir_path = Path(save_dir_path)
        if not save_dir_path.exists():
            raise ValueError(f"Dataset does not exist at `{save_dir_path}`")

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
