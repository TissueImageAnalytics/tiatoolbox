"""Define classes and methods for dataset information."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from tiatoolbox import rcParam
from tiatoolbox.utils import download_data, unzip_data
from tiatoolbox.utils.misc import grab_files_from_dir
from huggingface_hub import hf_hub_download


class DatasetInfoABC(ABC):
    """Define an abstract class for holding dataset information.

    Enforcing such that following attributes must always be defined by
    the subclass.

    Property:
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
    def inputs(self: DatasetInfoABC) -> None:
        """A list of paths where each path points to a sample image."""
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self: DatasetInfoABC) -> None:
        """A list of labels where each is the label of the sample at the same index."""
        raise NotImplementedError

    @property
    @abstractmethod
    def label_names(self: DatasetInfoABC) -> None:
        """A dict indicates the possible associate name of each label value."""
        raise NotImplementedError


class KatherPatchDataset(DatasetInfoABC):
    """Define a class for holding the Kather dataset information.

    Args:
        save_dir_path (str, Path, or None):
            Path to the directory containing the Kather dataset with 
            label subdirectories (e.g., 'BACK/', 'NORM/', 'TUM/', etc.).
            If `None`, the dataset will be automatically downloaded from
            HuggingFace Hub and extracted to 
            '~/.tiatoolbox/dataset/kather100k-validation/'. The directory
            structure should contain subdirectories for each tissue class,
            with .tif image files inside each subdirectory.

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
        self: KatherPatchDataset,
        save_dir_path: Path | None = None,
    ) -> None:
        """Initialize :class:`KatherPatchDataset`."""
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
            download_dir = rcParam["TIATOOLBOX_HOME"] / "dataset"
            if not Path.exists(download_dir / "kather100k-validation"):
                save_zip_path = hf_hub_download(
                    repo_id="TIACentre/TIAToolBox_Remote_Samples",
                    filename="kather100k-train-nonorm-subset-20k.zip",
                    subfolder="datasets",
                    repo_type="dataset",
                    local_dir=download_dir,
                )
                unzip_data(Path(save_zip_path), download_dir)
            save_dir_path = download_dir / "kather100k-validation"

        # bring outside to prevent case where download fail
        dataset_path = Path(save_dir_path)
        if not dataset_path.exists():
            msg = f"Dataset does not exist at `{dataset_path}`"
            raise ValueError(msg)

        # What will happen if downloaded data get corrupted?
        uid_name_map = {}
        all_paths = []
        for label_id, label_name in enumerate(label_names):
            paths = grab_files_from_dir(
                f"{dataset_path}/{label_name}/",
                file_types="*.tif",
            )
            paths = [[v, label_id] for v in paths]
            paths.sort()
            all_paths.extend(paths)
            uid_name_map[label_id] = label_name
        inputs, labels = list(zip(*all_paths, strict=False))

        self.label_names = uid_name_map
        self.inputs = list(inputs)  # type casting to list
        self.labels = list(labels)  # type casting to list
