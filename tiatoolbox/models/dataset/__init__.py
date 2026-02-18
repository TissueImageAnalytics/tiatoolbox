"""Contains dataset functionality for use with models in tiatoolbox."""

from tiatoolbox.models.dataset.classification import predefined_preproc_func

from .dataset_abc import (
    PatchDataset,
    PatchDatasetABC,
    WSIPatchDataset,
)
from .info import DatasetInfoABC, KatherPatchDataset

__all__ = [
    "DatasetInfoABC",
    "KatherPatchDataset",
    "PatchDataset",
    "PatchDatasetABC",
    "WSIPatchDataset",
    "predefined_preproc_func",
]
