"""Contains dataset functionality for use with models in tiatoolbox."""

from tiatoolbox.models.dataset.abc import PatchDatasetABC
from tiatoolbox.models.dataset.classification import (
    PatchDataset,
    WSIPatchDataset,
    predefined_preproc_func,
)
from tiatoolbox.models.dataset.interactive_segmentation import InteractiveSegmentorDataset
from tiatoolbox.models.dataset.info import DatasetInfoABC, KatherPatchDataset
