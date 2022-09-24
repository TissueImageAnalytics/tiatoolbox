"""Models package for the models implemented in tiatoolbox."""
from pathlib import Path

from tiatoolbox import _lazy_import

location = Path(__file__).parent

architecture = _lazy_import("architecture", location)
dataset = _lazy_import("dataset", location)
engine = _lazy_import("engine", location)
models_abc = _lazy_import("models_abc", location)

from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.models.engine.patch_predictor import (
    IOPatchPredictorConfig,
    PatchDataset,
    PatchPredictor,
    WSIPatchDataset,
)
from tiatoolbox.models.engine.semantic_segmentor import (
    DeepFeatureExtractor,
    IOSegmentorConfig,
    SemanticSegmentor,
    WSIStreamDataset,
)
