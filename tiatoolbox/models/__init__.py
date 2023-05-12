"""Models package for the models implemented in tiatoolbox."""
from tiatoolbox.models import architecture, dataset, engine, models_abc
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

from .architecture.hovernet import HoVerNet
from .architecture.hovernetplus import HoVerNetPlus
from .architecture.idars import IDaRS
from .architecture.mapde import MapDe
from .architecture.micronet import MicroNet
from .architecture.nuclick import NuClick
from .architecture.sccnn import SCCNN
from .engine.multi_task_segmentor import MultiTaskSegmentor
from .engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from .engine.patch_predictor import PatchPredictor
from .engine.semantic_segmentor import SemanticSegmentor

HoVerNet = HoVerNet
HoVerNetPlus = HoVerNetPlus
IDaRS = IDaRS
MapDe = MapDe
MicroNet = MicroNet
NuClick = NuClick
SCCNN = SCCNN

MultiTaskSegmentor = MultiTaskSegmentor
NucleusInstanceSegmentor = NucleusInstanceSegmentor
PatchPredictor = PatchPredictor
SemanticSegmentor = SemanticSegmentor
