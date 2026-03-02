"""Models package for the models implemented in tiatoolbox."""

from __future__ import annotations

from . import architecture, dataset, engines, models_abc
from .architecture.hovernet import HoVerNet
from .architecture.hovernetplus import HoVerNetPlus
from .architecture.idars import IDaRS
from .architecture.mapde import MapDe
from .architecture.micronet import MicroNet
from .architecture.nuclick import NuClick
from .architecture.sccnn import SCCNN
from .dataset import PatchDataset, WSIPatchDataset
from .engines.deep_feature_extractor import DeepFeatureExtractor
from .engines.io_config import (
    IOInstanceSegmentorConfig,
    IOPatchPredictorConfig,
    IOSegmentorConfig,
    ModelIOConfigABC,
)
from .engines.multi_task_segmentor import MultiTaskSegmentor
from .engines.nucleus_detector import NucleusDetector
from .engines.nucleus_instance_segmentor import NucleusInstanceSegmentor
from .engines.patch_predictor import PatchPredictor
from .engines.semantic_segmentor import SemanticSegmentor

__all__ = [
    "SCCNN",
    "DeepFeatureExtractor",
    "HoVerNet",
    "HoVerNetPlus",
    "IDaRS",
    "IOInstanceSegmentorConfig",
    "IOPatchPredictorConfig",
    "IOSegmentorConfig",
    "MapDe",
    "MicroNet",
    "ModelIOConfigABC",
    "MultiTaskSegmentor",
    "NuClick",
    "NucleusInstanceSegmentor",
    "PatchDataset",
    "PatchPredictor",
    "SemanticSegmentor",
    "WSIPatchDataset",
    "architecture",
    "dataset",
    "engine",
    "models_abc",
]
