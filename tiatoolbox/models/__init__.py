"""Models package for the models implemented in tiatoolbox."""

from __future__ import annotations

from . import architecture, dataset, engine, models_abc
from .architecture.hovernet import HoVerNet
from .architecture.hovernetplus import HoVerNetPlus
from .architecture.idars import IDaRS
from .architecture.mapde import MapDe
from .architecture.micronet import MicroNet
from .architecture.nuclick import NuClick
from .architecture.sccnn import SCCNN
from .dataset import PatchDataset, WSIPatchDataset, WSIStreamDataset
from .engine.io_config import (
    IOInstanceSegmentorConfig,
    IOPatchPredictorConfig,
    IOSegmentorConfig,
    ModelIOConfigABC,
)
from .engine.multi_task_segmentor import MultiTaskSegmentor
from .engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from .engine.patch_predictor import PatchPredictor
from .engine.semantic_segmentor import DeepFeatureExtractor, SemanticSegmentor

__all__ = [
    "architecture",
    "dataset",
    "engine",
    "models_abc",
    "SCCNN",
    "HoVerNet",
    "HoVerNetPlus",
    "IDaRS",
    "MapDe",
    "MicroNet",
    "MultiTaskSegmentor",
    "NuClick",
    "NucleusInstanceSegmentor",
    "PatchPredictor",
    "SemanticSegmentor",
    "IOPatchPredictorConfig",
    "IOSegmentorConfig",
    "IOInstanceSegmentorConfig",
    "ModelIOConfigABC",
    "DeepFeatureExtractor",
    "WSIStreamDataset",
    "WSIPatchDataset",
    "PatchDataset",
]
