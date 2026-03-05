"""Models package for the models implemented in tiatoolbox."""

from __future__ import annotations

from . import architecture, dataset, engine, models_abc
from .architecture.hovernet import HoVerNet
from .architecture.hovernetplus import HoVerNetPlus
from .architecture.idars import IDaRS
from .architecture.mapde import MapDe
from .architecture.micronet import MicroNet
from .architecture.nuclick import NuClick
from .architecture.sam import SAM
from .architecture.sccnn import SCCNN
from .dataset import PatchDataset, WSIPatchDataset
from .engine.deep_feature_extractor import DeepFeatureExtractor
from .engine.io_config import (
    IOInstanceSegmentorConfig,
    IOPatchPredictorConfig,
    IOSegmentorConfig,
    ModelIOConfigABC,
)
from .engine.multi_task_segmentor import MultiTaskSegmentor
from .engine.nucleus_detector import NucleusDetector
from .engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from .engine.patch_predictor import PatchPredictor
from .engine.prompt_segmentor import PromptSegmentor
from .engine.semantic_segmentor import SemanticSegmentor

__all__ = [
    "SAM",
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
    "NucleusDetector",
    "NucleusInstanceSegmentor",
    "PatchDataset",
    "PatchPredictor",
    "PromptSegmentor",
    "SemanticSegmentor",
    "WSIPatchDataset",
    "architecture",
    "dataset",
    "engine",
    "models_abc",
]
