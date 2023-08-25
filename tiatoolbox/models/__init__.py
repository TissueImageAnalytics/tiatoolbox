"""Models package for the models implemented in tiatoolbox."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

__all__ = [
    "architecture",
    "dataset",
    "engine",
    "models_abc",
    "HoVerNet",
    "HoVerNetPlus",
    "IDaRS",
    "MapDe",
    "MicroNet",
    "NuClick",
    "SCCNN",
    "MultiTaskSegmentor",
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
    "load_torch_model",
]


def load_torch_model(model: torch.nn.Module, weights: str | Path) -> torch.nn.Module:
    """Helper function to load a torch model.

    Args:
        model (torch.nn.Module):
            A torch model.
        weights (str or Path):
            Path to pretrained weights.

    Returns:
        torch.nn.Module:
            Torch model with pretrained weights loaded on CPU.

    """
    # ! assume to be saved in single GPU mode
    # always load on to the CPU
    saved_state_dict = torch.load(weights, map_location="cpu")
    return model.load_state_dict(saved_state_dict, strict=True)
