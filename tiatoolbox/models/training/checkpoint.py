"""Checkpoint utilities for model training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # pragma: no cover
    from torch import nn


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return underlying module when wrapped with DataParallel/DDP."""
    if hasattr(model, "module"):
        return model.module
    return model


def extract_model_state_dict(model: nn.Module) -> dict[str, Any]:
    """Extract a model state dict in inference-compatible format."""
    return _unwrap_model(model).state_dict()


def save_model_weights(model: nn.Module, output_path: str | Path) -> None:
    """Save model weights that can be consumed by TIAToolbox inference engines."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(extract_model_state_dict(model), output_path)


def save_checkpoint(state: dict[str, Any], output_path: str | Path) -> None:
    """Save full trainer state to a checkpoint file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint file from disk."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        msg = f"Checkpoint file does not exist: `{checkpoint_path}`."
        raise FileNotFoundError(msg)
    return torch.load(checkpoint_path, map_location=map_location)
