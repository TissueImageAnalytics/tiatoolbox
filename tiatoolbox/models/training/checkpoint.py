"""Checkpoint utilities for model training."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

import torch

from tiatoolbox.models.models_abc import unwrap_model

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator
    from typing import BinaryIO

    from torch import nn


def extract_model_state_dict(model: nn.Module) -> dict[str, Any]:
    """Extract the state dict of the model inside any parallel/compile wrappers."""
    return unwrap_model(model).state_dict()


def save_model_weights(model: nn.Module, output_path: str | Path) -> None:
    """Save model weights in the training export format."""
    save_checkpoint(extract_model_state_dict(model), output_path)


@contextmanager
def _atomic_output(output_path: str | Path) -> Iterator[BinaryIO]:
    """Replace a completed file atomically, keeping the previous file on failure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_checkpoint(state: dict[str, Any], output_path: str | Path) -> None:
    """Save full trainer state to a checkpoint file."""
    with _atomic_output(output_path) as stream:
        torch.save(state, stream)


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint file from disk."""
    checkpoint_path = Path(checkpoint_path)
    return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
