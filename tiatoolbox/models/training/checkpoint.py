"""Checkpoint utilities for model training."""

from __future__ import annotations

from collections.abc import Mapping
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
    """Extract the unwrapped module state dict."""
    return _unwrap_model(model).state_dict()


def _iter_state_dict_candidates(
    payload: Any,
) -> list[Mapping[str, Any] | Any]:
    """Yield plausible state-dict payloads from a loaded checkpoint object."""
    candidates: list[Mapping[str, Any] | Any] = []

    if isinstance(payload, Mapping):
        extracted_state_dict = payload.get("model_state_dict")
        if isinstance(extracted_state_dict, Mapping):
            candidates.append(extracted_state_dict)
            if "model" not in extracted_state_dict:
                candidates.append({"model": extracted_state_dict})

        candidates.append(payload)
        if "model" not in payload and "model_state_dict" not in payload:
            candidates.append({"model": payload})
        return candidates

    return [payload]


def load_model_state_dict(
    model: nn.Module,
    payload: Any,
    *,
    strict: bool = True,
) -> Any:
    """Load model weights from a bare state dict or a trainer checkpoint payload."""
    base_model = _unwrap_model(model)
    first_error: Exception | None = None

    for candidate in _iter_state_dict_candidates(payload):
        try:
            return base_model.load_state_dict(candidate, strict=strict)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            if first_error is None:
                first_error = error

    if first_error is None:
        msg = "Unable to resolve a compatible state dict payload."
        raise RuntimeError(msg)
    raise first_error


def save_model_weights(model: nn.Module, output_path: str | Path) -> None:
    """Save model weights in the training export format."""
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
