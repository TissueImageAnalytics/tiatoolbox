"""Utility helpers for TIAToolbox training workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence


def stratified_split_indices(
    targets: Sequence[int],
    val_fraction: float = 0.2,
    seed: int = 123,
) -> tuple[list[int], list[int]]:
    """Create stratified train/validation index splits from class targets.

    Args:
        targets (Sequence[int]):
            Per-sample class targets.
        val_fraction (float):
            Fraction of samples to place in the validation split.
        seed (int):
            Random seed used for the split.

    Returns:
        tuple[list[int], list[int]]:
            Training indices followed by validation indices.

    Raises:
        ValueError:
            If no targets are provided or `val_fraction` is outside `(0, 1)`.

    """
    if len(targets) == 0:
        msg = "`targets` must contain at least one item."
        raise ValueError(msg)

    if not 0 < val_fraction < 1:
        msg = "`val_fraction` must be between 0 and 1."
        raise ValueError(msg)

    indices = np.arange(len(targets))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=np.asarray(targets),
    )

    return train_indices.tolist(), val_indices.tolist()
