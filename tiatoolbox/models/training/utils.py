"""Index-based train/validation splitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

if TYPE_CHECKING:
    from collections.abc import Sequence


def split_indices(
    num_samples: int,
    val_fraction: float = 0.2,
    *,
    stratify: Sequence | None = None,
    groups: Sequence | None = None,
    seed: int = 123,
) -> tuple[list[int], list[int]]:
    """Split sample indices without binding transforms or constructing datasets.

    Args:
        num_samples (int): Number of samples in the source dataset.
        val_fraction (float): Validation fraction of samples, or of unique groups
            when ``groups`` is supplied. Unequal groups give unequal sample counts.
        stratify (Sequence or None): One class label per sample for stratification.
        groups (Sequence or None): One patient/slide identifier per sample. Each
            group belongs entirely to one split. Grouping and stratification cannot
            be combined here; use a scikit-learn splitter for more complex policies.
        seed (int): Seed used only for the split.

    Returns:
        tuple[list[int], list[int]]: Disjoint train and validation indices covering
            all samples. Apply them to separate dataset instances with matching
            sample order and class mappings to keep validation transforms separate.

    Raises:
        ValueError: Invalid sizes, labels, or incompatible split settings.
    """
    if num_samples <= 1:
        msg = "`num_samples` must be at least two."
        raise ValueError(msg)
    if not 0 < val_fraction < 1:
        msg = "`val_fraction` must be between 0 and 1."
        raise ValueError(msg)
    if stratify is not None and groups is not None:
        msg = "Choose either `stratify` or `groups`, or use a custom index splitter."
        raise ValueError(msg)
    for name, values in (("stratify", stratify), ("groups", groups)):
        if values is not None and (np.ndim(values) != 1 or len(values) != num_samples):
            msg = f"`{name}` must contain one value per sample."
            raise ValueError(msg)
    indices = np.arange(num_samples)
    if groups is not None:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=seed
        )
        train, val = next(splitter.split(indices, groups=groups))
    else:
        train, val = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=seed,
            stratify=stratify,
        )
    return train.tolist(), val.tolist()
