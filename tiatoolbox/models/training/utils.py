"""Utility helpers for TIAToolbox training workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from torch.utils.data import Dataset


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


def _dataset_targets(dataset: Dataset[Any]) -> Sequence[int]:
    """Return conventional class targets from a dataset, if available."""
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return targets

    labels = getattr(dataset, "labels", None)
    if labels is not None:
        return labels

    samples = getattr(dataset, "samples", None)
    if samples is not None:
        return [target for _, target in samples]

    msg = (
        "`stratify=True` requires a dataset with `targets`, `labels`, or "
        "`samples` attributes. Pass an explicit target sequence instead."
    )
    raise ValueError(msg)


def split_dataset(
    dataset: Dataset[Any],
    val_fraction: float = 0.2,
    *,
    stratify: bool | Sequence[int] | None = None,
    seed: int = 123,
) -> tuple[Subset[Any], Subset[Any]]:
    """Split a dataset into train and validation subsets.

    Args:
        dataset (Dataset[Any]):
            Dataset to split.
        val_fraction (float):
            Fraction of samples to place in the validation split.
        stratify (bool | Sequence[int] | None):
            If ``True``, read conventional class targets from ``dataset`` and use a
            stratified split. If a target sequence is provided, use it directly.
            If ``None`` or ``False``, perform a shuffled random split.
        seed (int):
            Random seed used for the split.

    Returns:
        tuple[Subset[Any], Subset[Any]]:
            Training subset followed by validation subset.

    Raises:
        ValueError:
            If the dataset is empty, ``val_fraction`` is outside ``(0, 1)``, or
            explicit stratification targets do not match the dataset length.

    """
    if len(dataset) == 0:
        msg = "`dataset` must contain at least one item."
        raise ValueError(msg)

    if not 0 < val_fraction < 1:
        msg = "`val_fraction` must be between 0 and 1."
        raise ValueError(msg)

    indices = np.arange(len(dataset))
    if stratify is None or stratify is False:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=seed,
            shuffle=True,
        )
    else:
        targets = _dataset_targets(dataset) if stratify is True else stratify
        if len(targets) != len(dataset):
            msg = "`stratify` targets must match the dataset length."
            raise ValueError(msg)
        train_indices, val_indices = stratified_split_indices(
            targets,
            val_fraction=val_fraction,
            seed=seed,
        )

    return Subset(dataset, list(train_indices)), Subset(dataset, list(val_indices))


def make_dataloaders(
    train_dataset: Dataset[Any],
    val_dataset: Dataset[Any] | None = None,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    train_shuffle: bool = True,
    val_shuffle: bool = False,
    **dataloader_kwargs: object,
) -> tuple[DataLoader[Any], DataLoader[Any] | None]:
    """Create train and validation dataloaders from explicit datasets.

    Args:
        train_dataset (Dataset[Any]):
            Training dataset.
        val_dataset (Dataset[Any] | None):
            Optional validation dataset.
        batch_size (int):
            Batch size for both dataloaders.
        num_workers (int):
            Number of worker processes for both dataloaders.
        train_shuffle (bool):
            Whether to shuffle the training dataloader.
        val_shuffle (bool):
            Whether to shuffle the validation dataloader.
        **dataloader_kwargs (Any):
            Additional keyword arguments passed to both ``DataLoader`` instances.

    Returns:
        tuple[DataLoader[Any], DataLoader[Any] | None]:
            Training dataloader followed by optional validation dataloader.

    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        **dataloader_kwargs,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=val_shuffle,
            num_workers=num_workers,
            **dataloader_kwargs,
        )
    return train_loader, val_loader
