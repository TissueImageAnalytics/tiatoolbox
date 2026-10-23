"""Sampling utilities for training datasets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler

from tiatoolbox.annotation import AnnotationStore, SQLiteStore
from tiatoolbox.tools.patchextraction import get_patch_extractor
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator, Sequence

    from tiatoolbox.type_hints import Resolution, Units


def _normalize_xy_shape(
    shape: int | tuple[int, int],
    argument_name: str,
) -> tuple[int, int]:
    """Normalize scalar or pair shape arguments into `(width, height)`."""
    if isinstance(shape, int):
        if shape <= 0:
            msg = f"`{argument_name}` must be a positive integer."
            raise ValueError(msg)
        return (shape, shape)

    if len(shape) != 2:  # noqa: PLR2004
        msg = f"`{argument_name}` must have length 2."
        raise ValueError(msg)

    width, height = int(shape[0]), int(shape[1])
    if width <= 0 or height <= 0:
        msg = f"`{argument_name}` dimensions must be positive."
        raise ValueError(msg)
    return (width, height)


def generate_slide_patch_coordinates(
    input_img: str | Path | np.ndarray | WSIReader,
    patch_size: int | tuple[int, int],
    *,
    stride: int | tuple[int, int] | None = None,
    resolution: Resolution = 0,
    units: Units = "level",
    within_bound: bool = False,
    input_mask: (
        str | Path | np.ndarray | VirtualWSIReader | AnnotationStore | None
    ) = None,
    min_mask_ratio: float = 0.0,
    store_filter: str | None = None,
) -> np.ndarray:
    """Generate patch bounds for one slide using TIAToolbox patch extraction."""
    patch_size_xy = _normalize_xy_shape(patch_size, "patch_size")
    stride_xy = (
        _normalize_xy_shape(stride, "stride") if stride is not None else patch_size_xy
    )

    owned_mask = None
    owned_reader = None
    if isinstance(input_mask, (str, Path)) and Path(input_mask).suffix.lower() in {
        ".db",
        ".sqlite",
        ".sqlite3",
    }:
        if not Path(input_mask).is_file():
            msg = f"Annotation mask store does not exist: {input_mask}"
            raise ValueError(msg)
        owned_mask = SQLiteStore(input_mask)
        input_mask = owned_mask
    try:
        if not isinstance(input_img, WSIReader):
            owned_reader = WSIReader.open(input_img)
            input_img = owned_reader
        extractor = get_patch_extractor(
            "slidingwindow",
            input_img=input_img,
            patch_size=patch_size_xy,
            resolution=resolution,
            units=units,
            stride=stride_xy,
            within_bound=within_bound,
            input_mask=input_mask,
            min_mask_ratio=min_mask_ratio,
            store_filter=store_filter,
        )
    finally:
        if owned_mask is not None:
            owned_mask.close()
        if owned_reader is not None:
            owned_reader.close()
    coordinates = np.asarray(extractor.coordinate_list, dtype=np.int64)

    if coordinates.ndim != 2 or coordinates.shape[-1] != 4:  # noqa: PLR2004
        msg = "Generated coordinates must have shape `(N, 4)`."
        raise ValueError(msg)

    return coordinates


def _normalize_labels(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Normalize sampler labels to a 1D int64 numpy array."""
    if isinstance(labels, torch.Tensor):
        labels_array = labels.detach().cpu().numpy()
    elif isinstance(labels, np.ndarray):
        labels_array = labels
    else:
        labels_array = np.asarray(labels)

    if labels_array.ndim != 1:
        msg = "`labels` must be one-dimensional."
        raise ValueError(msg)
    if labels_array.size == 0:
        msg = "`labels` must contain at least one element."
        raise ValueError(msg)

    return labels_array.astype(np.int64, copy=False)


class ClassBalancedIndexSampler(WeightedRandomSampler):
    """Index sampler that inversely weights samples by class frequency."""

    def __init__(
        self: ClassBalancedIndexSampler,
        labels: Sequence[int] | np.ndarray | torch.Tensor,
        *,
        num_samples: int | None = None,
        replacement: bool = True,
        ignore_labels: set[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        """Initialize :class:`ClassBalancedIndexSampler`."""
        labels_array = _normalize_labels(labels)

        ignored = set(ignore_labels or set())
        eligible_mask = np.array(
            [int(label) not in ignored for label in labels_array],
            dtype=bool,
        )
        if not np.any(eligible_mask):
            msg = "No samples remain after applying `ignore_labels`."
            raise ValueError(msg)

        eligible_labels = labels_array[eligible_mask]
        unique_labels, counts = np.unique(eligible_labels, return_counts=True)
        label_weights = {
            int(label): 1.0 / float(count)
            for label, count in zip(unique_labels, counts, strict=False)
        }

        sample_weights = np.zeros_like(labels_array, dtype=np.float64)
        for index, label in enumerate(labels_array):
            if int(label) in label_weights:
                sample_weights[index] = label_weights[int(label)]

        default_num_samples = int(np.sum(eligible_mask))
        resolved_num_samples = (
            int(num_samples) if num_samples is not None else default_num_samples
        )
        if resolved_num_samples <= 0:
            msg = "`num_samples` must be a positive integer."
            raise ValueError(msg)
        if not replacement and resolved_num_samples > default_num_samples:
            msg = (
                "`num_samples` cannot exceed the number of non-ignored samples "
                "when `replacement=False`."
            )
            raise ValueError(msg)

        super().__init__(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=resolved_num_samples,
            replacement=bool(replacement),
            generator=generator,
        )


class GroupedBlockSampler(Sampler[int]):
    """Visit samples in shuffled blocks that each come from one group.

    Each group's samples, for example the patches of one slide, are shuffled
    and cut into blocks of ``block_size``; the blocks are then shuffled. Every
    sample is visited once per epoch in random order, but consecutive indices
    share a group, so a batch of ``B`` samples touches about
    ``B / block_size + 1`` groups and per-worker reader/store caches stay warm
    when there are many slides. Pass one group ID per sample, such as
    ``dataset.sample_slide_indices`` (indexed by the subset's indices when
    sampling from a ``Subset``).
    """

    def __init__(
        self: GroupedBlockSampler,
        groups: Sequence[int] | np.ndarray | torch.Tensor,
        block_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        """Initialize :class:`GroupedBlockSampler`."""
        self.groups = _normalize_labels(groups)
        if block_size <= 0:
            msg = "`block_size` must be a positive integer."
            raise ValueError(msg)
        self.block_size = int(block_size)
        self.generator = generator

    def __len__(self: GroupedBlockSampler) -> int:
        """Return the number of samples visited per epoch."""
        return len(self.groups)

    def __iter__(self: GroupedBlockSampler) -> Iterator[int]:
        """Yield every index once, in shuffled single-group blocks."""
        generator = self.generator
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(self.groups), generator=generator).numpy()
        # A stable sort by group keeps each group's members in shuffled order.
        by_group = order[np.argsort(self.groups[order], kind="stable")]
        group_starts = np.flatnonzero(np.diff(self.groups[by_group])) + 1
        blocks = [
            members[start : start + self.block_size]
            for members in np.split(by_group, group_starts)
            for start in range(0, len(members), self.block_size)
        ]
        for block in torch.randperm(len(blocks), generator=generator).tolist():
            yield from blocks[block].tolist()
