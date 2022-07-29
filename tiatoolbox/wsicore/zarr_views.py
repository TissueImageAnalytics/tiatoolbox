from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np
import zarr
from scipy import ndimage


class AffineZarrView:
    """A view wrapper for a zarr array that applies an affine transform."""

    def __init__(
        self, array: zarr.core.Array, transform: np.ndarray, transform_kwargs=None
    ):
        self._is_view = True
        self.array = array
        self.transform = transform
        self.inv_transform = np.linalg.inv(transform)
        self.transform_kwargs = transform_kwargs or {"order": 1}

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def ndim(self) -> int:
        return self.array.ndim - 1

    @property
    def size(self) -> int:
        return self.array.size

    def __len__(self) -> int:
        return self.array.shape[-1]

    def __repr__(self) -> str:
        return f"AffineZarrView({self.array})"

    def __getitem__(self, index: Tuple[slice, slice]) -> np.ndarray:
        is_tuple = isinstance(index, tuple)
        all_slices = all(isinstance(x, slice) for x in index)
        if not (is_tuple and all_slices):
            raise TypeError(f"Index must be a tuple of slices, not {type(index)}")

        # Extend index slices to match array dimensions
        index = [
            s
            for s, _ in itertools.zip_longest(
                index, self.array.shape, fillvalue=slice(None)
            )
        ]

        # Convert slices with None to indices
        indices = [s.indices(l) for s, l in zip(index, self.array.shape)]

        # Read the array with affine transform applied
        spatial_indices = indices[:-1]
        channels_indices = indices[-1]
        out_shape = np.array(
            [stop - start for start, stop, _ in spatial_indices], dtype=int
        )
        return np.dstack(
            (
                ndimage.affine_transform(
                    self.array[:, :, i],
                    self.transform,
                    offset=[start for start, _, _ in spatial_indices],
                    output_shape=out_shape,
                    **self.transform_kwargs,
                )
                for i in range(*channels_indices)
            )
        )
