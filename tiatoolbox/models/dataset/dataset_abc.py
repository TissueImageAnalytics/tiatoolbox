"""Define dataset abstract classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    try:
        from typing import TypeGuard
    except ImportError:
        from typing_extensions import TypeGuard  # to support python <3.10


import numpy as np
import torch

from tiatoolbox.utils import imread

input_type = Union[list[Union[str, Path, np.ndarray]], np.ndarray]


class PatchDatasetABC(ABC, torch.utils.data.Dataset):
    """Define abstract base class for patch dataset."""

    inputs: input_type
    labels: list[int] | np.ndarray

    def __init__(
        self: PatchDatasetABC,
    ) -> None:
        """Initialize :class:`PatchDatasetABC`."""
        super().__init__()
        self._preproc = self.preproc
        self.data_is_npy_alike = False
        self.inputs = []
        self.labels = []

    @staticmethod
    def _check_shape_integrity(shapes: list | np.ndarray) -> None:
        """Checks the integrity of input shapes.

        Args:
            shapes (list or np.ndarray):
                input shape to check.

        Raises:
            ValueError: If the shape is not valid.

        """
        if any(len(v) != 3 for v in shapes):  # noqa: PLR2004
            msg = "Each sample must be an array of the form HWC."
            raise ValueError(msg)

        max_shape = np.max(shapes, axis=0)
        if (shapes - max_shape[None]).sum() != 0:
            msg = "Images must have the same dimensions."
            raise ValueError(msg)

    @staticmethod
    def _are_paths(inputs: input_type) -> TypeGuard[Iterable[Path]]:
        """TypeGuard to check that input array contains only paths."""
        return all(isinstance(v, (Path, str)) for v in inputs)

    @staticmethod
    def _are_npy_like(inputs: input_type) -> TypeGuard[Iterable[np.ndarray]]:
        """TypeGuard to check that input array contains only np.ndarray."""
        return all(isinstance(v, np.ndarray) for v in inputs)

    def _check_input_integrity(self: PatchDatasetABC, mode: str) -> None:
        """Check that variables received during init are valid.

        These checks include:
            - Input is of a singular data type, such as a list of paths.
            - If it is list of images, all images are of the same height
              and width.

        """
        if mode == "patch":
            self.data_is_npy_alike = False

            msg = (
                "Input must be either a list/array of images "
                "or a list of valid image paths."
            )

            # When a list of paths is provided
            if self._are_paths(self.inputs):
                if any(not Path(v).exists() for v in self.inputs):
                    # at least one of the paths are invalid
                    raise ValueError(
                        msg,
                    )
                # Preload test for sanity check
                shapes = [self.load_img(v).shape for v in self.inputs]
                self.data_is_npy_alike = False

            elif self._are_npy_like(self.inputs):
                shapes = [v.shape for v in self.inputs]
                self.data_is_npy_alike = True

            else:
                raise ValueError(msg)

            self._check_shape_integrity(shapes)

            # If input is a numpy array
            if isinstance(self.inputs, np.ndarray):
                # Check that input array is numerical
                if not np.issubdtype(self.inputs.dtype, np.number):
                    # ndarray of mixed data types
                    msg = "Provided input array is non-numerical."
                    raise ValueError(msg)
                self.data_is_npy_alike = True

        elif not isinstance(self.inputs, (list, np.ndarray)):
            msg = "`inputs` should be a list of patch coordinates."
            raise ValueError(msg)

    @staticmethod
    def load_img(path: str | Path) -> np.ndarray:
        """Load an image from a provided path.

        Args:
            path (str or Path): Path to an image file.

        Returns:
            :class:`numpy.ndarray`:
                Image as a numpy array.

        """
        path = Path(path)

        if path.suffix not in (".npy", ".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            msg = f"Cannot load image data from `{path.suffix}` files."
            raise ValueError(msg)

        return imread(path, as_uint8=False)

    @staticmethod
    def preproc(image: np.ndarray) -> np.ndarray:
        """Define the pre-processing of this class of loader."""
        return image

    @property
    def preproc_func(self: PatchDatasetABC) -> Callable:
        """Return the current pre-processing function of this instance.

        The returned function is expected to behave as follows:
        >>> transformed_img = func(img)

        """
        return self._preproc

    @preproc_func.setter
    def preproc_func(self: PatchDatasetABC, func: Callable) -> None:
        """Set the pre-processing function for this instance.

        If `func=None`, the method will default to `self.preproc`.
        Otherwise, `func` is expected to be callable and behaves as
        follows:

        >>> transformed_img = func(img)

        """
        if func is None:
            self._preproc = self.preproc
        elif callable(func):
            self._preproc = func
        else:
            msg = f"{func} is not callable!"
            raise ValueError(msg)

    def __len__(self: PatchDatasetABC) -> int:
        """Return the length of the instance attributes."""
        return len(self.inputs)

    @abstractmethod
    def __getitem__(self: PatchDatasetABC, idx: int) -> None:
        """Get an item from the dataset."""
        ...  # pragma: no cover
