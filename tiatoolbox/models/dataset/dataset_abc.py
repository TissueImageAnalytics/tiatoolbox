"""Defines dataset abstract classes."""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from tiatoolbox.utils import imread


class PatchDatasetABC(ABC, torch.utils.data.Dataset):
    """Defines abstract base class for patch dataset."""

    def __init__(
        self,
    ) -> None:
        """Initializes :class:`PatchDatasetABC`."""
        super().__init__()
        self._preproc = self.preproc
        self.data_is_npy_alike = False
        self.inputs = []
        self.labels = []

    @staticmethod
    def _check_shape_integrity(shapes):
        """Checks the integrity of input shapes.

        Args:
            shapes (list or np.ndarray):
                input shape to check.

        Raises:
            ValueError: If the shape is not valid.

        """
        if any(len(v) != 3 for v in shapes):
            msg = "Each sample must be an array of the form HWC."
            raise ValueError(msg)

        max_shape = np.max(shapes, axis=0)
        if (shapes - max_shape[None]).sum() != 0:
            msg = "Images must have the same dimensions."
            raise ValueError(msg)

    def _check_input_integrity(self, mode):
        """Check that variables received during init are valid.

        These checks include:
            - Input is of a singular data type, such as a list of paths.
            - If it is list of images, all images are of the same height
              and width.

        """
        if mode == "patch":
            self.data_is_npy_alike = False
            is_all_paths = all(isinstance(v, (Path, str)) for v in self.inputs)
            is_all_npy = all(isinstance(v, np.ndarray) for v in self.inputs)

            msg = (
                "Input must be either a list/array of images "
                "or a list of valid image paths."
            )

            if not (is_all_paths or is_all_npy or isinstance(self.inputs, np.ndarray)):
                raise ValueError(
                    msg,
                )

            shapes = None
            # When a list of paths is provided
            if is_all_paths:
                if any(not Path.exists(v) for v in self.inputs):
                    # at least one of the paths are invalid
                    raise ValueError(
                        msg,
                    )
                # Preload test for sanity check
                shapes = [self.load_img(v).shape for v in self.inputs]
                self.data_is_npy_alike = False

            if is_all_npy:
                shapes = [v.shape for v in self.inputs]
                self.data_is_npy_alike = True

            if shapes:
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
    def load_img(path):
        """Load an image from a provided path.

        Args:
            path (str): Path to an image file.

        """
        path = Path(path)

        if path.suffix not in (".npy", ".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            msg = f"Cannot load image data from `{path.suffix}` files."
            raise ValueError(msg)

        return imread(path, as_uint8=False)

    @staticmethod
    def preproc(image):
        """Define the pre-processing of this class of loader."""
        return image

    @property
    def preproc_func(self):
        """Return the current pre-processing function of this instance.

        The returned function is expected to behave as follows:
        >>> transformed_img = func(img)

        """
        return self._preproc

    @preproc_func.setter
    def preproc_func(self, func):
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

    def __len__(self) -> int:
        """Returns the length of the instance attributes."""
        return len(self.inputs)

    @abstractmethod
    def __getitem__(self, idx):
        """Defines the behaviour when an item is accessed."""
        ...  # pragma: no cover
