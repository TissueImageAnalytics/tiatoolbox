"""Define classes and methods for classification datasets."""
from __future__ import annotations

from typing import TYPE_CHECKING

import PIL
from PIL import Image
from torchvision import transforms

from tiatoolbox.models.dataset import dataset_abc

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


class _TorchPreprocCaller:
    """Wrapper for applying PyTorch transforms.

    Args:
        preprocs (list):
            List of torchvision transforms for preprocessing the image.
            The transforms will be applied in the order that they are
            given in the list. For more information, visit the following
            link: https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(self: _TorchPreprocCaller, preprocs: list) -> None:
        self.func = transforms.Compose(preprocs)

    def __call__(self: _TorchPreprocCaller, img: np.ndarray) -> Image:
        img = PIL.Image.fromarray(img)
        img = self.func(img)
        return img.permute(1, 2, 0)


def predefined_preproc_func(dataset_name: str) -> _TorchPreprocCaller:
    """Get the preprocessing information used for the pretrained model.

    Args:
        dataset_name (str):
            Dataset name used to determine what preprocessing was used.

    Returns:
        _TorchPreprocCaller:
            Preprocessing function for transforming the input data.

    """
    preproc_dict = {
        "kather100k": [
            transforms.ToTensor(),
        ],
        "pcam": [
            transforms.ToTensor(),
        ],
    }

    if dataset_name not in preproc_dict:
        msg = f"Predefined preprocessing for dataset `{dataset_name}` does not exist."
        raise ValueError(
            msg,
        )

    preprocs = preproc_dict[dataset_name]
    return _TorchPreprocCaller(preprocs)


class PatchDataset(dataset_abc.PatchDatasetABC):
    """Define PatchDataset for torch inference.

    Define a simple patch dataset, which inherits from the
      `torch.utils.data.Dataset` class.

    Attributes:
        inputs (list or np.ndarray):
            Either a list of patches, where each patch is a ndarray or a
            list of valid path with its extension be (".jpg", ".jpeg",
            ".tif", ".tiff", ".png") pointing to an image.
        labels (list):
            List of labels for sample at the same index in `inputs`.
            Default is `None`.

    Examples:
        >>> # A user defined preproc func and expected behavior
        >>> preproc_func = lambda img: img/2  # reduce intensity by half
        >>> transformed_img = preproc_func(img)
        >>> # create a dataset to get patches preprocessed by the above function
        >>> ds = PatchDataset(
        ...     inputs=['/A/B/C/img1.png', '/A/B/C/img2.png'],
        ...     labels=["labels1", "labels2"],
        ... )

    """

    def __init__(
        self: PatchDataset,
        inputs: np.ndarray | list,
        labels: list | None = None,
    ) -> None:
        """Initialize :class:`PatchDataset`."""
        super().__init__()

        self.data_is_npy_alike = False

        self.inputs = inputs
        self.labels = labels

        # perform check on the input
        self._check_input_integrity(mode="patch")

    def __getitem__(self: PatchDataset, idx: int) -> dict:
        """Get an item from the dataset."""
        patch = self.inputs[idx]

        # Mode 0 is list of paths
        if not self.data_is_npy_alike:
            patch = self.load_img(patch)

        # Apply preprocessing to selected patch
        patch = self._preproc(patch)

        data = {
            "image": patch,
        }
        if self.labels is not None:
            data["label"] = self.labels[idx]
            return data

        return data
