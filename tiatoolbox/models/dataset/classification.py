"""Define classes and methods for classification datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchvision import transforms

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import torch
    from PIL import Image


class _TorchPreprocCaller:
    """Wrapper for applying PyTorch transforms.

    Args:
        preprocs (list):
            List of torchvision transforms for preprocessing the image.
            The transforms will be applied in the order that they are
            given in the list. For more information, visit the following
            link: https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(
        self: _TorchPreprocCaller, preprocs: list, *, dataset_name: str | None = None
    ) -> None:
        self.func = transforms.Compose(preprocs)
        self.dataset_name = dataset_name

    def __call__(self: _TorchPreprocCaller, img: np.ndarray | Image) -> torch.Tensor:
        tensor: torch.Tensor = self.func(img)
        return tensor.permute((1, 2, 0))

    def __eq__(self: _TorchPreprocCaller, other: object) -> bool:
        """Treat predefined preprocessing for the same dataset as equal."""
        name = getattr(self, "dataset_name", None)
        if name is None or not isinstance(other, _TorchPreprocCaller):
            return NotImplemented
        return name == getattr(other, "dataset_name", None)

    def __hash__(self: _TorchPreprocCaller) -> int:
        """Hash consistently with equality."""
        name = getattr(self, "dataset_name", None)
        return hash(name) if name is not None else id(self)


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

    # Lets trained-model configs record and restore the same preprocessing.
    return _TorchPreprocCaller(preproc_dict[dataset_name], dataset_name=dataset_name)
