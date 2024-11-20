"""Define Abstract Base Class for Models defined in tiatoolbox."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch._dynamo
from torch import device as torch_device
from torch import nn

torch._dynamo.config.suppress_errors = True  # skipcq: PYL-W0212  # noqa: SLF001


if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    import numpy as np


def load_torch_model(model: nn.Module, weights: str | Path) -> nn.Module:
    """Helper function to load a torch model.

    Args:
        model (torch.nn.Module):
            A torch model.
        weights (str or Path):
            Path to pretrained weights.

    Returns:
        torch.nn.Module:
            Torch model with pretrained weights loaded on CPU.

    """
    # ! assume to be saved in single GPU mode
    # always load on to the CPU
    saved_state_dict = torch.load(weights, map_location="cpu")
    model.load_state_dict(saved_state_dict, strict=True)
    return model


def model_to(model: torch.nn.Module, device: str = "cpu") -> torch.nn.Module:
    """Transfers model to cpu/gpu.

    Args:
        model (torch.nn.Module):
            PyTorch defined model.
        device (str):
            Transfers model to the specified device. Default is "cpu".

    Returns:
        torch.nn.Module:
            The model after being moved to cpu/gpu.

    """
    if device != "cpu":
        # DataParallel work only for cuda
        model = torch.nn.DataParallel(model)

    device = torch.device(device)
    return model.to(device)


class ModelABC(ABC, torch.nn.Module):
    """Abstract base class for models used in tiatoolbox."""

    def __init__(self: ModelABC) -> None:
        """Initialize Abstract class ModelABC."""
        super().__init__()
        self._postproc = self.postproc
        self._preproc = self.preproc

    @abstractmethod
    # This is generic abc, else pylint will complain
    def forward(self: ModelABC, *args: tuple[Any, ...], **kwargs: dict) -> None:
        """Torch method, this contains logic for using layers defined in init."""
        ...  # pragma: no cover

    @staticmethod
    def preproc(image: np.ndarray) -> np.ndarray:
        """Define the pre-processing of this class of model."""
        return image

    @staticmethod
    def postproc(image: np.ndarray) -> np.ndarray:
        """Define the post-processing of this class of model."""
        return image

    @property
    def preproc_func(self: ModelABC) -> Callable:
        """Return the current pre-processing function of this instance."""
        return self._preproc

    @preproc_func.setter
    def preproc_func(self: ModelABC, func: Callable) -> None:
        """Set the pre-processing function for this instance.

        If `func=None`, the method will default to `self.preproc`.
        Otherwise, `func` is expected to be callable.

        Examples:
            >>> # expected usage
            >>> # model is a subclass object of this ModelABC
            >>> # `func` is a user defined function
            >>> model = ModelABC()
            >>> model.preproc_func = func
            >>> transformed_img = model.preproc_func(image=np.ndarray)

        """
        if func is not None and not callable(func):
            msg = f"{func} is not callable!"
            raise ValueError(msg)

        if func is None:
            self._preproc = self.preproc
        else:
            self._preproc = func

    @property
    def postproc_func(self: ModelABC) -> Callable:
        """Return the current post-processing function of this instance."""
        return self._postproc

    @postproc_func.setter
    def postproc_func(self: ModelABC, func: Callable) -> None:
        """Set the pre-processing function for this instance of model.

        If `func=None`, the method will default to `self.postproc`.
        Otherwise, `func` is expected to be callable and behave as
        follows:

        Examples:
            >>> # expected usage
            >>> # model is a subclass object of this ModelABC
            >>> # `func` is a user defined function
            >>> model = ModelABC()
            >>> model.postproc_func = func
            >>> transformed_img = model.postproc_func(image=np.ndarray)

        """
        if func is not None and not callable(func):
            msg = f"{func} is not callable!"
            raise ValueError(msg)

        if func is None:
            self._postproc = self.postproc
        else:
            self._postproc = func

    def to(self: ModelABC, device: str = "cpu") -> torch.nn.Module:
        """Transfers model to cpu/gpu.

        Args:
            self (ModelABC):
                PyTorch defined model.
            device (str):
                Transfers model to the specified device. Default is "cpu".

        Returns:
            torch.nn.Module:
                The model after being moved to cpu/gpu.

        """
        device = torch_device(device)
        model = super().to(device)

        # If target device istorch.cuda and more
        # than one GPU is available, use DataParallel
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)  # pragma: no cover

        return model

    def load_weights_from_file(self: ModelABC, weights: str | Path) -> torch.nn.Module:
        """Helper function to load a torch model.

        Args:
            self (ModelABC):
                A torch model as :class:`ModelABC`.
            weights (str or Path):
                Path to pretrained weights.

        Returns:
            torch.nn.Module:
                Torch model with pretrained weights loaded on CPU.

        """
        # ! assume to be saved in single GPU mode
        # always load on to the CPU
        saved_state_dict = torch.load(weights, map_location="cpu")
        return super().load_state_dict(saved_state_dict, strict=True)
