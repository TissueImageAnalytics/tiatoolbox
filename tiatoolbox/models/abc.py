# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Defines Abstract Base Class for Models defined in tiatoolbox."""
from abc import ABC, abstractmethod

import torch.nn as nn


class IOConfigABC(ABC):
    """Define an abstract class for holding predictor I/O information.

    Enforcing such that following attributes must always be defined by
    the subclass.

    Attributes:
        input_resolutions (list):
            Define the resolution of each input, incase the predictor
            receives variable input. Must be in the same order as
            network input.
        output_resolutions (dict):
            Define the resolution of each output, incase the predictor
            return variable output.Must be in the same order as network
            output.

    """

    @property
    @abstractmethod
    def input_resolutions(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def output_resolutions(self):
        raise NotImplementedError


class ModelABC(ABC, nn.Module):
    """Abstract base class for models used in tiatoolbox."""

    def __init__(self):
        super().__init__()
        self._postproc = self.postproc
        self._preproc = self.preproc

    @abstractmethod
    # noqa
    # This is generic abc, else pylint will complain
    def forward(self, *args, **kwargs):
        """Torch method, this contains logic for using layers defined in init."""
        ...  # pragma: no cover

    @staticmethod
    @abstractmethod
    def infer_batch(model, batch_data, on_gpu):
        """Run inference on an input batch.

        Contains logic for forward operation as well as I/O aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (ndarray):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            on_gpu (bool):
                Whether to run inference on a GPU.

        """
        ...  # pragma: no cover

    @staticmethod
    def preproc(image):
        """Define the pre-processing of this class of model."""
        return image

    @staticmethod
    def postproc(image):
        """Define the post-processing of this class of model."""
        return image

    @property
    def preproc_func(self):
        """Return the current pre-processing function of this instance."""
        return self._preproc

    @preproc_func.setter
    def preproc_func(self, func):
        """Set the pre-processing function for this instance.

        If `func=None`, the method will default to `self.preproc`.
        Otherwise, `func` is expected to be callable.

        Examples:
            >>> # expected usage
            >>> # model is a subclass object of this ModelABC
            >>> # `func` is an user defined function
            >>> model.preproc_func = func
            >>> transformed_img = model.preproc_func(img)

        """
        if func is not None and not callable(func):
            raise ValueError(f"{func} is not callable!")

        if func is None:
            self._preproc = self.preproc
        else:
            self._preproc = func

    @property
    def postproc_func(self):
        """Return the current post-processing function of this instance."""
        return self._postproc

    @postproc_func.setter
    def postproc_func(self, func):
        """Set the pre-processing function for this instance of model.

        If `func=None`, the method will default to `self.postproc`.
        Otherwise, `func` is expected to be callable and behave as
        follows:

        Examples:
            >>> # expected usage
            >>> # model is a subclass object of this ModelABC
            >>> # `func` is an user defined function
            >>> model.postproc_func = func
            >>> transformed_img = model.postproc_func(img)

        """
        if func is not None and not callable(func):
            raise ValueError(f"{func} is not callable!")

        if func is None:
            self._postproc = self.postproc
        else:
            self._postproc = func
