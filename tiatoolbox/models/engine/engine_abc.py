from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np


class IOConfigABC(ABC):
    """Define an abstract class for holding predictor I/O information.

    Enforcing such that following attributes must always be defined by
    the subclass.

    """

    def __init__(
        self,
        input_resolutions: List[dict],
        patch_input_shape: Union[List[int], np.ndarray, Tuple[int]],
        stride_shape: Union[List[int], np.ndarray, Tuple[int]],
        **kwargs,
    ):
        self._kwargs = kwargs
        self.resolution_unit = input_resolutions[0]["units"]
        self.patch_input_shape = patch_input_shape
        self.stride_shape = stride_shape

        self._validate()

        if self.resolution_unit == "mpp":
            self.highest_input_resolution = min(
                self.input_resolutions, key=lambda x: x["resolution"],
            )
        else:
            self.highest_input_resolution = max(
                self.input_resolutions, key=lambda x: x["resolution"],
            )

    def _validate(self):
        """Validate the data format."""
        resolutions = self.input_resolutions + self.output_resolutions
        units = [v["units"] for v in resolutions]
        units = np.unique(units)
        if len(units) != 1 or units[0] not in [
            "power",
            "baseline",
            "mpp",
        ]:
            raise ValueError(f"Invalid resolution units `{units[0]}`.")

    @property
    @abstractmethod
    def input_resolutions(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def output_resolutions(self):
        raise NotImplementedError


class EngineABC(ABC):
    """Abstract base class for engines used in tiatoolbox."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def process_patch(self):
        raise NotImplementedError

    # how to deal with patches, list of patches/numpy arrays, WSIs
    # how to communicate with sub-processes.
    # define how to deal with patches as numpy/zarr arrays.
    # convert list of patches/numpy arrays to zarr and then pass to each sub-processes.
    # define how to read WSIs, read the image and convert to zarr array.
