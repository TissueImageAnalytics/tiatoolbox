from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np

from tiatoolbox.wsicore.wsimeta import Units


class IOConfigABC(ABC):
    """Define an abstract class for holding predictor I/O information.

    Enforcing such that following attributes must always be defined by
    the subclass.

    Args:
        input_resolutions (list):
            Resolution of each input head of model inference, must be in
            the same order as `target model.forward()`.
        stride_shape (:class:`numpy.ndarray`, list(int)):
            Stride in (x, y) direction for patch extraction.
        patch_input_shape (:class:`numpy.ndarray`, list(int)):
            Shape of the largest input in (height, width).

    """

    # We pre-define to follow enforcement, actual initialisation in init
    input_resolutions = None

    def __init__(
        self,
        input_resolutions: List[dict],
        patch_input_shape: Union[List[int], np.ndarray, Tuple[int]],
        stride_shape: Union[List[int], np.ndarray, Tuple[int]],
        **kwargs,
    ):
        self._kwargs = kwargs
        self.patch_input_shape = patch_input_shape
        self.stride_shape = stride_shape
        self.input_resolutions = input_resolutions
        self.output_resolutions = []
        # output_resolutions are equal to input resolutions by default
        # but these are customizable.
        self.resolution_unit = input_resolutions[0]["units"]

        for variable, value in kwargs.items():
            self.__setattr__(variable, value)

        if self.resolution_unit == "mpp":
            self.highest_input_resolution = min(
                self.input_resolutions, key=lambda x: x["resolution"]
            )
        else:
            self.highest_input_resolution = max(
                self.input_resolutions, key=lambda x: x["resolution"]
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

    @staticmethod
    def scale_to_highest(resolutions: List[dict], units: Units):
        """Get the scaling factor from input resolutions.

        This will convert resolutions to a scaling factor with respect to
        the highest resolution found in the input resolutions list.

        Args:
            resolutions (list):
                A list of resolutions where one is defined as
                `{'resolution': value, 'unit': value}`
            units (Units):
                Units that the resolutions are at.

        Returns:
            :class:`numpy.ndarray`:
                A 1D array of scaling factors having the same length as
                `resolutions`

        """
        old_val = [v["resolution"] for v in resolutions]
        if units not in ["baseline", "mpp", "power"]:
            raise ValueError(
                f"Unknown units `{units}`. "
                "Units should be one of 'baseline', 'mpp' or 'power'."
            )
        if units == "baseline":
            return old_val
        if units == "mpp":
            return np.min(old_val) / np.array(old_val)
        return np.array(old_val) / np.max(old_val)

    def to_baseline(self):
        """Return a new config object converted to baseline form.

        This will return a new :class:`IOSegmentorConfig` where
        resolutions have been converted to baseline format with the
        highest possible resolution found in both input and output as
        reference.

        """
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
