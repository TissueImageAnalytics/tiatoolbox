from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from tiatoolbox.wsicore.wsimeta import Units


@dataclass
class ModelIOConfigABC:
    """Defines a data class for holding a deep learning model's I/O information.

    Enforcing such that following attributes must always be defined by
    the subclass.

    Args:
        input_resolutions (list(dict)):
            Resolution of each input head of model inference, must be in
            the same order as `target model.forward()`.
        stride_shape (:class:`numpy.ndarray`, list(int), tuple(int)):
            Stride in (x, y) direction for patch extraction.
        patch_input_shape (:class:`numpy.ndarray`, list(int), tuple(int)):
            Shape of the largest input in (height, width).

    """

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
                Resolution units.

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
        """Returns a new config object converted to baseline form.

        This will return a new :class:`ModelIOConfigABC` where
        resolutions have been converted to baseline format with the
        highest possible resolution found in both input and output as
        reference.

        """
        resolutions = self.input_resolutions + self.output_resolutions
        save_resolution = getattr(self, "save_resolution", None)
        if save_resolution is not None:
            resolutions.append(save_resolution)

        scale_factors = self.scale_to_highest(resolutions, self.resolution_unit)
        num_input_resolutions = len(self.input_resolutions)

        end_idx = num_input_resolutions
        input_resolutions = [
            {"units": "baseline", "resolution": v} for v in scale_factors[:end_idx]
        ]

        return ModelIOConfigABC(
            input_resolutions=input_resolutions,
            patch_input_shape=self.patch_input_shape,
            stride_shape=self.stride_shape,
            **self._kwargs,
        )


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
