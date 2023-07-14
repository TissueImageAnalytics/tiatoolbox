from dataclasses import dataclass, field, replace
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
        patch_input_shape (:class:`numpy.ndarray`, list(int), tuple(int, int)):
            Shape of the largest input in (height, width).
        stride_shape (:class:`numpy.ndarray`, list(int), tuple(int)):
            Stride in (x, y) direction for patch extraction.
        output_resolutions (list(dict)):
            Resolution of each output head from model inference, must be
            in the same order as target model.infer_batch().

    Attributes:
        input_resolutions (list(dict)):
            Resolution of each input head of model inference, must be in
            the same order as `target model.forward()`.
        patch_input_shape (:class:`numpy.ndarray`, list(int), tuple(int, int)):
            Shape of the largest input in (height, width).
        stride_shape (:class:`numpy.ndarray`, list(int), tuple(int, int)):
            Stride in (x, y) direction for patch extraction.
        output_resolutions (list(dict)):
            Resolution of each output head from model inference, must be
            in the same order as target model.infer_batch().
        highest_input_resolution (dict):
            Highest resolution to process the image based on input and
            output resolutions. This helps to read the image at the optimal
            resolution and improves performance.

    Examples:
        >>> # Defining io for a base network and converting to baseline.
        >>> ioconfig = ModelIOConfigABC(
        ...     input_resolutions=[{"units": "mpp", "resolution": 0.5}],
        ...     output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        ...     patch_input_shape=(224, 224),
        ...     stride_shape=(224, 224),
        ... )
        >>> ioconfig = ioconfig.to_baseline()

    """

    input_resolutions: List[dict]
    patch_input_shape: Union[List[int], np.ndarray, Tuple[int, int]]
    stride_shape: Union[List[int], np.ndarray, Tuple[int, int]] = None
    output_resolutions: List[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.stride_shape is None:
            self.stride_shape = self.patch_input_shape

        self.resolution_unit = self.input_resolutions[0]["units"]

        if self.resolution_unit == "mpp":
            self.highest_input_resolution = min(
                self.input_resolutions, key=lambda x: x["resolution"]
            )
        else:
            self.highest_input_resolution = max(
                self.input_resolutions, key=lambda x: x["resolution"]
            )

        self._validate()

    def _validate(self):
        """Validate the data format."""
        resolutions = self.input_resolutions + self.output_resolutions
        units = [v["units"] for v in resolutions]
        units = np.unique(units)

        if len(units) != 1:
            raise ValueError(
                f"Multiple resolution units found: `{units}`. "
                f"Mixing resolution units is not allowed."
            )

        if units[0] not in [
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

        return replace(self, input_resolutions=input_resolutions, output_resolutions=[])


@dataclass
class IOSegmentorConfig(ModelIOConfigABC):
    """Contains semantic segmentor input and output information.

    Args:
        input_resolutions (list(dict)):
            Resolution of each input head of model inference, must be in
            the same order as `target model.forward()`.
        patch_input_shape (:class:`numpy.ndarray`, list(int), tuple(int, int)):
            Shape of the largest input in (height, width).
        stride_shape (:class:`numpy.ndarray`, list(int), tuple(int)):
            Stride in (x, y) direction for patch extraction.
        output_resolutions (list(dict)):
            Resolution of each output head from model inference, must be
            in the same order as target model.infer_batch().
        patch_output_shape (:class:`numpy.ndarray`, list(int)):
            Shape of the largest output in (height, width).
        save_resolution (dict):
            Resolution to save all output.

    Attributes:
        input_resolutions (list(dict)):
            Resolution of each input head of model inference, must be in
            the same order as `target model.forward()`.
        patch_input_shape (:class:`numpy.ndarray`, list(int), tuple(int, int)):
            Shape of the largest input in (height, width).
        stride_shape (:class:`numpy.ndarray`, list(int), tuple(int, int)):
            Stride in (x, y) direction for patch extraction.
        output_resolutions (list(dict)):
            Resolution of each output head from model inference, must be
            in the same order as target model.infer_batch().
        patch_output_shape (:class:`numpy.ndarray`, list(int)):
            Shape of the largest output in (height, width).
        save_resolution (dict):
            Resolution to save all output.
        highest_input_resolution (dict):
            Highest resolution to process the image based on input and
            output resolutions. This helps to read the image at the optimal
            resolution and improves performance.

    Examples:
        >>> # Defining io for a network having 1 input and 1 output at the
        >>> # same resolution
        >>> ioconfig = IOSegmentorConfig(
        ...     input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        ...     output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        ...     patch_input_shape=(2048, 2048),
        ...     patch_output_shape=(1024, 1024),
        ...     stride_shape=(512, 512),
        ... )
        >>> # Defining io for a network having 3 input and 2 output
        >>> # at the same resolution, the output is then merged at a
        >>> # different resolution.
        >>> ioconfig = IOSegmentorConfig(
        ...     input_resolutions=[
        ...         {"units": "mpp", "resolution": 0.25},
        ...         {"units": "mpp", "resolution": 0.50},
        ...         {"units": "mpp", "resolution": 0.75},
        ...     ],
        ...     output_resolutions=[
        ...         {"units": "mpp", "resolution": 0.25},
        ...         {"units": "mpp", "resolution": 0.50},
        ...     ],
        ...     patch_input_shape=(2048, 2048),
        ...     patch_output_shape=(1024, 1024),
        ...     stride_shape=(512, 512),
        ...     save_resolution={"units": "mpp", "resolution": 4.0},
        ... )

    """

    patch_output_shape: Union[List[int], np.ndarray, Tuple[int, int]] = None
    save_resolution: dict = None

    def to_baseline(self):
        """Returns a new config object converted to baseline form.

        This will return a new :class:`IOSegmentorConfig` where
        resolutions have been converted to baseline format with the
        highest possible resolution found in both input and output as
        reference.

        """
        new_config = super().to_baseline()
        resolutions = self.input_resolutions + self.output_resolutions
        if self.save_resolution is not None:
            resolutions.append(self.save_resolution)

        scale_factors = self.scale_to_highest(resolutions, self.resolution_unit)
        num_input_resolutions = len(self.input_resolutions)
        num_output_resolutions = len(self.output_resolutions)

        end_idx = num_input_resolutions + num_output_resolutions
        output_resolutions = [
            {"units": "baseline", "resolution": v}
            for v in scale_factors[num_input_resolutions:end_idx]
        ]

        save_resolution = None
        if self.save_resolution is not None:
            save_resolution = {"units": "baseline", "resolution": scale_factors[-1]}

        return replace(
            self,
            input_resolutions=new_config.input_resolutions,
            output_resolutions=output_resolutions,
            save_resolution=save_resolution,
        )


class IOPatchPredictorConfig(ModelIOConfigABC):
    """Contains patch predictor input and output information.

    Args:
        input_resolutions (list(dict)):
            Resolution of each input head of model inference, must be in
            the same order as `target model.forward()`.
        patch_input_shape (:class:`numpy.ndarray`, list(int), tuple(int)):
            Shape of the largest input in (height, width).
        stride_shape (:class:`numpy.ndarray`, list(int), tuple(int)):
            Stride in (x, y) direction for patch extraction.
        output_resolutions (list(dict)):
            Resolution of each output head from model inference, must be
            in the same order as target model.infer_batch().

    Attributes:
        input_resolutions (list(dict)):
            Resolution of each input head of model inference, must be in
            the same order as `target model.forward()`.
        patch_input_shape (:class:`numpy.ndarray`, list(int), tuple(int)):
            Shape of the largest input in (height, width).
        stride_shape (:class:`numpy.ndarray`, list(int), tuple(int)):
            Stride in (x, y) direction for patch extraction.
        output_resolutions (list(dict)):
            Resolution of each output head from model inference, must be
            in the same order as target model.infer_batch().
        highest_input_resolution (dict):
            Highest resolution to process the image based on input and
            output resolutions. This helps to read the image at the optimal
            resolution and improves performance.

    Examples:
        >>> # Defining io for a patch predictor network
        >>> ioconfig = IOPatchPredictorConfig(
        ...     input_resolutions=[{"units": "mpp", "resolution": 0.5}],
        ...     output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        ...     patch_input_shape=(224, 224),
        ...     stride_shape=(224, 224),
        ... )

    """


@dataclass
class IOInstanceSegmentorConfig(IOSegmentorConfig):
    """Contains instance segmentor input and output information.

    Args:
        input_resolutions (list(dict)):
            Resolution of each input head of model inference, must be in
            the same order as `target model.forward()`.
        patch_input_shape (:class:`numpy.ndarray`, list(int), tuple(int, int)):
            Shape of the largest input in (height, width).
        stride_shape (:class:`numpy.ndarray`, list(int), tuple(int, int)):
            Stride in (x, y) direction for patch extraction.
        output_resolutions (list(dict)):
            Resolution of each output head from model inference, must be
            in the same order as target model.infer_batch().
        patch_output_shape (:class:`numpy.ndarray`, list(int)):
            Shape of the largest output in (height, width).
        save_resolution (dict):
            Resolution to save all output.
        margin (int):
            Tile margin to accumulate the output.
        tile_shape (tuple(int, int)):
            Tile shape to process the WSI.

    Attributes:
        input_resolutions (list(dict)):
            Resolution of each input head of model inference, must be in
            the same order as `target model.forward()`.
        patch_input_shape (:class:`numpy.ndarray`, list(int), tuple(int, int)):
            Shape of the largest input in (height, width).
        stride_shape (:class:`numpy.ndarray`, list(int), tuple(int)):
            Stride in (x, y) direction for patch extraction.
        output_resolutions (list(dict)):
            Resolution of each output head from model inference, must be
            in the same order as target model.infer_batch().
        patch_output_shape (:class:`numpy.ndarray`, list(int)):
            Shape of the largest output in (height, width).
        save_resolution (dict):
            Resolution to save all output.
        highest_input_resolution (dict):
            Highest resolution to process the image based on input and
            output resolutions. This helps to read the image at the optimal
            resolution and improves performance.
        margin (int):
            Tile margin to accumulate the output.
        tile_shape (tuple(int, int)):
            Tile shape to process the WSI.

    Examples:
        >>> # Defining io for a network having 1 input and 1 output at the
        >>> # same resolution
        >>> ioconfig = IOInstanceSegmentorConfig(
        ...     input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        ...     output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        ...     patch_input_shape=(2048, 2048),
        ...     patch_output_shape=(1024, 1024),
        ...     stride_shape=(512, 512),
        ...     margin=128,
        ...     tile_shape=(1024, 1024),
        ... )
        >>> # Defining io for a network having 3 input and 2 output
        >>> # at the same resolution, the output is then merged at a
        >>> # different resolution.
        >>> ioconfig = IOInstanceSegmentorConfig(
        ...     input_resolutions=[
        ...         {"units": "mpp", "resolution": 0.25},
        ...         {"units": "mpp", "resolution": 0.50},
        ...         {"units": "mpp", "resolution": 0.75},
        ...     ],
        ...     output_resolutions=[
        ...         {"units": "mpp", "resolution": 0.25},
        ...         {"units": "mpp", "resolution": 0.50},
        ...     ],
        ...     patch_input_shape=(2048, 2048),
        ...     patch_output_shape=(1024, 1024),
        ...     stride_shape=(512, 512),
        ...     save_resolution={"units": "mpp", "resolution": 4.0},
        ...     margin=128,
        ...     tile_shape=(1024, 1024),
        ... )

    """

    margin: int = None
    tile_shape: Tuple[int, int] = None

    def to_baseline(self):
        """Returns a new config object converted to baseline form.

        This will return a new :class:`IOSegmentorConfig` where
        resolutions have been converted to baseline format with the
        highest possible resolution found in both input and output as
        reference.

        """
        new_config = super().to_baseline()

        return replace(
            self,
            input_resolutions=new_config.input_resolutions,
            output_resolutions=new_config.output_resolutions,
            save_resolution=new_config.save_resolution,
        )
