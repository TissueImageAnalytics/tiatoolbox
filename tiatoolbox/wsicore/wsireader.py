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

"""This module defines classes which can read image data from WSI formats."""
import copy
import math
import numbers
import os
import pathlib
import re
import warnings
from datetime import datetime
from numbers import Number
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import openslide
import pandas as pd
import tifffile
import zarr

from tiatoolbox import utils
from tiatoolbox.tools import tissuemask
from tiatoolbox.utils.env_detection import pixman_warning
from tiatoolbox.utils.exceptions import FileNotSupported
from tiatoolbox.wsicore.wsimeta import WSIMeta

pixman_warning()

NumPair = Tuple[Number, Number]
IntPair = Tuple[int, int]
Bounds = Tuple[Number, Number, Number, Number]
IntBounds = Tuple[int, int, int, int]
Resolution = Union[Number, Tuple[Number, Number], np.ndarray]
WSIReaderInput = Union[str, pathlib.Path, np.ndarray, "WSIReader"]


class WSIReader:
    """Base whole slide image (WSI) reader class.

    This class defines functions for reading pixel data and metadata
    from whole slide image (WSI) files.

    Attributes:
        input_img (pathlib.Path): Input path to WSI file.

    Args:
        input_img (:obj:`str` or :obj:`pathlib.Path` or :class:`numpy.ndarray`):
            Input path to WSI.
        mpp (:obj:`tuple` or :obj:`list` or :obj:`None`, optional):
            The MPP of the WSI. If not provided, the MPP is approximated
            from the objective power.
        power (:obj:`float` or :obj:`None`, optional):
            The objective power of the WSI. If not provided, the power
            is approximated from the MPP.

    """

    @staticmethod  # noqa: A003
    def open(
        input_img: Union[str, pathlib.Path, np.ndarray],
        mpp: Optional[Tuple[Number, Number]] = None,
        power: Optional[Number] = None,
    ) -> "WSIReader":
        """Return an appropriate :class:`.WSIReader` object.

        Args:
            input_img (str, pathlib.Path, :class:`numpy.ndarray`, or :obj:WSIReader):
                Input to create a WSI object from.
                Supported types of input are: `str` and `pathlib.Path` which point to
                the location on the disk where image is stored, :class:`numpy.ndarray`
                in which the input image in the form of numpy array (HxWxC) is stored,
                or :obj:WSIReader which is an already created tiatoolbox WSI handler.
                In the latter case, the function directly passes the input_imge to the
                output.
            mpp (tuple):
                (x, y) tuple of the MPP in the units of the input image.
            power (float):
                Objective power of the input image.

        Returns:
            WSIReader: an object with base :class:`.WSIReader` as base class.

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import get_wsireader
            >>> wsi = get_wsireader(input_img="./sample.svs")

        """
        if isinstance(input_img, (str, pathlib.Path)):
            _, _, suffixes = utils.misc.split_path_name_ext(input_img)

            if suffixes[-1] not in [
                ".svs",
                ".npy",
                ".ndpi",
                ".mrxs",
                ".ndpi",
                ".tif",
                ".tiff",
                ".jp2",
                ".png",
                ".jpg",
                ".jpeg",
            ]:
                raise FileNotSupported(
                    f"File {input_img} is not a supported file format."
                )

            if suffixes[-1] in (".npy",):
                input_img = np.load(input_img)
                return VirtualWSIReader(input_img, mpp=mpp, power=power)

            if suffixes[-2:] in ([".ome", ".tiff"],):
                return TIFFWSIReader(input_img, mpp=mpp, power=power)

            if suffixes[-1] in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                return VirtualWSIReader(input_img, mpp=mpp, power=power)

            if suffixes[-1] in (".svs", ".ndpi", ".mrxs"):
                return OpenSlideWSIReader(input_img, mpp=mpp, power=power)

            if suffixes[-1] in (".jp2",):
                return OmnyxJP2WSIReader(input_img, mpp=mpp, power=power)

        if isinstance(input_img, np.ndarray):
            return VirtualWSIReader(input_img, mpp=mpp, power=power)

        if isinstance(
            input_img,
            WSIReader,
        ):
            # input is already a tiatoolbox wsi handler
            return input_img
        raise TypeError("Invalid input. Must be a file path or an ndarray image.")

    def __init__(
        self,
        input_img: Union[str, pathlib.Path, np.ndarray],
        mpp: Optional[Tuple[Number, Number]] = None,
        power: Optional[Number] = None,
    ) -> None:
        if isinstance(input_img, np.ndarray):
            self.input_path = None
        else:
            self.input_path = pathlib.Path(input_img)
        self._m_info = None

        # Set a manual mpp value
        if mpp and isinstance(mpp, Number):
            mpp = (mpp, mpp)
        if mpp and (not hasattr(mpp, "__len__") or len(mpp) != 2):
            raise TypeError("Invalid mpp: Must be an iterable of length 2")
        self._manual_mpp = tuple(mpp) if mpp else None

        # Set a manual power value
        if power and not isinstance(power, Number):
            raise TypeError("Invalid power: Power must be a number")
        self._manual_power = power

    @property
    def info(self) -> WSIMeta:
        """WSI metadata property.

        This property is cached and only generated on the first call.

        Returns:
            WSIMeta: An object containing normalized slide metadata

        """
        # In Python>=3.8 this could be replaced with functools.cached_property
        if self._m_info is not None:
            return copy.deepcopy(self._m_info)
        self._m_info = self._info()
        if self._manual_mpp:
            self._m_info.mpp = np.array(self._manual_mpp)
        if self._manual_power:
            self._m_info.objective_power = self._manual_power
        return self._m_info

    @info.setter
    def info(self, meta: WSIMeta) -> None:
        """WSI metadata setter.

        Args:
            meta (WSIMeta): Metadata object.

        """
        self._m_info = meta

    def _info(self) -> WSIMeta:
        """WSI metadata internal getter used to update info property.

        Missing values for MPP and objective power are approximated and
        a warning raised. Objective power is calculated as the mean of
        the :func:utils.transforms.mpp2common_objective_power in x and
        y. MPP (x and y) is approximated using objective power via
        :func:utils.transforms.objective_power2mpp.

        Returns:
            WSIMeta: An object containing normalized slide metadata.

        """
        raise NotImplementedError

    def _relative_level_scales(
        self, resolution: Resolution, units: str
    ) -> List[np.ndarray]:
        """Calculate scale of each level in the WSI relative to given resolution.

        Find the relative scale of each image pyramid / resolution level
        of the WSI relative to the given resolution and units.

        Values > 1 indicate that the level has a larger scale than the
        target and < 1 indicates that it is smaller.

        Args:
            resolution (float or tuple(float)): Scale to calculate
                relative to units.
            units (str): Units of the scale. Allowed values are: mpp,
                power, level, baseline. Baseline refers to the largest
                resolution in the WSI (level 0).

        Raises:
            ValueError: Missing MPP metadata
            ValueError: Missing objective power metadata
            ValueError: Invalid units

        Returns:
            list: Scale for each level relative to the given scale and
                units.

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import get_wsireader
            >>> wsi = get_wsireader(input_img="./CMU-1.ndpi")
            >>> print(wsi._relative_level_scales(0.5, "mpp"))
            [array([0.91282519, 0.91012514]), array([1.82565039, 1.82025028]) ...

            >>> from tiatoolbox.wsicore.wsireader import get_wsireader
            >>> wsi = get_wsireader(input_img="./CMU-1.ndpi")
            >>> print(wsi._relative_level_scales(0.5, "baseline"))
            [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

        """
        info = self.info

        def np_pair(x):
            """Ensure input x is a numpy array of length 2."""
            if isinstance(x, numbers.Number):
                # If one number is given, the same value is used for x and y
                return np.array([x] * 2)
            return np.array(x)

        @np.vectorize
        def level_to_downsample(x):
            """Get the downsample for a level, interpolating non-integer levels."""
            if isinstance(x, int) or int(x) == x:
                # Return the downsample for the level
                return info.level_downsamples[int(x)]
            # Linearly interpolate between levels
            floor = int(np.floor(x))
            ceil = int(np.ceil(x))
            floor_downsample = info.level_downsamples[floor]
            ceil_downsample = info.level_downsamples[ceil]
            return np.interp(x, [floor, ceil], [floor_downsample, ceil_downsample])

        resolution = np_pair(resolution)

        if units not in ("mpp", "power", "level", "baseline"):
            raise ValueError("Invalid units")
        if units == "mpp":
            if info.mpp is None:
                raise ValueError("MPP is None")
            base_scale = info.mpp
        if units == "power":
            if info.objective_power is None:
                raise ValueError("Objective power is None")
            base_scale = 1 / info.objective_power
            resolution = 1 / resolution
        if units == "level":
            if any(resolution >= len(info.level_downsamples)):
                raise ValueError(
                    f"Target scale level {resolution} "
                    f"> number of levels {len(info.level_downsamples)} in WSI"
                )
            base_scale = 1
            resolution = level_to_downsample(resolution)
        if units == "baseline":
            base_scale = 1
            resolution = 1 / resolution

        return [(base_scale * ds) / resolution for ds in info.level_downsamples]

    def _find_optimal_level_and_downsample(
        self, resolution: Resolution, units: str, precision: int = 3
    ) -> Tuple[int, np.ndarray]:
        """Find the optimal level to read at for a desired resolution and units.

        The optimal level is the most downscaled level of the image
        pyramid (or multi-resolution layer) which is larger than the
        desired target resolution. The returned scale is the downsample
        factor required, post read, to achieve the desired resolution.

        Args:
            resolution (float or tuple(float)): Resolution to
                find optimal read parameters for
            units (str): Units of the scale. Allowed values are the same
                as for `WSIReader._relative_level_scales`
            precision (int or optional): Decimal places to use when
                finding optimal scale. This can be adjusted to avoid
                errors when an unnecessary precision is used. E.g.
                1.1e-10 > 1 is insignificant in most cases.
                Defaults to 3.

        Returns:
            tuple: Optimal read level and scale factor between the
                optimal level and the target scale (usually <= 1):
                - :py:obj:`int` - Level
                - np.ndarray - Scale factor in X and Y

        """
        level_scales = self._relative_level_scales(resolution, units)
        level_resolution_sufficient = [
            all(np.round(x, decimals=precision) <= 1) for x in level_scales
        ]
        # Check if level 0 is lower resolution than required (scale > 1)
        if not any(level_resolution_sufficient):
            level = 0
        else:
            # Find the first level with relative scale >= 1.
            # Note: np.argmax finds the index of the first True element.
            # Here it is used on a reversed list to find the first
            # element <=1, which is the same element as the last <=1
            # element when counting forward in the regular list.
            reverse_index = np.argmax(level_resolution_sufficient[::-1])
            # Convert the index from the reversed list to the regular index (level)
            level = (len(level_scales) - 1) - reverse_index
        scale = level_scales[level]

        # Check for requested resolution > than baseline resolution
        if any(np.array(scale) > 1):
            warnings.warn(
                "Read: Scale > 1."
                "This means that the desired resolution is higher"
                " than the WSI baseline (maximum encoded resolution)."
                " Interpolation of read regions may occur."
            )
        return level, scale

    def find_read_rect_params(
        self,
        location: IntPair,
        size: IntPair,
        resolution: Resolution,
        units: str,
        precision: int = 3,
    ) -> Tuple[int, IntPair, IntPair, NumPair, IntPair]:
        """Find optimal parameters for reading a rect at a given resolution.

        Reading the image at full baseline resolution and re-sampling to
        the desired resolution would require a large amount of memory
        and be very slow. This function checks the other resolutions
        stored in the WSI's pyramid of resolutions to find the lowest
        resolution (smallest level) which is higher resolution (a larger
        level) than the requested output resolution.

        In addition to finding this 'optimal level', the scale
        factor to apply after reading in order to obtain the
        desired resolution is found along with conversions of the
        location and size into level and baseline coordinates.

        Args:
            location (tuple(int)): Location in terms of the baseline
                image (level 0) resolution.
            size (tuple(int)): Desired output size in pixels (width,
                height) tuple.
            resolution (float): Desired output resolution.
            units (str): Units of scale, default = "level".
                Supported units are:
                - microns per pixel ('mpp')
                - objective power ('power')
                - pyramid / resolution level ('level')
                - pixels per baseline pixel ("baseline")
            precision (int, optional): Decimal places to use when
                finding optimal scale. See
                :func:`find_optimal_level_and_downsample` for more.

        Returns:
            tuple: Parameters for reading the requested region
            - :py:obj:`int` - Optimal read level
            - :py:obj:`tuple` - Read location in level coordinates
                - :py:obj:`int` - X location
                - :py:obj:`int` - Y location
            - :py:obj:`tuple` - Region size in level coordinates
                - :py:obj:`int` - Width
                - :py:obj:`int` - Height
            - :py:obj:`tuple` - Scaling to apply after level read to achieve
              desired output resolution.
                - :py:obj:`float` - X scale factor
                - :py:obj:`float` - Y scale factor
            - :py:obj:`tuple` - Region size in baseline coordinates
                - :py:obj:`int` - Width
                - :py:obj:`int` - Height

        """
        read_level, post_read_scale_factor = self._find_optimal_level_and_downsample(
            resolution, units, precision
        )
        info = self.info
        level_downsample = info.level_downsamples[read_level]
        baseline_read_size = np.round(
            np.array(size) * level_downsample / post_read_scale_factor
        ).astype(int)
        level_read_size = np.round(np.array(size) / post_read_scale_factor).astype(int)
        level_location = np.round(np.array(location) / level_downsample).astype(int)
        return (
            read_level,
            level_location,
            level_read_size,
            post_read_scale_factor,
            baseline_read_size,
        )

    def _find_read_params_at_resolution(
        self, location: IntPair, size: IntPair, resolution: Resolution, units: str
    ) -> Tuple[int, NumPair, IntPair, IntPair, IntPair, IntPair]:
        """Works similarly to `_find_read_rect_params`.

        Return the information necessary for scaling. While `_find_read_rect_params`
        assumes location to be at baseline. This function assumes location to be at
        requested resolution.

        Args:
            location (tuple(int)): Location in the requested resolution system.
            size (tuple(int)): Desired output size in pixels (width,
                height) tuple and in the requested resolution system.
            resolution (float): Desired output resolution.
            units (str): Units of scale, default = "level".
                Supported units are:
                - microns per pixel ('mpp')
                - objective power ('power')
                - pyramid / resolution level ('level')
                - pixels per baseline pixel ("baseline")

        Returns:
            tuple: Parameters for reading the requested region
            - :py:obj:`int` - Optimal read level
            - :py:obj:`tuple` - Scaling to apply after level read to achieve
              desired output resolution.
                - :py:obj:`float` - X scale factor
                - :py:obj:`float` - Y scale factor
            - :py:obj:`tuple` - Region size in read level coordinates
                - :py:obj:`int` - Width
                - :py:obj:`int` - Height
            - :py:obj:`tuple` - Region location in read level coordinates
                - :py:obj:`int` - X location
                - :py:obj:`int` - Y location
            - :py:obj:`tuple` - Region size in level 0 coordinates
                - :py:obj:`int` - Width
                - :py:obj:`int` - Height
            - :py:obj:`tuple` - Region location level 0 coordinates
                - :py:obj:`int` - X location
                - :py:obj:`int` - Y location

        """
        (
            read_level,
            # read_level to requested resolution (full)
            read_level_to_resolution_scale_factor,
        ) = self._find_optimal_level_and_downsample(
            resolution,
            units,
        )
        info = self.info

        # Do we need sanity check for input form ?
        requested_location = np.array(location)
        requested_size = np.array(size)
        baseline_to_read_level_scale_factor = 1 / info.level_downsamples[read_level]

        baseline_to_resolution_scale_factor = (
            baseline_to_read_level_scale_factor * read_level_to_resolution_scale_factor
        )

        size_at_baseline = requested_size / baseline_to_resolution_scale_factor
        location_at_baseline = (
            requested_location.astype(np.float32) / baseline_to_resolution_scale_factor
        )
        size_at_read_level = requested_size / read_level_to_resolution_scale_factor
        location_at_read_level = (
            requested_location.astype(np.float32)
            / read_level_to_resolution_scale_factor
        )
        output = (
            size_at_read_level,
            location_at_read_level,
            size_at_baseline,
            location_at_baseline,
        )
        output = tuple([np.ceil(v).astype(np.int64) for v in output])
        return (
            read_level,
            read_level_to_resolution_scale_factor,
        ) + output

    def _bounds_at_resolution_to_baseline(
        self, bounds: Bounds, resolution: Resolution, units: str
    ) -> Bounds:
        """Find corresponding bounds in baseline.

        Find corresponding bounds in baseline given the input
        is at requested resolution.

        """
        bounds_at_resolution = np.array(bounds)
        tl_at_resolution = bounds_at_resolution[:2]  # is in XY
        br_at_resolution = bounds_at_resolution[2:]
        size_at_resolution = br_at_resolution - tl_at_resolution
        # Find parameters for optimal read
        (
            _,  # read_level,
            _,  # read_level_to_requested_scale_factor,
            _,  # size_at_read_level,
            _,  # location_at_read_level,
            size_at_baseline,
            location_at_baseline,
        ) = self._find_read_params_at_resolution(
            tl_at_resolution, size_at_resolution, resolution, units
        )
        tl_at_baseline = location_at_baseline
        br_at_baseline = tl_at_baseline + size_at_baseline
        return np.concatenate([tl_at_baseline, br_at_baseline])  # bounds at baseline

    def slide_dimensions(
        self, resolution: Resolution, units: str, precisions: int = 3
    ) -> IntPair:
        """Return the size of WSI at requested resolution.

        Args:
            resolution (int or float or tuple(float)): resolution to
                read thumbnail at, default = 1.25 (objective power)
            units (str): resolution units, default = "power"

        Returns:
            :py:obj:`tuple`: shape of WSI in (width, height).

        Examples:
            >>> from tiatoolbox.wsicore import wsireader
            >>> wsi = get_wsireader(input_img="./CMU-1.ndpi")
            >>> slide_shape = wsi.slide_dimensions(0.55, 'mpp')

        """
        wsi_shape_at_baseline = self.info.slide_dimensions
        # Find parameters for optimal read
        (_, _, wsi_shape_at_resolution, _,) = self._find_read_bounds_params(
            [0, 0] + list(wsi_shape_at_baseline), resolution, units, precisions
        )
        return wsi_shape_at_resolution

    def _find_read_bounds_params(
        self, bounds: Bounds, resolution: Resolution, units: str, precision: int = 3
    ) -> Tuple[int, IntBounds, IntPair, IntPair, np.ndarray]:
        """Find optimal parameters for reading bounds at a given resolution.

        Args:
            bounds (tuple(int)): Tuple of (start_x, start_y, end_x,
                end_y) i.e. (left, top, right, bottom) of the region in
                baseline reference frame.
            resolution (float): desired output resolution
            units (str): the units of scale, default = "level".
                Supported units are: microns per pixel (mpp), objective
                power (power), pyramid / resolution level (level),
                pixels per baseline pixel (baseline).
            precision (int, optional): Decimal places to use when
                finding optimal scale. See
                :func:`find_optimal_level_and_downsample` for more.

        Returns:
            tuple: Parameters for reading the requested bounds area
                - :py:obj:`int` - Optimal read level
                - :py:obj:`tuple` - Bounds of the region in level coordinates
                    - :py:obj:`int` - Left (start x value)
                    - :py:obj:`int` - Top (start y value)
                    - :py:obj:`int` - Right (end x value)
                    - :py:obj:`int` - Bottom (end y value)
                - :py:obj:`tuple` - Expected size of the output image
                    - :py:obj:`int` - Width
                    - :py:obj:`int` - Height
                - np.ndarray - Scale factor of re-sampling to apply after reading.

        """
        start_x, start_y, end_x, end_y = bounds
        read_level, post_read_scale_factor = self._find_optimal_level_and_downsample(
            resolution, units, precision
        )
        info = self.info
        level_downsample = info.level_downsamples[read_level]
        location = np.array([start_x, start_y])
        size = np.array([end_x - start_x, end_y - start_y])
        level_size = np.round(np.array(size) / level_downsample).astype(int)
        level_location = np.round(location / level_downsample).astype(int)
        level_bounds = (*level_location, *(level_location + level_size))
        output_size = np.round(level_size * post_read_scale_factor).astype(int)
        return (read_level, level_bounds, output_size, post_read_scale_factor)

    def convert_resolution_units(self, input_res, input_unit, output_unit=None):
        """Converts resolution value between different units.

        This function accepts a resolution and its units in the input and converts
        it to all other units ('mpp', 'power', 'baseline'). To achieve resolution
        in 'mpp' and 'power' units in theoutput, WSI meta data should cotain `mpp`
        and `objective_power` information, respectively.

        Args:
            input_res (float): the resolution which we want to convert to
            the other units.
            input_unit (str): the unit of the input resolution (`input_res`). Acceptable
            input_units are 'mpp', 'power', 'baseline', and 'level'.
            output_unit (str): the desired unit to which we want to convert
            the `input_res`. Acceptable values for `output_unit` are: 'mpp',
            'power', and 'baseline'. If `output_unit` is not provided, all of
            the conversions to all of the mentioned units will be returned in a
            dictionary.


        Returns:
            output_res (float or dictionary): either a float which is the
            converted `input_res` to the desired `output_unit` or a dictionary
            containing the converted `input_res` to all acceptable units (`'mpp'`,
            `'power'`, `'baseline'`). If there is not enough meta data to calculate
            a unit (like `mpp` or `power`), they will be set to None in the dictionary.

        """
        baseline_mpp = self.info.mpp
        baseline_power = self.info.objective_power
        if input_unit not in {"mpp", "power", "level", "baseline"}:
            raise ValueError(
                "Invalid input_unit: argument accepts only one of the following "
                " options: `'mpp'`, `'power'`, `'level'`, `'baseline'`."
            )
        if output_unit not in {"mpp", "power", "baseline", None}:
            raise ValueError(
                "Invalid output_unit: argument accepts only one of the following"
                " options: `'mpp'`, `'power'`, `'baseline'`, or None (to return"
                " all units)."
            )
        if baseline_mpp is None and input_unit == "mpp":
            raise ValueError(
                "Missing 'mpp': `input_unit` has been set to 'mpp' while there "
                "is no information about 'mpp' in WSI meta data."
            )
        if baseline_power is None and input_unit == "power":
            raise ValueError(
                "Missing 'objective_power': `input_unit` has been set to 'power' while "
                "there is no information about 'objective_power' in WSI meta data."
            )

        # calculate the output_res based on input_unit and resolution
        output_dict = {
            "mpp": None,
            "power": None,
            "baseline": None,
        }
        if input_unit == "mpp":
            if isinstance(input_res, (list, tuple, np.ndarray)):
                output_dict["mpp"] = np.array(input_res)
            else:
                output_dict["mpp"] = np.array([input_res, input_res])
            output_dict["baseline"] = baseline_mpp[0] / output_dict["mpp"][0]
            if baseline_power is not None:
                output_dict["power"] = output_dict["baseline"] * baseline_power
        elif input_unit == "power":
            output_dict["baseline"] = input_res / baseline_power
            output_dict["power"] = input_res
            if baseline_mpp is not None:
                output_dict["mpp"] = baseline_mpp / output_dict["baseline"]
        elif input_unit == "level":
            level_scales = self._relative_level_scales(input_res, input_unit)
            output_dict["baseline"] = level_scales[0]
            if baseline_mpp is not None:
                output_dict["mpp"] = baseline_mpp / output_dict["baseline"]
            if baseline_power is not None:
                output_dict["power"] = output_dict["baseline"] * baseline_power
        else:  # input_unit == 'baseline'
            output_dict["baseline"] = input_res
            if baseline_mpp is not None:
                output_dict["mpp"] = baseline_mpp / output_dict["baseline"]
            if baseline_power is not None:
                output_dict["power"] = baseline_power * output_dict["baseline"]
        out_res = output_dict[output_unit] if output_unit is not None else output_dict
        if out_res is None:
            warnings.warn(
                "Although unit coversion from input_unit has been done, the requested "
                "output_unit is returned as None. Probably due to missing 'mpp' or "
                "'objective_power' in slide's meta data.",
                UserWarning,
            )
        return out_res

    def _find_tile_params(
        self, tile_objective_value: Number
    ) -> Tuple[int, IntPair, int, Number]:
        """Find the params for save tiles."""
        rescale = self.info.objective_power / tile_objective_value
        if not rescale.is_integer():
            raise ValueError(
                "Tile objective value must be an integer multiple of the "
                "objective power of the slide."
            )
        try:
            level = np.log2(rescale)
            if not level.is_integer():
                raise ValueError
            level = np.int(level)
            slide_dimension = self.info.level_dimensions[level]
            rescale = 1
        # Raise index error if desired pyramid level not embedded
        # in level_dimensions
        except IndexError:
            level = 0
            slide_dimension = self.info.level_dimensions[level]
            rescale = np.int(rescale)
            warnings.warn(
                "Reading WSI at level 0. Desired tile_objective_value"
                + str(tile_objective_value)
                + "not available.",
                UserWarning,
            )
        except ValueError:
            level = 0
            slide_dimension = self.info.level_dimensions[level]
            rescale = 1
            warnings.warn(
                "Reading WSI at level 0. Reading at tile_objective_value"
                + str(tile_objective_value)
                + "not allowed.",
                UserWarning,
            )
            tile_objective_value = self.info.objective_power

        return level, slide_dimension, rescale, tile_objective_value

    def _read_rect_at_resolution(
        self,
        location: NumPair,
        size: NumPair,
        resolution: Resolution = 0,
        units: str = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: Union[Number, Iterable[NumPair]] = 0,
        **kwargs,
    ) -> np.ndarray:
        """Internal helper to perform `read_rect` at resolution.

        In actuality, `read_rect` at resolution is synonymous with
        calling `read_bound` at resolution because `size` has always been
        within the resolution system.

        """
        tl = np.array(location)
        br = location + np.array(size)
        bounds = np.concatenate([tl, br])
        return self.read_bounds(
            bounds,
            resolution=resolution,
            units=units,
            interpolation=interpolation,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
            coord_space="resolution",
            **kwargs,
        )

    def read_rect(
        self,
        location: IntPair,
        size: IntPair,
        resolution: Resolution = 0,
        units: str = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: Union[Number, Iterable[NumPair]] = 0,
        coord_space: str = "baseline",
        **kwargs,
    ) -> np.ndarray:
        """Read a region of the whole slide image at a location and size.

        Location is in terms of the baseline image (level 0  /
        maximum resolution), and size is the output image size.

        Reads can be performed at different resolutions by supplying a
        pair of arguments for the resolution and the
        units of resolution. If meta data does not specify `mpp` or `objective_power`
        then `baseline` units should be selected with resolution 1.0

        The field of view varies with resolution. For a fixed
        field of view see :func:`read_bounds`.

        Args:
            location (tuple(int)): (x, y) tuple giving
                the top left pixel in the baseline (level 0)
                reference frame.
            size (tuple(int)): (width, height) tuple
                giving the desired output image size.
            resolution (int or float or tuple(float)): resolution at
                which to read the image, default = 0. Either a single
                number or a sequence of two numbers for x and y are
                valid. This value is in terms of the corresponding
                units. For example: resolution=0.5 and units="mpp" will
                read the slide at 0.5 microns per-pixel, and
                resolution=3, units="level" will read at level at
                pyramid level / resolution layer 3.
            units (str): the units of resolution, default = "level".
                Supported units are: microns per pixel (mpp), objective
                power (power), pyramid / resolution level (level),
                pixels per baseline pixel (baseline).
            interpolation (str): Method to use when resampling the output
                image. Possible values are "linear", "cubic", "lanczos",
                "area", and "optimse". Defaults to 'optimise' which
                will use cubic interpolation for upscaling and area
                interpolation for downscaling to avoid moiré patterns.
            pad_mode (str): Method to use when padding at the edges of the
                image. Defaults to 'constant'. See :func:`numpy.pad` for
                available modes.
            coord_space (str): default to "baseline", this is a
                flag to indicate if the input `bounds` is in the baseline
                coordinate system ("baseline") or is in the requested resolution
                system ("resolution").
            **kwargs (dict): Extra key-word arguments for reader
                specific parameters. Currently only used by
                VirtualWSIReader. See class docstrings for more
                information.

        Returns:
            :class:`numpy.ndarray`: array of size MxNx3 M=size[0], N=size[1]

        Example:
            >>> from tiatoolbox.wsicore.wsireader import get_wsireader
            >>> # Load a WSI image
            >>> wsi = get_wsireader(input_img="./CMU-1.ndpi")
            >>> location = (0, 0)
            >>> size = (256, 256)
            >>> # Read a region at level 0 (baseline / full resolution)
            >>> img = wsi.read_rect(location, size)
            >>> # Read a region at 0.5 microns per pixel (mpp)
            >>> img = wsi.read_rect(location, size, 0.5, "mpp")
            >>> # This could also be written more verbosely as follows
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=(0.5, 0.5),
            ...     units="mpp",
            ... )

        Note: The field of view varies with resolution when using
        :func:`read_rect`.

        .. figure:: ../images/read_rect_tissue.png
            :width: 512
            :alt: Diagram illustrating read_rect

        As the location is in the baseline reference frame but the size
        (width and height) is the output image size, the field of view
        therefore changes as resolution changes.

        If the WSI does not have a resolution layer
        corresponding exactly to the requested resolution
        (shown above in white with a dashed outline), a larger
        resolution is downscaled to achieve the correct requested output
        resolution.

        If the requested resolution is higher than the
        baseline (maximum resultion of the image), then bicubic
        interpolation is applied to the output image.

        .. figure:: ../images/read_rect-interpolated-reads.png
            :width: 512
            :alt: Diagram illustrating read_rect interpolting between levels

        When reading between the levels stored in the WSI, the coordinates
        of the requested region are projected to the next highest
        resolution. This resolution is then decoded and downsampled
        to produced the desired output. This is a major source of
        variability in the time take to perform a read operation. Reads
        which require reading a large region before downsampling will
        be significantly slower than reading at a fixed level.

        Examples:

            >>> from tiatoolbox.wsicore.wsireader import get_wsireader
            >>> # Load a WSI image
            >>> wsi = get_wsireader(input_img="./CMU-1.ndpi")
            >>> location = (0, 0)
            >>> size = (256, 256)
            >>> # The resolution can be different in x and y, e.g.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=(0.5, 0.75),
            ...     units="mpp",
            ... )
            >>> # Several units can be used including: objective power,
            >>> # microns per pixel, pyramid/resolution level, and
            >>> # fraction of baseline.
            >>> # E.g. Read a region at an objective power of 10x
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=10,
            ...     units="power",
            ... )
            >>> # Read a region at pyramid / resolution level 1
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=1,
            ...     units="level",
            ... )
            >>> # Read at a fractional level, this will linearly
            >>> # interpolate the downsampling factor between levels.
            >>> # E.g. if levels 0 and 1 have a downsampling of 1x and
            >>> # 2x of baseline, then level 0.5 will correspond to a
            >>> # downsampling factor 1.5x of baseline.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.5,
            ...     units="level",
            ... )
            >>> # Read a region at half of the full / baseline
            >>> # resolution.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.5,
            ...     units="baseline",
            ... )
            >>> # Read at a higher resolution than the baseline
            >>> # (interpolation applied to output)
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=1.25,
            ...     units="baseline",
            ... )
            >>> # Assuming the image has a native mpp of 0.5,
            >>> # interpolation will be applied here.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.25,
            ...     units="mpp",
            ... )

        """
        raise NotImplementedError

    def read_bounds(
        self,
        bounds: Bounds,
        resolution: Resolution = 0,
        units: str = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: Union[Number, Iterable[NumPair]] = 0,
        coord_space: str = "baseline",
        **kwargs,
    ) -> np.ndarray:
        """Read a region of the whole slide image within given bounds.

        Bounds are in terms of the baseline image (level 0  /
        maximum resolution).

        Reads can be performed at different resolutions by supplying a
        pair of arguments for the resolution and the
        units of resolution. If meta data does not specify `mpp` or `objective_power`
        then `baseline` units should be selected with resolution 1.0

        The output image size may be different
        to the width and height of the bounds as the resolution will
        affect this. To read a region with a fixed output image size see
        :func:`read_rect`.

        Args:
            bounds (tuple(int)): By default, this is a tuple of (start_x,
                start_y, end_x, end_y) i.e. (left, top, right, bottom) of
                the region in baseline reference frame. However, with
                `coord_space="resolution"`, the bound is expected to
                be at the requested resolution system.
            resolution (int or float or tuple(float)): resolution at
                which to read the image, default = 0. Either a single
                number or a sequence of two numbers for x and y are
                valid. This value is in terms of the corresponding
                units. For example: resolution=0.5 and units="mpp" will
                read the slide at 0.5 microns per-pixel, and
                resolution=3, units="level" will read at level at
                pyramid level / resolution layer 3.
            units (str): the units of resolution, default = "level".
                Supported units are: microns per pixel (mpp), objective
                power (power), pyramid / resolution level (level),
                pixels per baseline pixel (baseline).
            interpolation (str): Method to use when resampling the output
                image. Possible values are "linear", "cubic", "lanczos",
                "area", and "optimse". Defaults to 'optimise' which
                will use cubic interpolation for upscaling and area
                interpolation for downscaling to avoid moiré patterns.
            pad_mode (str): Method to use when padding at the edges of the
                image. Defaults to 'constant'. See :func:`numpy.pad` for
                available modes.
            coord_space (str): default to "baseline", this is a
                flag to indicate if the input `bounds` is in the baseline
                coordinate system ("baseline") or is in the requested resolution
                system ("resolution").
            **kwargs (dict): Extra key-word arguments for reader
                specific parameters. Currently only used by
                :obj:`VirtualWSIReader`. See class docstrings for more
                information.

        Returns:
            :class:`numpy.ndarray`: array of size MxNx3
            M=end_h-start_h, N=end_w-start_w

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import get_wsireader
            >>> from matplotlib import pyplot as plt
            >>> wsi = get_wsireader(input_img="./CMU-1.ndpi")
            >>> # Read a region at level 0 (baseline / full resolution)
            >>> bounds = [1000, 2000, 2000, 3000]
            >>> img = wsi.read_bounds(bounds)
            >>> plt.imshow(img)
            >>> # This could also be written more verbosely as follows
            >>> img = wsi.read_bounds(
            ...     bounds,
            ...     resolution=0,
            ...     units="level",
            ... )
            >>> plt.imshow(img)

        Note: The field of view remains the same as resolution is varied
        when using :func:`read_bounds`.

        .. figure:: ../images/read_bounds_tissue.png
            :width: 512
            :alt: Diagram illustrating read_bounds

        This is because
        the bounds are in the baseline (level 0) reference
        frame. Therefore, varying the resolution does not change what is
        visible within the output image.

        If the WSI does not have a resolution layer
        corresponding exactly to the requested resolution
        (shown above in white with a dashed outline), a larger
        resolution is downscaled to achieve the correct requested output
        resolution.

        If the requested resolution is higher than the
        baseline (maximum resultion of the image), then bicubic
        interpolation is applied to the output image.

        """
        raise NotImplementedError

    def read_region(self, location: NumPair, level: int, size: IntPair) -> np.ndarray:
        """Read a region of the whole slide image (OpenSlide format args).

        This function is to help with writing code which is backwards
        compatible with OpenSlide. As such, it has the same arguments.

        This internally calls :func:`read_rect` which should be
        implemented by any :class:`WSIReader` subclass.
        Therefore, some WSI formats which
        are not supported by OpenSlide, such as Omnyx JP2 files, may
        also be readable with the same syntax.

        Args:
            location (tuple(int)): (x, y) tuple giving the top left
             pixel in the level 0 reference frame.
            level (int): the level number.
            size (tuple(int)): (width, height) tuple giving the region
             size.

        Returns:
            :class:`numpy.ndarray`: array of size MxNx3.

        """
        return self.read_rect(
            location=location, size=size, resolution=level, units="level"
        )

    def slide_thumbnail(self, resolution: Resolution = 1.25, units: str = "power"):
        """Read the whole slide image thumbnail (1.25x by default).

        For more information on resolution and units see :func:`read_rect`

        Args:
            resolution (int or float or tuple(float)): resolution to
                read thumbnail at, default = 1.25 (objective power)
            units (str): resolution units, default = "power"

        Returns:
            :class:`numpy.ndarray`: thumbnail image

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import get_wsireader
            >>> wsi = get_wsireader(input_img="./CMU-1.ndpi")
            >>> slide_thumbnail = wsi.slide_thumbnail()

        """
        slide_dimensions = self.info.slide_dimensions
        bounds = (0, 0, *slide_dimensions)
        return self.read_bounds(bounds, resolution=resolution, units=units)

    def tissue_mask(
        self,
        method: str = "otsu",
        resolution: Resolution = 1.25,
        units: str = "power",
        **masker_kwargs,
    ) -> "VirtualWSIReader":
        """Create a tissue mask and wrap it in a VirtualWSIReader.

        For the morphological method, mpp is used for calculating the
        scale of the morphological operations. If no mpp is available,
        objective power is used instead to estimate a good scale.
        This can be overridden with a custom size, via passing a `kernel_size`
        key-word argument in `masker_kwargs`, see
        :class:`tissuemask.MorphologicalMasker` for more.


        Args:
            method (str): Method to use for creating the mask. Defaults
                to 'otsu'. Methods are: otsu, morphological.
            resolution (float): Resolution to produce the mask at.
                Defaults to 1.25.
            units (str): Units of resolution. Defaults to 'power'.
            **masker_kwargs: Extra kwargs passed to the masker class.

        """
        thumbnail = self.slide_thumbnail(resolution, units)
        if method not in ["otsu", "morphological"]:
            raise ValueError(f"Invalid tissue masking method: {method}")
        if method == "otsu":
            masker = tissuemask.OtsuTissueMasker(**masker_kwargs)
        if method == "morphological":
            mpp = None
            power = None
            if units == "power":
                power = resolution
            if units == "mpp":
                mpp = resolution
            masker = tissuemask.MorphologicalMasker(
                mpp=mpp, power=power, **masker_kwargs
            )
        mask_img = masker.fit_transform([thumbnail])[0]
        return VirtualWSIReader(mask_img.astype(np.uint8), info=self.info, mode="bool")

    def save_tiles(
        self,
        output_dir: Union[str, pathlib.Path],
        tile_objective_value: int,
        tile_read_size: Tuple[int, int],
        tile_format: str = ".jpg",
        verbose: bool = True,
    ) -> None:
        """Generate image tiles from whole slide images.

        Args:
            output_dir(str or pathlib.Path):
                Output directory to save the tiles.
            tile_objective_value (int):
                Objective value at which tile is generated.
            tile_read_size (tuple(int)):
                Tile (width, height).
            tile_format (str):
                File format to save image tiles, defaults to ".jpg".
            verbose (bool):
                Print output, default to True.

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import get_wsireader
            >>> wsi = get_wsireader(input_img="./CMU-1.ndpi")
            >>> wsi.save_tiles(output_dir='./dev_test',
            ...     tile_objective_value=10,
            ...     tile_read_size=(2000, 2000))

            >>> from tiatoolbox.wsicore.wsireader import get_wsireader
            >>> wsi = get_wsireader(input_img="./CMU-1.ndpi")
            >>> slide_param = wsi.info

        """
        output_dir = pathlib.Path(output_dir, self.input_path.name)

        level, slide_dimension, rescale, tile_objective_value = self._find_tile_params(
            tile_objective_value
        )

        tile_read_size = np.multiply(tile_read_size, rescale)
        slide_h = slide_dimension[1]
        slide_w = slide_dimension[0]
        tile_h = tile_read_size[1]
        tile_w = tile_read_size[0]

        iter_tot = 0
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True)
        data = []

        vertical_tiles = int(math.ceil((slide_h - tile_h) / tile_h + 1))
        horizontal_tiles = int(math.ceil((slide_w - tile_w) / tile_w + 1))
        for iter_tot, (h, w) in enumerate(np.ndindex(vertical_tiles, horizontal_tiles)):
            start_h = h * tile_h
            end_h = (h * tile_h) + tile_h
            start_w = w * tile_w
            end_w = (w * tile_w) + tile_w

            end_h = min(end_h, slide_h)
            end_w = min(end_w, slide_w)

            # convert to baseline reference frame
            bounds = start_w, start_h, end_w, end_h
            baseline_bounds = tuple([bound * (2 ** level) for bound in bounds])
            # Read image region
            im = self.read_bounds(baseline_bounds, level)

            if verbose:
                format_str = (
                    "Tile%d:  start_w:%d, end_w:%d, "
                    "start_h:%d, end_h:%d, "
                    "width:%d, height:%d"
                )

                print(
                    format_str
                    % (
                        iter_tot,
                        start_w,
                        end_w,
                        start_h,
                        end_h,
                        end_w - start_w,
                        end_h - start_h,
                    ),
                    flush=True,
                )

            # Rescale to the correct objective value
            if rescale != 1:
                im = utils.transforms.imresize(img=im, scale_factor=rescale)

            img_save_name = (
                "_".join(
                    [
                        "Tile",
                        str(tile_objective_value),
                        str(int(start_w / rescale)),
                        str(int(start_h / rescale)),
                    ]
                )
                + tile_format
            )

            utils.misc.imwrite(image_path=output_dir.joinpath(img_save_name), img=im)

            data.append(
                [
                    iter_tot,
                    img_save_name,
                    start_w,
                    end_w,
                    start_h,
                    end_h,
                    im.shape[0],
                    im.shape[1],
                ]
            )

        # Save information on each slide to relate to the whole slide image
        df = pd.DataFrame(
            data,
            columns=[
                "iter",
                "Tile_Name",
                "start_w",
                "end_w",
                "start_h",
                "end_h",
                "size_w",
                "size_h",
            ],
        )
        df.to_csv(output_dir.joinpath("Output.csv"), index=False)

        # Save slide thumbnail
        slide_thumb = self.slide_thumbnail()
        utils.misc.imwrite(
            output_dir.joinpath("slide_thumbnail" + tile_format), img=slide_thumb
        )


class OpenSlideWSIReader(WSIReader):
    """Reader for OpenSlide supported whole-slide images.

    Supported WSI formats:

    - Aperio (.svs, .tif)
    - Hamamatsu (.vms, .vmu, .ndpi)
    - Leica (.scn)
    - MIRAX (.mrxs)
    - Philips (.tiff)
    - Sakura (.svslide)
    - Trestle (.tif)
    - Ventana (.bif, .tif)
    - Generic tiled TIFF (.tif)


    Attributes:
        openslide_wsi (:obj:`openslide.OpenSlide`)

    """

    def __init__(
        self,
        input_img: Union[str, pathlib.Path, np.ndarray],
        mpp: Optional[Tuple[Number, Number]] = None,
        power: Optional[Number] = None,
    ) -> None:
        super().__init__(input_img=input_img, mpp=mpp, power=power)
        self.openslide_wsi = openslide.OpenSlide(filename=str(self.input_path))

    def read_rect(
        self,
        location,
        size,
        resolution=0,
        units="level",
        interpolation="optimise",
        pad_mode="constant",
        pad_constant_values=0,
        coord_space="baseline",
        **kwargs,
    ):
        if coord_space == "resolution":
            return self._read_rect_at_resolution(
                location,
                size,
                resolution=resolution,
                units=units,
                interpolation=interpolation,
                pad_mode=pad_mode,
                pad_constant_values=pad_constant_values,
            )

        # Find parameters for optimal read
        (
            read_level,
            level_location,
            level_size,
            post_read_scale,
            _,
        ) = self.find_read_rect_params(
            location=location,
            size=size,
            resolution=resolution,
            units=units,
        )

        wsi = self.openslide_wsi

        # Read at optimal level and corrected read size
        im_region = wsi.read_region(location, read_level, level_size)
        im_region = np.array(im_region)

        # Apply padding outside of the slide area
        im_region = utils.image.crop_and_pad_edges(
            bounds=utils.transforms.locsize2bounds(level_location, level_size),
            max_dimensions=self.info.level_dimensions[read_level],
            region=im_region,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
        )

        # Resize to correct scale if required
        im_region = utils.transforms.imresize(
            img=im_region,
            scale_factor=post_read_scale,
            output_size=size,
            interpolation=interpolation,
        )

        return utils.transforms.background_composite(image=im_region)

    def read_bounds(
        self,
        bounds,
        resolution=0,
        units="level",
        interpolation="optimise",
        pad_mode="constant",
        pad_constant_values=0,
        coord_space="baseline",
        **kwargs,
    ):
        # convert from requested to `baseline`
        bounds_at_baseline = bounds
        if coord_space == "resolution":
            bounds_at_baseline = self._bounds_at_resolution_to_baseline(
                bounds, resolution, units
            )
            _, size_at_requested = utils.transforms.bounds2locsize(bounds)
            # don't use the `output_size` (`size_at_requested`) here
            # because the rounding error at `bounds_at_baseline` leads to
            # different `size_at_requested` (keeping same read resolution
            # but base image is of different scale)
            (
                read_level,
                bounds_at_read_level,
                _,
                post_read_scale,
            ) = self._find_read_bounds_params(
                bounds_at_baseline, resolution=resolution, units=units
            )
        else:  # duplicated portion with VirtualReader, factoring out ?
            # Find parameters for optimal read
            (
                read_level,
                bounds_at_read_level,
                size_at_requested,
                post_read_scale,
            ) = self._find_read_bounds_params(
                bounds_at_baseline, resolution=resolution, units=units
            )

        wsi = self.openslide_wsi

        # Read at optimal level and corrected read size
        location_at_baseline = bounds_at_baseline[:2]
        _, size_at_read_level = utils.transforms.bounds2locsize(bounds_at_read_level)
        im_region = wsi.read_region(
            location=location_at_baseline, level=read_level, size=size_at_read_level
        )
        im_region = np.array(im_region)

        # Apply padding outside of the slide area
        im_region = utils.image.crop_and_pad_edges(
            bounds=bounds_at_read_level,
            max_dimensions=self.info.level_dimensions[read_level],
            region=im_region,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
        )

        # Resize to correct scale if required
        if coord_space == "resolution":
            im_region = utils.transforms.imresize(
                img=im_region,
                output_size=size_at_requested,
                interpolation=interpolation,
            )
        else:
            im_region = utils.transforms.imresize(
                img=im_region,
                scale_factor=post_read_scale,
                output_size=size_at_requested,
                interpolation=interpolation,
            )

        return utils.transforms.background_composite(image=im_region)

    def _info(self):
        """Openslide WSI meta data reader.

        Returns:
            WSIMeta: containing meta information.

        """
        props = self.openslide_wsi.properties
        if openslide.PROPERTY_NAME_OBJECTIVE_POWER in props:
            objective_power = float(props[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        else:
            objective_power = None

        slide_dimensions = self.openslide_wsi.level_dimensions[0]
        level_count = self.openslide_wsi.level_count
        level_dimensions = self.openslide_wsi.level_dimensions
        level_downsamples = self.openslide_wsi.level_downsamples
        vendor = props.get(openslide.PROPERTY_NAME_VENDOR)

        # Find microns per pixel (mpp)
        # Initialise to None (value if cannot be determined)
        mpp = None
        # Check OpenSlide for mpp metadata first
        try:
            mpp_x = float(props[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(props[openslide.PROPERTY_NAME_MPP_Y])
            mpp = (mpp_x, mpp_y)
        # Fallback to TIFF resolution units and convert to mpp
        except KeyError:
            tiff_res_units = props.get("tiff.ResolutionUnit")
            try:
                microns_per_unit = {
                    "centimeter": 1e4,  # 10k
                    "inch": 25400,
                }
                x_res = float(props["tiff.XResolution"])
                y_res = float(props["tiff.YResolution"])
                mpp_x = 1 / x_res * microns_per_unit[tiff_res_units]
                mpp_y = 1 / y_res * microns_per_unit[tiff_res_units]
                mpp = [mpp_x, mpp_y]
                warnings.warn(
                    "Metadata: Falling back to TIFF resolution tag"
                    " for microns-per-pixel (MPP)."
                )
            except KeyError:
                warnings.warn("Metadata: Unable to determine microns-per-pixel (MPP).")

        # Fallback to calculating objective power from mpp
        if objective_power is None:
            if mpp is not None:
                objective_power = utils.misc.mpp2common_objective_power(
                    float(np.mean(mpp))
                )
                warnings.warn(
                    "Metadata: Objective power inferred from microns-per-pixel (MPP)."
                )
            else:
                warnings.warn("Metadata: Unable to determine objective power.")

        return WSIMeta(
            file_path=self.input_path,
            axes="YXS",
            objective_power=objective_power,
            slide_dimensions=slide_dimensions,
            level_count=level_count,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            vendor=vendor,
            mpp=mpp,
            raw=dict(**props),
        )


class OmnyxJP2WSIReader(WSIReader):
    """Class for reading Omnyx JP2 images.

    Supported WSI formats:

    - Omnyx JPEG-2000 (.jp2)

    Attributes:
        glymur_wsi (:obj:`glymur.Jp2k`)

    """

    def __init__(
        self,
        input_img: Union[str, pathlib.Path, np.ndarray],
        mpp: Optional[Tuple[Number, Number]] = None,
        power: Optional[Number] = None,
    ) -> None:
        super().__init__(input_img=input_img, mpp=mpp, power=power)
        import glymur

        glymur.set_option("lib.num_threads", os.cpu_count() or 1)
        self.glymur_wsi = glymur.Jp2k(filename=str(self.input_path))

    def read_rect(
        self,
        location,
        size,
        resolution=0,
        units="level",
        interpolation="optimise",
        pad_mode="constant",
        pad_constant_values=0,
        coord_space="baseline",
        **kwargs,
    ):
        if coord_space == "resolution":
            return self._read_rect_at_resolution(
                location,
                size,
                resolution=resolution,
                units=units,
                interpolation=interpolation,
                pad_mode=pad_mode,
                pad_constant_values=pad_constant_values,
            )

        # Find parameters for optimal read
        (
            read_level,
            _,
            _,
            post_read_scale,
            baseline_read_size,
        ) = self.find_read_rect_params(
            location=location,
            size=size,
            resolution=resolution,
            units=units,
        )

        stride = 2 ** read_level
        glymur_wsi = self.glymur_wsi
        bounds = utils.transforms.locsize2bounds(
            location=location, size=baseline_read_size
        )
        im_region = utils.image.safe_padded_read(
            image=glymur_wsi,
            bounds=bounds,
            stride=stride,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
        )

        im_region = utils.transforms.imresize(
            img=im_region,
            scale_factor=post_read_scale,
            output_size=size,
            interpolation=interpolation,
        )

        return utils.transforms.background_composite(image=im_region)

    def read_bounds(
        self,
        bounds,
        resolution=0,
        units="level",
        interpolation="optimise",
        pad_mode="constant",
        pad_constant_values=0,
        coord_space="baseline",
        **kwargs,
    ):

        bounds_at_baseline = bounds
        if coord_space == "resolution":
            bounds_at_baseline = self._bounds_at_resolution_to_baseline(
                bounds, resolution, units
            )
            _, size_at_requested = utils.transforms.bounds2locsize(bounds)
            # don't use the `output_size` (`size_at_requested`) here
            # because the rounding error at `bounds_at_baseline` leads to
            # different `size_at_requested` (keeping same read resolution
            # but base image is of different scale)
            (read_level, _, _, post_read_scale,) = self._find_read_bounds_params(
                bounds_at_baseline, resolution=resolution, units=units
            )
        else:  # duplicated portion with VirtualReader, factoring out ?
            # Find parameters for optimal read
            (
                read_level,
                _,  # bounds_at_read_level,
                size_at_requested,
                post_read_scale,
            ) = self._find_read_bounds_params(
                bounds_at_baseline, resolution=resolution, units=units
            )
        glymur_wsi = self.glymur_wsi

        stride = 2 ** read_level

        im_region = utils.image.safe_padded_read(
            image=glymur_wsi,
            bounds=bounds_at_baseline,
            stride=stride,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
        )

        # Resize to correct scale if required
        if coord_space == "resolution":
            im_region = utils.transforms.imresize(
                img=im_region,
                output_size=size_at_requested,
                interpolation=interpolation,
            )
        else:
            im_region = utils.transforms.imresize(
                img=im_region,
                scale_factor=post_read_scale,
                output_size=size_at_requested,
                interpolation=interpolation,
            )

        return utils.transforms.background_composite(image=im_region)

    def _info(self):
        """JP2 meta data reader.

        Returns:
            WSIMeta: containing meta information

        """
        import glymur

        glymur_wsi = self.glymur_wsi
        box = glymur_wsi.box
        description = box[3].xml.find("description")
        m = re.search(r"(?<=AppMag = )\d\d", description.text)
        objective_power = np.int(m.group(0))
        image_header = box[2].box[0]
        slide_dimensions = (image_header.width, image_header.height)

        # Determine level_count
        cod = None
        for segment in glymur_wsi.codestream.segment:
            if isinstance(segment, glymur.codestream.CODsegment):
                cod = segment

        if cod is None:
            warnings.warn(
                "Metadata: JP2 codestream missing COD segment! "
                "Cannot determine number of decompositions (levels)"
            )
            level_count = 1
        else:
            level_count = cod.num_res

        level_downsamples = [2 ** n for n in range(level_count)]

        level_dimensions = [
            (int(slide_dimensions[0] / 2 ** n), int(slide_dimensions[1] / 2 ** n))
            for n in range(level_count)
        ]

        vendor = "Omnyx JP2"
        m = re.search(r"(?<=MPP = )\d*\.\d+", description.text)
        mpp_x = float(m.group(0))
        mpp_y = float(m.group(0))
        mpp = [mpp_x, mpp_y]

        return WSIMeta(
            file_path=self.input_path,
            axes="YXS",
            objective_power=objective_power,
            slide_dimensions=slide_dimensions,
            level_count=level_count,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            vendor=vendor,
            mpp=mpp,
            raw=self.glymur_wsi.box,
        )


class VirtualWSIReader(WSIReader):
    """Class for reading non-pyramidal images e.g., visual fields.

    Supported formats:

    - .jpg
    - .png
    - :class:`numpy.ndarray`

    This reader uses :func:`tiatoolbox.utils.image.sub_pixel_read` to
    allow reading low resolution images as if they are larger i.e. with
    'virtual' pyramid resolutions. This is useful for reading low
    resolution masks as if they were stretched to overlay a higher
    resolution WSI.

    Extra key-word arugments given to :func:`~WSIReader.read_region`
    and :func:`~WSIReader.read_bounds` will be passed to
    :func:`~tiatoolbox.utils.image.sub_pixel_read`.

    Attributes:
        img (:class:`numpy.ndarray`)
        mode (str)

    Args:
        input_img (str, pathlib.Path, ndarray): input path to WSI.
        info (WSIMeta): Metadata for the virtual wsi.
        mode (str): Mode of the input image. Default is 'rgb'. Allowed
            values are: rgb, bool.

    """

    def __init__(
        self,
        input_img: Union[str, pathlib.Path, np.ndarray],
        mpp: Optional[Tuple[Number, Number]] = None,
        power: Optional[Number] = None,
        info: WSIMeta = None,
        mode="rgb",
    ) -> None:
        super().__init__(
            input_img=input_img,
            mpp=mpp,
            power=power,
        )
        if mode.lower() not in ["rgb", "bool"]:
            raise ValueError("Invalid mode.")
        self.mode = mode.lower()
        if isinstance(input_img, np.ndarray):
            self.img = input_img
        else:
            self.img = utils.misc.imread(self.input_path)

        if info is not None:
            self._m_info = info

    def _info(self):
        """Visual Field meta data getter.

        This generates a WSIMeta object for the slide if none exists.
        There is 1 level with dimensions equal to the image and no
        mpp, objective power, or vendor data.


        Returns:
            WSIMeta: containing meta information.

        """
        param = WSIMeta(
            file_path=self.input_path,
            axes="YSX",
            objective_power=None,
            # align to XY to match with OpenSlide
            slide_dimensions=self.img.shape[:2][::-1],
            level_count=1,
            level_dimensions=(self.img.shape[:2][::-1],),
            level_downsamples=[1.0],
            vendor=None,
            mpp=None,
            raw=None,
        )
        if self._m_info is None:
            self._m_info = param
        return self._m_info

    def _find_params_from_baseline(self, location, baseline_read_size):
        """Convert read parameters from (virtual) baseline coordinates.

        Args:
            location (tuple(int)): Location of the location to read in
                (virtual) baseline coordinates.
            baseline_read_size (tuple(int)): Size of the region to read
                in (virtual) baseline coordinates.

        """
        baseline_size = np.array(self.info.slide_dimensions)
        image_size = np.array(self.img.shape[:2][::-1])
        size_ratio = image_size / baseline_size
        image_location = np.array(location, dtype=np.float32) * size_ratio
        read_size = np.array(baseline_read_size) * size_ratio
        return image_location, read_size

    def read_rect(
        self,
        location,
        size,
        resolution=0,
        units="level",
        interpolation="optimise",
        pad_mode="constant",
        pad_constant_values=0,
        coord_space="baseline",
        **kwargs,
    ):
        if coord_space == "resolution":
            return self._read_rect_at_resolution(
                location,
                size,
                resolution=resolution,
                units=units,
                interpolation=interpolation,
                pad_mode=pad_mode,
                pad_constant_values=pad_constant_values,
            )

        # Find parameters for optimal read
        (_, _, _, _, baseline_read_size,) = self.find_read_rect_params(
            location=location,
            size=size,
            resolution=resolution,
            units=units,
        )

        image_location, image_read_size = self._find_params_from_baseline(
            location, baseline_read_size
        )

        bounds = utils.transforms.locsize2bounds(
            location=image_location,
            size=image_read_size,
        )

        if interpolation in [None, "none"]:
            output_size = None
        else:
            output_size = size

        im_region = utils.image.sub_pixel_read(
            self.img,
            bounds,
            output_size=output_size,
            interpolation=interpolation,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
            read_kwargs=kwargs,
        )

        if self.mode == "rgb":
            return utils.transforms.background_composite(image=im_region)
        return im_region

    def read_bounds(
        self,
        bounds,
        resolution=0,
        units="level",
        interpolation="optimise",
        pad_mode="constant",
        pad_constant_values=0,
        coord_space="baseline",
        **kwargs,
    ):

        # convert from requested to `baseline`
        bounds_at_baseline = bounds
        if coord_space == "resolution":
            bounds_at_baseline = self._bounds_at_resolution_to_baseline(
                bounds, resolution, units
            )
            _, size_at_requested = utils.transforms.bounds2locsize(bounds)
            # * Find parameters for optimal read
            # don't use the `output_size` (`size_at_requested`) here
            # because the rounding error at `bounds_at_baseline` leads to
            # different `size_at_requested` (keeping same read resolution
            # but base image is of different scale)
            _, _, _, post_read_scale = self._find_read_bounds_params(
                bounds_at_baseline,
                resolution=resolution,
                units=units,
            )
        else:
            # * Find parameters for optimal read
            _, _, size_at_requested, post_read_scale = self._find_read_bounds_params(
                bounds_at_baseline,
                resolution=resolution,
                units=units,
            )

        location_at_read, size_at_read = self._find_params_from_baseline(
            *utils.transforms.bounds2locsize(bounds_at_baseline)
        )
        bounds_at_read = utils.transforms.locsize2bounds(location_at_read, size_at_read)

        if interpolation in [None, "none"]:
            interpolation = None

        im_region = utils.image.sub_pixel_read(
            self.img,
            bounds_at_read,
            output_size=size_at_requested,
            interpolation=interpolation,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
            read_kwargs=kwargs,
        )

        if coord_space == "resolution":
            # do this to enforce output size is as defined by input bounds
            im_region = utils.transforms.imresize(
                img=im_region, output_size=size_at_requested
            )
        else:
            im_region = utils.transforms.imresize(
                img=im_region,
                scale_factor=post_read_scale,
                output_size=size_at_requested,
            )

        if self.mode == "rgb":
            return utils.transforms.background_composite(image=im_region)
        return im_region


class ArrayView:
    """An object for viewing a zarr array with a different index ordering.

    Used to allow YXS index order reads for arrays with axes in other
    orders such as SYX. Currently supported axes are:
    - YXS
    - SYX

    """

    def __init__(self, array: zarr.Array, axes: str) -> None:
        """Initialise the view object.

        Args:
            array (zarr.Array): Zarr Array to read from.
            axes (str): Axes ordering string. Allowed values are YXS and SYX.

        """
        self.array = array
        self.axes = axes
        self._shape = dict(zip(self.axes, self.array.shape))

    @property
    def shape(self):
        return tuple(self._shape[c] for c in "YXS")

    def __getitem__(self, index):
        # Normalize to a tuple of length = len(self.axes)
        if not isinstance(index, tuple):
            index = (index,)
        while len(index) < len(self.axes):
            index = (*index, slice(None))

        if self.axes == "YXS":
            return self.array[index]
        if self.axes == "SYX":
            y, x, s = index
            index = (s, y, x)
            return np.rollaxis(self.array[index], 0, 3)
        raise Exception("Unsupported axes")


class TIFFWSIReader(WSIReader):
    def __init__(
        self,
        input_img: Union[str, pathlib.Path, np.ndarray],
        mpp: Optional[Tuple[Number, Number]] = None,
        power: Optional[Number] = None,
        series="auto",
        cache_size=2 ** 28,
    ) -> None:
        super().__init__(input_img=input_img, mpp=mpp, power=power)
        self.tiff = tifffile.TiffFile(self.input_path)
        self._axes = self.tiff.pages[0].axes
        if not any([self.tiff.is_svs, self.tiff.is_ome]):
            raise ValueError("Unsupported TIFF WSI format.")

        self.series_n = series
        # Find the largest series if series="auto"
        if self.tiff.series is None or len(self.tiff.series) == 0:  # pragma: no cover
            raise Exception("TIFF does not contain any valid series.")
        if self.series_n == "auto":
            all_series = self.tiff.series or []
            series_areas = [
                np.prod(self._shape_channels_last(np.array(s.pages[0].shape))[:2])
                for s in all_series  # skipcq: PYL-E1133
            ]
            self.series_n = np.argmax(series_areas)
        self._tiff_series = self.tiff.series[self.series_n]
        self._zarr_store = tifffile.imread(
            self.input_path, series=self.series_n, aszarr=True
        )
        self._zarr_lru_cache = zarr.LRUStoreCache(self._zarr_store, max_size=cache_size)
        self._zarr_group = zarr.open(self._zarr_lru_cache)
        if not isinstance(self._zarr_group, zarr.hierarchy.Group):
            group = zarr.hierarchy.group()
            group[0] = self._zarr_group
            self._zarr_group = group
        self.level_arrays = {
            int(key): ArrayView(array, axes=self.info.axes)
            for key, array in self._zarr_group.items()
        }
        # Using the zarr array view method gives a ValueError
        # self.zarr_views = zarr.hierarchy.group()
        # for key, array in self.zarr_group.items():
        #     self.zarr_views[key] = array.view(self._shape_channels_last(array.shape))

    def _shape_channels_last(self, shape):
        """Make a level shape tuple in YXS order.

        Args:
            shape (tuple(int)): Input shape tuple.

        Returns:
            Shape in YXS order.

        """
        if self._axes == "YXS":
            return shape
        if self._axes == "SYX":
            return np.roll(shape, -1)
        raise Exception("Unsupported axes")

    def _parse_svs_metadata(self) -> dict:
        """Extract SVS specific metadata.

        Returns:
            dict: Dictionary of kwargs for WSIMeta.

        """
        raw = {}
        mpp = None
        objective_power = None
        vendor = "Aperio"

        description = self.tiff.pages[0].description
        raw["Description"] = description
        parts = description.split("|")
        description_headers, key_value_pairs = parts[0], parts[1:]
        description_headers = description_headers.split(";")

        software, photometric_info = description_headers[0].splitlines()
        raw["Software"] = software
        raw["Photometric Info"] = photometric_info

        def parse_svs_tag(string: str) -> Tuple[str, Union[Number, str]]:
            """Parse SVS key-value string.

            Infers types of data by trial and error with a fallback to
            the original string type.

            Args:
                string (str): key-value string in SVS format: "key=value".

            Returns:
                tuple: Key-value pair.

            """
            pair = string.split("=")
            if len(pair) != 2:
                raise ValueError(
                    "Invalid metadata. Expected string of the format 'key=value'."
                )
            key, value_string = pair
            key = key.strip()
            value_string = value_string.strip()
            value = value_string.strip()

            def us_date(string: str) -> datetime:
                return datetime.strptime(string, r"%m/%d/%y")

            def time(string: str) -> datetime:
                return datetime.strptime(string, r"%H:%M:%S")

            casting_prescedence = [us_date, time, int, float]
            value = value_string
            for cast in casting_prescedence:
                try:
                    value = cast(value_string)
                    return key, value
                except ValueError:
                    continue

            return key, value

        svs_tags = dict(parse_svs_tag(string) for string in key_value_pairs)
        raw["SVS Tags"] = svs_tags
        mpp = svs_tags.get("MPP")
        if mpp is not None:
            mpp = [mpp] * 2
        objective_power = svs_tags.get("AppMag")

        return {
            "objective_power": objective_power,
            "vendor": vendor,
            "mpp": mpp,
            "raw": raw,
        }

    def _parse_ome_metadata(self) -> dict:
        # The OME-XML should be in each IFD but is optional. It must be
        # present in the first IFD. We simply get the description from
        # the first IFD.
        from defusedxml.ElementTree import fromstring as et_from_string

        description = self.tiff.pages[0].description
        xml = et_from_string(description)
        namespaces = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
        xml_series = xml.findall("ome:Image", namespaces)[self.series_n]

        raw = {
            "Description": description,
            "OME-XML": xml,
        }

        objective_power = None
        mpp = None
        vendor = None

        xml_pixels = xml_series.find("ome:Pixels", namespaces)
        mppx = xml_pixels.attrib.get("PhysicalSizeX")
        mppy = xml_pixels.attrib.get("PhysicalSizeY")
        if mppx is not None and mppy is not None:
            mpp = [mppx, mppy]
        elif mppx is not None or mppy is not None:
            warnings.warn("Only one MPP value found. Using it for both X  and Y.")
            mpp = [mppx or mppy] * 2

        instrument_ref = xml_series.find("ome:InstrumentRef", namespaces)
        objective_settings = xml_series.find("ome:ObjectiveSettings", namespaces)
        instrument_ref_id = instrument_ref.attrib["ID"]
        objective_settings_id = objective_settings.attrib["ID"]
        instruments = {
            instrument.attrib["ID"]: instrument
            for instrument in xml.findall("ome:Instrument", namespaces)
        }
        objectives = {
            (instrument_id, objective.attrib["ID"]): objective
            for instrument_id, instrument in instruments.items()
            for objective in instrument.findall("ome:Objective", namespaces)
        }

        try:
            objective = objectives[(instrument_ref_id, objective_settings_id)]
            objective_power = float(objective.attrib.get("NominalMagnification"))
        except KeyError:
            raise KeyError("No matching Instrument for image InstrumentRef in OME-XML.")

        return {
            "objective_power": objective_power,
            "vendor": vendor,
            "mpp": mpp,
            "raw": raw,
        }

    def _info(self):
        """TIFF metadata constructor.

        Returns:
            WSIMeta: Containing metadata.

        """
        level_count = len(self._zarr_group)
        level_dimensions = [
            np.array(self._shape_channels_last(p.shape)[:2][::-1])
            for p in self._zarr_group.values()
        ]
        slide_dimensions = level_dimensions[0]
        level_downsamples = [(level_dimensions[0] / x)[0] for x in level_dimensions]
        # The tags attribute object will not pickle or deepcopy,
        # so a copy with only python values or tifffile enums is made.
        tiff_tags = {
            code: {
                "code": code,
                "value": tag.value,
                "name": tag.name,
                "count": tag.count,
                "type": tag.dtype,
            }
            for code, tag in self.tiff.pages[0].tags.items()
        }

        if self.tiff.is_svs:
            filetype_params = self._parse_svs_metadata()
        if self.tiff.is_ome:
            filetype_params = self._parse_ome_metadata()
        filetype_params["raw"]["TIFF Tags"] = tiff_tags

        return WSIMeta(
            file_path=self.input_path,
            slide_dimensions=slide_dimensions,
            axes=self._axes,
            level_count=level_count,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            **filetype_params,
        )

    def read_rect(
        self,
        location,
        size,
        resolution=0,
        units="level",
        interpolation="optimise",
        pad_mode="constant",
        pad_constant_values=0,
        coord_space="baseline",
        **kwargs,
    ):
        if coord_space == "resolution":
            return self._read_rect_at_resolution(
                location,
                size,
                resolution=resolution,
                units=units,
                interpolation=interpolation,
                pad_mode=pad_mode,
                pad_constant_values=pad_constant_values,
            )

        # Find parameters for optimal read
        (
            read_level,
            _,
            _,
            post_read_scale,
            baseline_read_size,
        ) = self.find_read_rect_params(
            location=location,
            size=size,
            resolution=resolution,
            units=units,
        )

        bounds = utils.transforms.locsize2bounds(
            location=location, size=baseline_read_size
        )
        im_region = utils.image.safe_padded_read(
            image=self.level_arrays[read_level],
            bounds=bounds,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
        )

        im_region = utils.transforms.imresize(
            img=im_region,
            scale_factor=post_read_scale,
            output_size=size,
            interpolation=interpolation,
        )

        return utils.transforms.background_composite(image=im_region)

    def read_bounds(
        self,
        bounds,
        resolution=0,
        units="level",
        interpolation="optimise",
        pad_mode="constant",
        pad_constant_values=0,
        coord_space="baseline",
        **kwargs,
    ):
        bounds_at_baseline = bounds
        if coord_space == "resolution":
            bounds_at_baseline = self._bounds_at_resolution_to_baseline(
                bounds, resolution, units
            )
            _, size_at_requested = utils.transforms.bounds2locsize(bounds)
            # don't use the `output_size` (`size_at_requested`) here
            # because the rounding error at `bounds_at_baseline` leads to
            # different `size_at_requested` (keeping same read resolution
            # but base image is of different scale)
            (read_level, _, _, post_read_scale,) = self._find_read_bounds_params(
                bounds_at_baseline, resolution=resolution, units=units
            )
        else:  # duplicated portion with VirtualReader, factoring out ?
            # Find parameters for optimal read
            (
                read_level,
                _,
                size_at_requested,
                post_read_scale,
            ) = self._find_read_bounds_params(
                bounds_at_baseline, resolution=resolution, units=units
            )

        im_region = utils.image.sub_pixel_read(
            image=self.level_arrays[read_level],
            bounds=bounds_at_baseline,
            output_size=size_at_requested,
            interpolation=interpolation,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
            read_kwargs=kwargs,
        )

        if coord_space == "resolution":
            # do this to enforce output size is as defined by input bounds
            im_region = utils.transforms.imresize(
                img=im_region, output_size=size_at_requested
            )
        else:
            im_region = utils.transforms.imresize(
                img=im_region,
                scale_factor=post_read_scale,
                output_size=size_at_requested,
            )

        return im_region


def get_wsireader(input_img):
    """Return an appropriate :class:`.WSIReader` object.

    Args:
        input_img (str, pathlib.Path, :class:`numpy.ndarray`, or :obj:WSIReader):
          Input to create a WSI object from.
          Supported types of input are: `str` and `pathlib.Path` which point to
          the location on the disk where image is stored, :class:`numpy.ndarray`
          in which the input image in the form of numpy array (HxWxC) is stored,
          or :obj:WSIReader which is an already created tiatoolbox WSI handler.
          In the latter case, the function directly passes the input_imge to the
          output.

    Returns:
        WSIReader: an object with base :class:`.WSIReader` as base class.

    Examples:
        >>> from tiatoolbox.wsicore.wsireader import get_wsireader
        >>> wsi = get_wsireader(input_img="./sample.svs")

    """
    warnings.warn(
        "get_wsireader is deprecated. Please use WSIReader.open instead",
        DeprecationWarning,
    )
    return WSIReader.open(input_img)
