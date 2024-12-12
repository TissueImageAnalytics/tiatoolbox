"""This module defines a dataclass which holds metadata about a WSI.

With this class, metadata is in a normalized consistent format
which is quite useful when working with many different WSI formats.
The raw metadata is also preserved and accessible via a dictionary. The
format of this dictionary may vary between WSI formats.

"""

from __future__ import annotations

from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tiatoolbox import logger

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    from tiatoolbox.typing import Resolution, Units


class WSIMeta:
    """Whole slide image metadata class.

    Args:
            slide_dimensions (int, int):
                Tuple containing the width and height of the WSI. These
                are for the baseline (full resolution) image if the WSI
                is a pyramid or multi-resolution.
            level_dimensions (list):
                A list of dimensions for each level of the pyramid or
                for each resolution in the WSI.
            objective_power (float, optional):
                The power of the objective lens used to create the
                image.
            level_count: (int, optional):
                The number of levels or resolutions in the WSI. If not
                given this is assigned len(level_dimensions). Defaults
                to None.
            level_downsamples (:obj:`list` of :obj:`float`):
                List of scale values which describe how many times
                smaller the current level is compared with the baseline.
            vendor (str, optional):
                Scanner vendor/manufacturer description.
            mpp (float, float, optional):
                Microns per pixel.
            file_path (Path, optional):
                Path to the corresponding WSI file.
            raw (dict, optional):
                Dictionary of unprocessed metadata extracted from the
                WSI format. For JPEG-2000 images this contains an xml
                object under the key "xml".

    Attributes:
        slide_dimensions (tuple(int)):
            Tuple containing the width and height of the WSI. These are
            for the baseline (full resolution) image if the WSI is a
            pyramid or multi-resolution. Required.
        axes (str):
            Axes ordering of the image. This is most relevant for
            OME-TIFF images where the axes ordering can vary. For most
            images this with be "YXS" i.e. the image is store in the
            axis order of Y coordinates first, then X coordinates, and
            colour channels last.
        level_dimensions (list):
            A list of dimensions for each level of the pyramid or for
            each resolution in the WSI. Defaults to [slide_dimension].
        objective_power (float):
            The magnification power of the objective lens used to scan
            the image. Not always present or accurate. Defaults to None.
        level_count: (int):
            The number of levels or resolutions in the WSI. If not given
            this is assigned len(level_dimensions). Defaults to
            len(level_dimensions).
        level_downsamples (:obj:`list` of :obj:`float`):
            List of scale values which describe how many times smaller
            the current level is compared with the baseline. Defaults to
            (1,).
        vendor (str):
            Scanner vendor/manufacturer description.
        mpp (float, float, optional):
            Microns per pixel. Derived from objective power and sensor
            size. Not always present or accurate. Defaults to None.
        file_path (Path):
            Path to the corresponding WSI file. Defaults to None.
        raw (dict):
            Dictionary of unprocessed metadata extracted from the WSI
            format. For JP2 images this contains an xml object under the
            key "xml". Defaults to empty dictionary.

    """

    _valid_axes_characters = "YXSTZ"

    def __init__(
        self: WSIMeta,
        slide_dimensions: tuple[int, int],
        axes: str,
        level_dimensions: Sequence[tuple[int, int]] | None = None,
        objective_power: float | None = None,
        level_count: int | None = None,
        level_downsamples: Sequence[float] | None = (1,),
        vendor: str | None = None,
        mpp: Sequence[float] | None = None,
        file_path: Path | None = None,
        raw: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize WSIMeta."""
        self.axes = axes
        self.objective_power = float(objective_power) if objective_power else None
        self.slide_dimensions = tuple(int(x) for x in slide_dimensions)
        self.level_dimensions = (
            tuple((int(w), int(h)) for w, h in level_dimensions)
            if level_dimensions is not None
            else [self.slide_dimensions]
        )
        self.level_downsamples = (
            [float(x) for x in level_downsamples]
            if level_downsamples is not None
            else None
        )
        self.level_count = (
            int(level_count) if level_count is not None else len(self.level_dimensions)
        )
        self.vendor = str(vendor)
        self.mpp = np.array([float(x) for x in mpp]) if mpp is not None else None
        self.file_path = Path(file_path) if file_path is not None else None
        self.raw = raw if raw is not None else None

        self.validate()

    def validate(self: WSIMeta) -> bool:
        """Validate passed values and cast to Python types.

        Metadata values are often given as strings and must be
        parsed/cast to the appropriate python type e.g. "3.14" to 3.14
        etc.

        Returns:
            bool:
                True is validation passed, False otherwise.

        """
        passed = True

        # Fatal conditions: Should return False if not True

        if len(set(self.axes) - set(self._valid_axes_characters)) > 0:
            logger.warning(
                "Axes contains invalid characters. Valid characters are %s.",
                self._valid_axes_characters,
            )
            passed = False

        if self.level_count < 1:
            logger.warning("Level count is not a positive integer.")
            passed = False

        if self.level_dimensions is None:
            logger.warning("'level_dimensions' is None.")
            passed = False
        elif len(self.level_dimensions) != self.level_count:
            logger.warning("Length of level dimensions != level count")
            passed = False

        if self.level_downsamples is None:
            logger.warning("Level downsamples is None.")
            passed = False
        elif len(self.level_downsamples) != self.level_count:
            logger.warning("Length of level downsamples != level count")
            passed = False

        # Non-fatal conditions: Raise warning only, do not fail validation

        if self.raw is None:
            logger.warning("Raw data is None.")

        if all(x is None for x in [self.objective_power, self.mpp]):
            logger.warning("Unknown scale (no objective_power or mpp)")

        return passed

    def level_downsample(
        self: WSIMeta,
        level: float,
    ) -> float:
        """Get the downsample factor for a level.

        For non-integer values of `level`, the downsample factor is
        linearly interpolated between from the downsample factors of the
        level below and the level above.

        Args:
            level (float):
                Level to get downsample factor for.

        Returns:
            float:
                Downsample factor for the given level.

        """
        level_downsamples = self.level_downsamples
        if isinstance(level, int) or int(level) == level:
            # Return the downsample for the level
            return level_downsamples[int(level)]
        # Linearly interpolate between levels
        floor = int(np.floor(level))
        ceil = int(np.ceil(level))
        floor_downsample = level_downsamples[floor]
        ceil_downsample = level_downsamples[ceil]
        return np.interp(level, [floor, ceil], [floor_downsample, ceil_downsample])

    def relative_level_scales(
        self: WSIMeta,
        resolution: Resolution,
        units: Units,
    ) -> list[np.ndarray]:
        """Calculate scale of each level in the WSI relative to given resolution.

        Find the relative scale of each image pyramid / resolution level
        of the WSI relative to the given resolution and units.

        Values > 1 indicate that the level has a larger scale than the
        target and < 1 indicates that it is smaller.

        Args:
            resolution (Resolution):
                Scale to calculate relative to units.
            units (Units):
                Units of the scale. Allowed values are: `"mpp"`,
                `"power"`, `"level"`, `"baseline"`. Baseline refers to
                the largest resolution in the WSI (level 0).

        Raises:
            ValueError:
                Missing MPP metadata.
            ValueError:
                Missing objective power metadata.
            ValueError:
                Invalid units.

        Returns:
            list:
                Scale for each level relative to the given scale and
                units.

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
            >>> print(wsi.info.relative_level_scales(0.5, "mpp"))
            [array([0.91282519, 0.91012514]), array([1.82565039, 1.82025028]) ...

            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
            >>> print(wsi.info.relative_level_scales(0.5, "baseline"))
            [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

        """
        if units not in ("mpp", "power", "level", "baseline"):
            msg = "Invalid units"
            raise ValueError(msg)

        level_downsamples = self.level_downsamples

        def np_pair(x: Number | np.array) -> np.ndarray:
            """Ensure input x is a numpy array of length 2."""
            # If one number is given, the same value is used for x and y
            if isinstance(x, Number):
                return np.array([x] * 2)
            return np.array(x)

        if units == "level":
            if resolution >= len(level_downsamples):
                msg = (
                    f"Target scale level {resolution} > "
                    f"number of levels {len(level_downsamples)} in WSI"
                )
                raise ValueError(
                    msg,
                )
            base_scale, resolution = 1, self.level_downsample(resolution)

        resolution = np_pair(resolution)

        if units == "mpp":
            if self.mpp is None:
                msg = "MPP is None. Cannot determine scale in terms of MPP."
                raise ValueError(msg)
            base_scale = self.mpp

        if units == "power":
            if self.objective_power is None:
                msg = (
                    "Objective power is None. "
                    "Cannot determine scale in terms of objective power.",
                )
                raise ValueError(
                    msg,
                )
            base_scale, resolution = 1 / self.objective_power, 1 / resolution

        if units == "baseline":
            base_scale, resolution = 1, 1 / resolution

        return [
            (base_scale * downsample) / resolution for downsample in level_downsamples
        ]

    def as_dict(self: WSIMeta) -> dict:
        """Convert WSIMeta to dictionary of Python types.

        Returns:
            dict:
                Whole slide image metadata as dictionary.

        """
        mpp = (self.mpp, self.mpp) if self.mpp is None else tuple(self.mpp)

        return {
            "objective_power": self.objective_power,
            "slide_dimensions": self.slide_dimensions,
            "level_count": self.level_count,
            "level_dimensions": self.level_dimensions,
            "level_downsamples": self.level_downsamples,
            "vendor": self.vendor,
            "mpp": mpp,
            "file_path": self.file_path,
            "axes": self.axes,
        }
