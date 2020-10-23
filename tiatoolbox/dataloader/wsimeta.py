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
# The Original Code is Copyright (C) 2020, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""WSIMeta to save metadata information for WSIs."""
import warnings
from pathlib import Path
from typing import Sequence, Tuple, Optional, Mapping

import numpy as np


class WSIMeta:
    """Whole slide image metadata class.

    Attributes:
        objective_power (float): objective power at which the slide was scanned.
        slide_dimensions (tuple): Slide dimensions at level 0 in terms of width and
         height.
        level_dimensions (list): A list of image dimensions at each downsample level of
         the WSI.
        level_downsamples (list): A list of downsample factors for each level of WSI.
        level_count (int): The number of available levels in the whole slide image.
        vendor (str): vendor of the microscope which scanned the image.
        mpp (np.array): microns per pixel at base resolution.
        file_path (str, pathlib.Path): Path to input image.
        raw (xml, dict): raw meta data from Openslide or glymur

    Args:
        objective_power (float): objective power at which the slide was scanned.
        slide_dimensions (tuple): Slide dimensions at level 0 as [width, height].
        level_dimensions (list): A list of image dimensions at each downsample level of
         the WSI.
        level_downsamples (list): A list of downsample factors for each level of WSI.
        level_count (int): The number of available levels in the whole slide image.
        vendor (str): vendor of the microscope which scanned the image.
        mpp (np.array): microns per pixel at base resolution.
        file_path (str, pathlib.Path): Path to input image.
        raw (xml, dict): raw meta data from Openslide or glymur

    """

    def __init__(
        self,
        slide_dimensions: Tuple[int, int],
        level_dimensions: Optional[Sequence[Tuple[int, int]]] = None,
        objective_power: Optional[float] = None,
        level_count: Optional[int] = None,
        level_downsamples: Optional[Sequence[float]] = (1,),
        vendor: Optional[str] = None,
        mpp: Optional[Sequence[float]] = None,
        file_path: Optional[Path] = None,
        raw: Optional[Mapping[str, str]] = None,
    ):
        self.objective_power = float(objective_power) if objective_power else None
        self.slide_dimensions = tuple([int(x) for x in slide_dimensions])
        self.level_dimensions = (
            tuple([(int(w), int(h)) for w, h in level_dimensions])
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
        self.vendor = vendor
        self.mpp = np.array(mpp)
        self.file_path = file_path
        self.raw = raw if raw is not None else None

        self.validate()

    def validate(self):
        """
        Validate passed values and cast to Python types.
        Metadata values are often given as strings and must be parsed/cast to the
        appropriate python type e.g. "3.14" to 3.14 etc.

        Args:
            self (WSIMeta):

        Returns:
            bool: True is validation passed, False otherwise.
        """
        passed = True

        # Fatal conditions: Should return False if not True

        if self.level_count < 1:
            warnings.warn("Level count is not a positive integer")
            passed = False

        if self.level_dimensions is None:
            warnings.warn("level_dimensions is None")
            passed = False
        elif len(self.level_dimensions) != self.level_count:
            warnings.warn("Length of level dimensions != level count")
            passed = False

        if self.level_downsamples is None:
            warnings.warn("Level downsamples is None")
            passed = False
        elif len(self.level_downsamples) != self.level_count:
            warnings.warn("Length of level downsamples != level count")
            passed = False

        # Non-fatal conditions: Raise warning only, do not fail validation

        if self.raw is None:
            warnings.warn("Raw data is None")

        if all(x is None for x in [self.objective_power, self.mpp]):
            warnings.warn("Unknown scale (no objective_power or mpp)")

        return passed

    def as_dict(self):
        """Convert WSIMeta to dictionary to assist print and save in various formats.

        Args:
            self (WSIMeta):

        Returns:
            dict: whole slide image meta data as dictionary

        """
        param = {
            "objective_power": self.objective_power,
            "slide_dimensions": self.slide_dimensions,
            "level_count": self.level_count,
            "level_dimensions": self.level_dimensions,
            "level_downsamples": self.level_downsamples,
            "vendor": self.vendor,
            "mpp": tuple(self.mpp),
            "file_path": self.file_path,
        }
        return param
