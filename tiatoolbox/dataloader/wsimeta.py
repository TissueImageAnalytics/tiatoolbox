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

"""This module defines a dataclass which holds metadata about a WSI.

With this class, metadata is in a normalised consistent format
which is quite useful when working with many different WSI formats.
The raw metadata is also preserved and accessible via a dictionary. The
format of this dictionary may vary between WSI formats.
"""
import warnings
from pathlib import Path
from typing import Sequence, Tuple, Optional, Mapping

import numpy as np


class WSIMeta:
    """Whole slide image metadata class.

    Attributes:
        slide_dimensions (:obj:`tuple` of :obj:`int`): Tuple containing the width and
            height of the WSI. These are for the baseline (full resolution)
            image if the WSI is a pyramid or multi-resoltion. Required.
        level_dimensions (list): A list of dimensions for each level of the
            pyramid or for each resolution in the WSI. Defaults to
            [slide_dimension].
        objective_power (float): The magnification power of the objective lens
            used to scan the image. Not always present or accurate.
            Defaults to None.
        level_count: (int): The number of levels or resolutions
            in the WSI. If not given this is assigned
            len(level_dimensions). Defaults to len(level_dimensions).
        level_downsamples (:obj:`list` of :obj:`float`): List of scale
            values which describe how many times smaller the current level
            is compared with the baseline. Defaults to (1,).
        vendor (str): Scanner vendor/manufacturer description.
        mpp (float, float, optional): Microns per pixel. Derived from objective
            power and sensor size. Not always present or accurate.
            Defaults to None.
        file_path (Path): Path to the corresponding WSI file. Defaults to None.
        raw (dict): Dictionary of unprocessed metadata extracted
            from the WSI format. For JP2 images this contains an xml object
            under the key "xml". Defaults to empty dictionary.
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
        """Initialise the metadata object.

        Args:
            slide_dimensions (int, int): Tuple containing the width and
                height of the WSI. These are for the baseline (full resolution)
                image if the WSI is a pyramid or multi-resoltion.
            level_dimensions (list): A list of dimensions for each level of the
                pyramid or for each resolution in the WSI.
            objective_power (float, optional): The power of the objective lens
                used to create the image.
            level_count: (int, optional): The number of levels or resolutions
                in the WSI. If not given this is assigned
                len(level_dimensions). Defaults to None.
            level_downsamples (:obj:`list` of :obj:`float`): List of scale
                values which describe how many times smaller the current level
                is compared with the baseline.
            vendor (str, optional): Scanner vendor/manufacturer description.
            mpp (float, float, optional): Microns per pixel.
            file_path (Path, optional): Path to the corresponding WSI file.
            raw (dict, optional): Dictionary of unprocessed metadata extracted
                from the WSI format. For JPEG-2000 images this contains an xml
                object under the key "xml".
        """
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
        self.vendor = str(vendor)
        self.mpp = np.array([float(x) for x in mpp]) if mpp is not None else None
        self.file_path = Path(file_path) if file_path is not None else None
        self.raw = raw if raw is not None else None

        self.validate()

    def validate(self):
        """Validate passed values and cast to Python types.
        Metadata values are often given as strings and must be parsed/cast to the
        appropriate python type e.g. "3.14" to 3.14 etc.

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
        """Convert WSIMeta to dictionary of Python types.

        Returns:
            dict: whole slide image meta data as dictionary

        """
        if self.mpp is None:
            mpp = (self.mpp, self.mpp)
        else:
            mpp = tuple(self.mpp)
        param = {
            "objective_power": self.objective_power,
            "slide_dimensions": self.slide_dimensions,
            "level_count": self.level_count,
            "level_dimensions": self.level_dimensions,
            "level_downsamples": self.level_downsamples,
            "vendor": self.vendor,
            "mpp": mpp,
            "file_path": self.file_path,
        }
        return param
