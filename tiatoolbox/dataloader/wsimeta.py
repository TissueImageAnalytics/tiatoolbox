"""WSIMeta to save metadata information for WSIs"""
import warnings
from pathlib import Path
from typing import Sequence, Tuple, Optional, Mapping, Union

import numpy as np


class WSIMeta:
    """Whole slide image metadata class"""

    def __init__(
        self,
        input_dir: Union[str, Path],
        file_name: Union[str, Path],
        slide_dimensions: Tuple[int, int],
        objective_power: Optional[float] = None,
        level_count: int = 1,
        level_dimensions: Sequence[Tuple[int, int]] = None,
        level_downsamples: Sequence[float] = None,
        vendor: Optional[str] = None,
        mpp_x: Optional[float] = None,
        mpp_y: Optional[float] = None,
        raw: Mapping[str, str] = None,
    ):
        self.input_dir = Path(input_dir)
        self.file_name = file_name
        self.objective_power = float(objective_power) if objective_power else None
        self.slide_dimensions = (
            [int(x) for x in slide_dimensions] if slide_dimensions else None
        )
        self.level_count = int(level_count)
        self.level_dimensions = (
            [(int(w), int(h)) for w, h in level_dimensions]
            if level_dimensions
            else None
        )
        self.level_downsamples = (
            [float(x) for x in level_downsamples] if level_downsamples else None
        )
        self.vendor = vendor
        self.mpp_x = float(mpp_x) if mpp_x else None
        self.mpp_y = float(mpp_y) if mpp_y else None
        self.raw = raw

        self.validate()

    @property
    def filepath(self):
        return self.input_dir / self.file_name

    @property
    def mpp(self):
        return np.array([self.mmpx, self.mppy])

    def validate(self):
        """
        Validate passed values and cast to Python types

        Metadata values are often given as strings and must be parsed/cast to the
        appropriate python type e.g. "3.14" to 3.14 etc.

        Args:
            self (WSIMeta):

        Returns:
            bool: True is validation passed, False otherwise.
        """
        passed = True

        if not self.input_dir.exists():
            warnings.warn(f"Directory does not exist: {self.input_dir}")
            passed = False

        if not self.filepath.exists():
            warnings.warn(f"File does not exist: {self.filepath}")
            passed = False

        if self.level_count < 1:
            warnings.warn("Level count is not a positive integer")
            passed = False

        if self.level_dimensions and len(self.level_dimensions) != self.level_count:
            warnings.warn("Length of slide_dimensions != level_count")
            passed = False

        if self.level_downsamples and len(self.level_downsamples) != self.level_count:
            warnings.warn("Length of level_downsamples != level_count")
            passed = False

        if self.raw is None:
            warnings.warn("Raw data is None")
            passed = False

        if all(x is None for x in [self.objective_power, self.mpp_x, self.mpp_y]):
            warnings.warn("Unknown scale (no objective_power, mpp_x, or mpp_y)")

        return passed

    def as_dict(self):
        """
        Converts WSIMeta to dictionary to assist print and save in various formats

        Args:
            self (WSIMeta):

        Returns:
            dict: whole slide image meta data as dictionary

        """
        param = {
            "input_dir": self.input_dir,
            "objective_power": self.objective_power,
            "slide_dimension": self.slide_dimension,
            "level_count": self.level_count,
            "level_dimensions": self.level_dimensions,
            "level_downsamples": self.level_downsamples,
            "vendor": self.vendor,
            "mpp_x": self.mpp_x,
            "mpp_y": self.mpp_y,
            "file_name": self.file_name,
        }
        return param
