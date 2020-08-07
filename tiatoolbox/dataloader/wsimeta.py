"""WSIMeta to save metadata information for WSIs"""
import warnings
from pathlib import Path

import numpy as np


class WSIMeta:
    """Whole slide image meta data class"""

    def __init__(
        self,
        input_dir,
        file_name,
        objective_power=None,
        slide_dimensions=None,
        level_count=1,
        level_dimensions=None,
        level_downsamples=None,
        vendor=None,
        mpp_x=None,
        mpp_y=None,
        raw=None,
    ):
        self.input_dir = Path(input_dir)
        self.file_name = file_name
        self.objective_power = float(objective_power)
        self.slide_dimensions = [int(x) for x in slide_dimensions]
        self.level_count = int(level_count)
        self.level_dimensions = [int(x) for x in level_dimensions]
        self.level_downsamples = [float(x) for x in level_downsamples]
        self.vendor = vendor
        self.mpp_x = float(mpp_x)
        self.mpp_y = float(mpp_y)
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

        if len(self.level_dimensions) != self.level_count:
            warnings.warn("Length of slide_dimensions != level_count")
            passed = False

        if len(self.level_downsamples) != self.level_count:
            warnings.warn("Length of level_downsamples != level_count")
            passed = False

        if self.raw is None:
            warnings.warn("Raw data is None")
            passed = False

        if all(x is None for x in [self.objective_power, self.mpp_x, self.mpp_y]):
            warnings.warn("Unknown magnification (no objective_power, mpp_x, or mpp_y)")

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
