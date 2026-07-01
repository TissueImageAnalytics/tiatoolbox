"""WSI Reader module with support for multiple WSI formats."""

from __future__ import annotations

from .base import WSIReader
from .dicom import DICOMWSIReader

__all__ = ["DICOMWSIReader", "WSIReader"]
