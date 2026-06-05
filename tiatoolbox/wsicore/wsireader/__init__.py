"""WSI Reader module with support for multiple WSI formats."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import UTC, datetime
from numbers import Number

import fsspec
import matplotlib.colors as mcolors
import zarr
from zarr.experimental.cache_store import CacheStore

from tiatoolbox.wsicore.wsimeta import WSIMeta

from .annotation_store import AnnotationStoreReader

# Import base classes and utility functions
from .base import (
    ArrayView,
    VirtualWSIReader,
    WSIReader,
    _handle_tiff_wsi,
    _handle_virtual_wsi,
    fix_mangled_url_by_pathlib,
    is_dicom,
    is_ngff,
    is_tiled_tiff,
    is_zarr,
)
from .dicom import DICOMWSIReader
from .fsspec_json import FsspecJsonWSIReader
from .jp2 import JP2WSIReader
from .ngff import NGFFWSIReader

# Import reader classes
from .openslide import OpenSlideWSIReader
from .tiff import TIFFWSIReader, TIFFWSIReaderDelegate
from .transformed import TransformedWSIReader

__all__ = [
    # Standard library imports (re-exported for backward compatibility)
    "CacheStore",
    "Number",
    "UTC",
    "datetime",
    "defaultdict",
    "fsspec",
    "json",
    "logging",
    "mcolors",
    "re",
    "zarr",
    # Base class and utilities
    "WSIReader",
    "WSIMeta",
    "is_dicom",
    "is_ngff",
    "is_tiled_tiff",
    "is_zarr",
    "fix_mangled_url_by_pathlib",
    "_handle_virtual_wsi",
    "_handle_tiff_wsi",
    # Reader classes
    "OpenSlideWSIReader",
    "JP2WSIReader",
    "VirtualWSIReader",
    "ArrayView",
    "TIFFWSIReader",
    "TIFFWSIReaderDelegate",
    "FsspecJsonWSIReader",
    "DICOMWSIReader",
    "NGFFWSIReader",
    "AnnotationStoreReader",
    "TransformedWSIReader",
]
