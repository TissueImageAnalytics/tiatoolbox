"""This module defines classes which can read image data from WSI formats.

This module provides backward compatibility by importing from the refactored
wsireader submodule. All functionality has been extracted into separate modules
for better maintainability.
"""

from __future__ import annotations

# Import and re-export everything from the wsireader submodule
from .wsireader import (
    UTC,
    AnnotationStoreReader,
    CacheStore,
    DICOMWSIReader,
    FsspecJsonWSIReader,
    JP2WSIReader,
    NGFFWSIReader,
    Number,
    OpenSlideWSIReader,
    TIFFWSIReader,
    TIFFWSIReaderDelegate,
    TransformedWSIReader,
    VirtualWSIReader,
    WSIMeta,
    WSIReader,
    _handle_tiff_wsi,
    _handle_virtual_wsi,
    datetime,
    defaultdict,
    fix_mangled_url_by_pathlib,
    fsspec,
    is_dicom,
    is_ngff,
    is_tiled_tiff,
    is_zarr,
    json,
    logging,
    mcolors,
    re,
    zarr,
)
from tiatoolbox.wsicore.wsireader import ArrayView

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
    "TIFFWSIReader",
    "TIFFWSIReaderDelegate",
    "FsspecJsonWSIReader",
    "DICOMWSIReader",
    "NGFFWSIReader",
    "AnnotationStoreReader",
    "TransformedWSIReader",
]
