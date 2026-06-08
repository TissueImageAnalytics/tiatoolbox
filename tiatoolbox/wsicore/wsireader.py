"""This module defines classes which can read image data from WSI formats.

This module provides backward compatibility by importing from the refactored
wsireader submodule. All functionality has been extracted into separate modules
for better maintainability.
"""

from __future__ import annotations

# Import and re-export everything from the wsireader submodule
from .wsireader import (
    AnnotationStoreReader,
    DICOMWSIReader,
    FsspecJsonWSIReader,
    JP2WSIReader,
    NGFFWSIReader,
    OpenSlideWSIReader,
    TIFFWSIReader,
    TIFFWSIReaderDelegate,
    TransformedWSIReader,
    VirtualWSIReader,
    WSIMeta,
    WSIReader,
    _handle_tiff_wsi,
    _handle_virtual_wsi,
    fix_mangled_url_by_pathlib,
    is_dicom,
    is_ngff,
    is_tiled_tiff,
    is_zarr,
)

__all__ = [
    "AnnotationStoreReader",
    "DICOMWSIReader",
    "FsspecJsonWSIReader",
    "JP2WSIReader",
    "NGFFWSIReader",
    "OpenSlideWSIReader",
    "TIFFWSIReader",
    "TIFFWSIReaderDelegate",
    "TransformedWSIReader",
    "VirtualWSIReader",
    "WSIMeta",
    "WSIReader",
    "_handle_tiff_wsi",
    "_handle_virtual_wsi",
    "fix_mangled_url_by_pathlib",
    "is_dicom",
    "is_ngff",
    "is_tiled_tiff",
    "is_zarr",
]
