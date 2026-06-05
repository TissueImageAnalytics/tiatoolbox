"""WSI Reader module with support for multiple WSI formats."""

from __future__ import annotations

# Import base classes and utility functions
from .base import (
    WSIReader,
    is_dicom,
    is_ngff,
    is_tiled_tiff,
    is_zarr,
    fix_mangled_url_by_pathlib,
    _handle_virtual_wsi,
    _handle_tiff_wsi,
)

# Import reader classes
from .openslide import OpenSlideWSIReader
from .jp2 import JP2WSIReader
from .virtual import VirtualWSIReader, ArrayView
from .tiff import TIFFWSIReader, TIFFWSIReaderDelegate
from .fsspec_json import FsspecJsonWSIReader
from .dicom import DICOMWSIReader
from .ngff import NGFFWSIReader
from .annotation_store import AnnotationStoreReader
from .transformed import TransformedWSIReader

__all__ = [
    # Base class and utilities
    "WSIReader",
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
