"""WSI Reader module with support for multiple WSI formats."""

from __future__ import annotations

from tiatoolbox.wsicore.wsimeta import WSIMeta

# Import base classes and utility functions
from .base import (
    AnnotationStoreReader,
    ArrayView,
    DICOMWSIReader,
    FsspecJsonWSIReader,
    JP2WSIReader,
    NGFFWSIReader,
    OpenSlideWSIReader,
    TIFFWSIReader,
    TIFFWSIReaderDelegate,
    TransformedWSIReader,
    VirtualWSIReader,
    WSIReader,
    _handle_tiff_wsi,
    _handle_virtual_wsi,
    fix_mangled_url_by_pathlib,
    is_dicom,
    is_ngff,
    is_tiled_tiff,
    is_url,
    is_zarr,
)

__all__ = [
    "AnnotationStoreReader",
    "ArrayView",
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
    "is_url",
    "is_zarr",
]
