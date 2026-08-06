"""WSI Reader module with support for multiple WSI formats."""

from __future__ import annotations

from tiatoolbox.wsicore.wsimeta import WSIMeta

# Import base classes and utility functions
from .base import (
    AnnotationStoreReader,
    ArrayView,
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
)
from .detection import is_dicom, is_ngff, is_tiled_tiff, is_url, is_zarr

# Import reader classes
from .dicom import DICOMWSIReader

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
    "is_dicom",
    "is_ngff",
    "is_tiled_tiff",
    "is_url",
    "is_zarr",
]
