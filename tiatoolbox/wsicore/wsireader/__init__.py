"""WSI Reader module with support for multiple WSI formats."""

from __future__ import annotations

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
    is_url,
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
