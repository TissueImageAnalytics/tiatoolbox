"""Helps with WSIReader selection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Unpack

import numpy as np
import openslide
from upath import UPath

from tiatoolbox import utils
from tiatoolbox.utils.exceptions import FileNotSupportedError

from .detection import is_dicom, is_ngff, is_valid_zarr_fsspec

if TYPE_CHECKING:  # pragma: no cover
    from numbers import Number

    from .base import (
        AnnotationStoreReader,
        DICOMWSIReader,
        FsspecJsonWSIReader,
        NGFFWSIReader,
        OpenSlideWSIReader,
        TIFFWSIReader,
        VirtualWSIReader,
        WSIReader,
    )
    from .types import WSIReaderParams


def open_wsi(
    input_img: str | Path | np.ndarray | WSIReader,
    mpp: tuple[Number, Number] | None = None,
    power: Number | None = None,
    post_proc: str | callable | None = "auto",
    **kwargs: Unpack[WSIReaderParams],
) -> WSIReader:
    """Open a WSIReader with an appropriate object."""
    from .base import VirtualWSIReader, WSIReader  # noqa: PLC0415

    _validate_input(input_img)

    if isinstance(input_img, np.ndarray):
        return VirtualWSIReader(input_img, mpp=mpp, power=power, post_proc=post_proc)

    if isinstance(input_img, WSIReader):
        return input_img

    # Input is a string or Path, normalize to Path
    # UPath preserves s3 paths on Windows
    input_path = UPath(input_img)
    verify_supported_wsi(input_path, **kwargs)

    # Handle special cases first (DICOM, Zarr/NGFF, OME-TIFF)
    special_reader = _handle_special_cases(
        input_path, input_img, mpp, power, post_proc, **kwargs
    )
    if special_reader is not None:
        return special_reader

    return _default_reader(
        input_path=str(input_path),
        mpp=mpp,
        power=power,
        post_proc=post_proc,
    )


def verify_supported_wsi(input_path: Path, **kwargs: Unpack[WSIReaderParams]) -> None:
    """Verify that an input image is supported.

    Args:
        input_path (:class:`Path`):
            Input path to WSI.

    Raises:
        FileNotSupportedError:
            If the input image is not supported.

    """
    if is_ngff(fix_mangled_url_by_pathlib(input_path), **kwargs) or is_dicom(
        input_path
    ):
        return

    _, _, suffixes = utils.misc.split_path_name_ext(input_path)

    if suffixes and suffixes[-1] not in [
        ".svs",
        ".npy",
        ".ndpi",
        ".mrxs",
        ".tif",
        ".tiff",
        ".jp2",
        ".png",
        ".jpg",
        ".jpeg",
        ".zarr",
        ".db",
        ".qptiff",
        ".json",
    ]:
        msg = f"File {input_path} is not a supported file format."
        raise FileNotSupportedError(
            msg,
        )


def _handle_special_cases(
    input_path: Path,
    input_img: str | Path | np.ndarray,
    mpp: tuple[Number, Number] | None = None,
    power: Number | None = None,
    post_proc: str | callable | None = "auto",
    **kwargs: Unpack[WSIReaderParams],
) -> WSIReader | None:
    """Handle special cases for selecting the appropriate WSIReader.

    Args:
        input_path (Path):
            Path to the input image file.
        input_img (str | Path | np.ndarray):
            The input image or path.
        mpp (tuple[Number, Number] | None, optional):
            Microns per pixel resolution.
        power (Number | None, optional):
            Objective power.
        post_proc (str | callable | None, optional):
            Post-processing method or identifier.
        **kwargs (WSIReaderParams):
            Additional keyword arguments for specific reader types.

    Returns:
        WSIReader | None:
            An appropriate WSIReader instance if a match is found, otherwise None.

    Raises:
        FileNotSupportedError:
            If the file format is not supported for NGFF Zarr.

    """
    from .base import _handle_virtual_wsi  # noqa: PLC0415

    _, _, suffixes = utils.misc.split_path_name_ext(input_path)
    last_suffix = suffixes[-1]

    reader = (
        try_dicom(input_path, mpp, power, post_proc)
        or try_fsspec(input_img, mpp, power)
        or try_annotation_store(input_path, last_suffix, post_proc, kwargs)
        or try_ngff(
            fix_mangled_url_by_pathlib(input_path),
            last_suffix,
            mpp,
            power,
            **kwargs,
        )
        or try_ome_tiff(input_path, suffixes, last_suffix, mpp, power, post_proc)
        or try_tiff(input_path, last_suffix, mpp, power, post_proc)
        or try_openslide(input_path, last_suffix, mpp, power)
    )

    if reader is None:
        reader = _handle_virtual_wsi(last_suffix, input_path, mpp, power)

    return reader


def _default_reader(
    input_path: str | Path,
    mpp: tuple[Number, Number] | None = None,
    power: Number | None = None,
    post_proc: str | callable | None = "auto",
) -> WSIReader:
    """Fallback reader."""
    from .base import OpenSlideWSIReader  # noqa: PLC0415

    return OpenSlideWSIReader(
        input_path,
        mpp=mpp,
        power=power,
        post_proc=post_proc,
    )


def try_openslide(
    input_path: Path,
    last_suffix: str,
    mpp: tuple[Number, Number] | None,
    power: Number | None,
) -> OpenSlideWSIReader | None:
    """Try to create an OpenSlideWSIReader if the input is a TIFF file."""
    from .base import OpenSlideWSIReader  # noqa: PLC0415

    if last_suffix in (".tif", ".tiff"):
        try:
            return OpenSlideWSIReader(input_path, mpp=mpp, power=power)
        except (
            openslide.OpenSlideUnsupportedFormatError,
            openslide.OpenSlideError,
        ):
            return None
    return None


def try_dicom(
    input_path: Path,
    mpp: tuple[Number, Number] | None,
    power: Number | None,
    post_proc: str | callable | None,
) -> DICOMWSIReader | None:
    """Try to create a DICOMWSIReader if the input is a DICOM file."""
    from .base import DICOMWSIReader  # noqa: PLC0415

    if is_dicom(input_path):
        return DICOMWSIReader(input_path, mpp=mpp, power=power, post_proc=post_proc)
    return None


def try_fsspec(
    input_img: str | Path | np.ndarray,
    mpp: tuple[Number, Number] | None,
    power: Number | None,
) -> FsspecJsonWSIReader | None:
    """Try to create a FsspecJsonWSIReader if the input is a valid Zarr fsspec."""
    from .base import FsspecJsonWSIReader  # noqa: PLC0415

    if is_valid_zarr_fsspec(input_img):
        return FsspecJsonWSIReader(input_img, mpp=mpp, power=power)
    return None


def try_annotation_store(
    input_path: Path,
    last_suffix: str,
    post_proc: str | callable | None,
    kwargs: Unpack[WSIReaderParams],
) -> AnnotationStoreReader | None:
    """Try to create an AnnotationStoreReader if the file is a .db."""
    from .base import AnnotationStoreReader  # noqa: PLC0415

    if last_suffix == ".db":
        kwargs["post_proc"] = post_proc
        return AnnotationStoreReader(input_path, **kwargs)
    return None


def try_ngff(
    input_path: str | Path,
    last_suffix: str,
    mpp: tuple[Number, Number] | None,
    power: Number | None,
    **kwargs: Unpack[WSIReaderParams],
) -> NGFFWSIReader | None:
    """Try to create an NGFFWSIReader if the file is a valid NGFF Zarr."""
    from .base import NGFFWSIReader  # noqa: PLC0415

    if last_suffix == ".zarr":
        if not is_ngff(input_path, **kwargs):
            msg = f"File {input_path} does not appear to be a v0.4 NGFF zarr."
            raise FileNotSupportedError(msg)
        return NGFFWSIReader(input_path, mpp=mpp, power=power, **kwargs)
    return None


def try_ome_tiff(
    input_path: Path,
    suffixes: list[str],
    last_suffix: str,
    mpp: tuple[Number, Number] | None,
    power: Number | None,
    post_proc: str | callable | None,
) -> TIFFWSIReader | None:
    """Try to create a TIFFWSIReader for OME-TIFF or QPTIFF formats."""
    from .base import TIFFWSIReader  # noqa: PLC0415

    if (
        suffixes[-2:] in ([".ome", ".tiff"], [".ome", ".tif"])
        or last_suffix == ".qptiff"
    ):
        return TIFFWSIReader(input_path, mpp=mpp, power=power, post_proc=post_proc)
    return None


def try_tiff(
    input_path: Path,
    last_suffix: str,
    mpp: tuple[Number, Number] | None,
    power: Number | None,
    post_proc: str | callable | None,
) -> TIFFWSIReader | OpenSlideWSIReader | VirtualWSIReader | None:
    """Try to create a TIFFWSIReader.

    Try to create a TIFFWSIReader for standard TIFF formats,
    or fallback to virtual WSI.

    """
    from .base import _handle_tiff_wsi, _handle_virtual_wsi  # noqa: PLC0415

    if last_suffix not in {".tif", ".tiff"}:
        return None

    try:
        return _handle_tiff_wsi(
            input_path,
            mpp,
            power,
            post_proc,
        )

    except ValueError as exc:
        if "Unsupported TIFF WSI format" in str(exc):
            return _handle_virtual_wsi(
                last_suffix,
                input_path,
                mpp,
                power,
            )
        raise


def _validate_input(input_img: str | Path | np.ndarray | WSIReader) -> None:
    """Validate the input image type.

    Args:
        input_img (str | Path | np.ndarray): The input image, which
        must be a path, string, numpy array, or WSIReader.

    Raises:
        TypeError: If the input is not one of the accepted types.

    """
    from .base import WSIReader  # noqa: PLC0415

    if not isinstance(input_img, (WSIReader, np.ndarray, str, Path)):
        msg = "Invalid input: Must be a WSIReader, numpy array, string or Path"
        raise TypeError(msg)


def fix_mangled_url_by_pathlib(input_path: str | Path) -> str:
    """Fix URl mangled by Path."""
    # Fix Mangled URL
    return re.sub(r"^(s3|http|https|ftp|file):/(?!/)", r"\1://", str(input_path))
