"""This module defines classes which can read image data from WSI formats."""

from __future__ import annotations

import logging
import math
import re
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Unpack
from urllib.parse import urlparse

import numpy as np
import openslide
import pandas as pd
import tifffile
import zarr
from packaging.version import Version
from upath import UPath

from tiatoolbox import logger, utils
from tiatoolbox.annotation import AnnotationStore
from tiatoolbox.utils import postproc_defs
from tiatoolbox.utils.env_detection import pixman_warning
from tiatoolbox.utils.exceptions import FileNotSupportedError
from tiatoolbox.wsicore.wsimeta import WSIMeta

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    from tiatoolbox.type_hints import (
        Bounds,
        IntBounds,
        IntPair,
        NumPair,
        Resolution,
        Units,
    )
    from tiatoolbox.wsicore import WSIReaderParams
    from tiatoolbox.wsicore.metadata.ngff import Multiscales

    from . import (
        AnnotationStoreReader,
        DICOMWSIReader,
        FsspecJsonWSIReader,
        NGFFWSIReader,
        OpenSlideWSIReader,
        TIFFWSIReader,
    )

pixman_warning()

MIN_NGFF_VERSION = Version("0.4")
MAX_NGFF_VERSION = Version("0.4")


def is_dicom(path: Path) -> bool:
    """Check if the input is a DICOM file.

    Args:
        path (Path): Path to the file to check.

    Returns:
        bool: True if the file is a DICOM file.

    """
    path = Path(path)
    is_dcm = path.suffix.lower() == ".dcm"
    is_dcm_dir = path.is_dir() and any(
        p.suffix.lower() == ".dcm" for p in path.iterdir()
    )
    return is_dcm or is_dcm_dir


def is_tiled_tiff(path: Path) -> bool:
    """Check if the input is a tiled TIFF file.

    Args:
        path (Path):
            Path to the file to check.

    Returns:
        bool:
            True if the file is a tiled TIFF file.

    """
    path = Path(path)
    try:
        tif = tifffile.TiffFile(path)
    except tifffile.TiffFileError:
        return False
    return tif.pages[0].is_tiled


def is_zarr(path: Path, **kwargs: Unpack[WSIReaderParams]) -> bool:
    """Check if the input is a Zarr file.

    Args:
        path (Path):
            Path to the file to check.

    Returns:
        bool:
            True if the file is a Zarr file.

    """
    try:
        _ = zarr.open(path, **kwargs, mode="r")
    except Exception:  # skipcq: PYL-W0703  # noqa: BLE001
        return False

    return True


def is_ngff(  # skipcq: PY-R1000  # noqa: PLR0911
    path: str | Path,
    min_version: Version = MIN_NGFF_VERSION,
    max_version: Version = MAX_NGFF_VERSION,
    **kwargs: Unpack[WSIReaderParams],
) -> bool:
    """Check if the input is an NGFF file.

    This should return True for a zarr groups stored in a directory, zip
    file, or SQLite database.

    Args:
        path (Path):
            Path to the file to check.
        min_version (Tuple[int, ...]):
            Minimum version of the NGFF file to be considered valid.
        max_version (Tuple[int, ...]):
            Maximum version of the NGFF file to be considered valid.

    Returns:
        bool:
            True if the file is an NGFF file.

    """
    zarr_kwargs = {k: v for k, v in kwargs.items() if k in ["storage_options"]}
    try:
        zarr_group = zarr.open(path, mode="r", **zarr_kwargs)
    except Exception:  # skipcq: PYL-W0703  # noqa: BLE001
        return False
    if not isinstance(zarr_group, zarr.Group):
        return False
    group_attrs = zarr_group.attrs.asdict()
    try:
        multiscales: Multiscales = group_attrs.get("multiscales", [None])
        omero = group_attrs.get("omero")
        if not all(
            [
                isinstance(multiscales, list),
                isinstance(omero, dict),
                all(isinstance(m, dict) for m in multiscales),
            ],
        ):
            logger.warning(
                "The NGFF file is not valid. "
                "The multiscales and omero attributes "
                "must be present and of the correct type.",
            )
            return False
    except KeyError:
        return False
    multiscales_versions = {
        Version(scale["version"]) for scale in multiscales if "version" in scale
    }
    omero_version: str | None = omero.get("version")
    if omero_version:
        omero_version: Version = Version(omero_version)
        if omero_version < min_version:
            logger.warning(
                "The minimum supported version of the NGFF file is %s. "
                "But the versions of the multiscales in the file are %s.",
                min_version,
                multiscales_versions,
            )
            return False
        if omero_version > max_version:
            logger.warning(
                "The maximum supported version of the NGFF file is %s. "
                "But the versions of the multiscales in the file are %s.",
                max_version,
                multiscales_versions,
            )
            return True

    if len(multiscales_versions) > 1:
        logger.warning(
            "Found multiple versions for NGFF multiscales: %s",
            multiscales_versions,
        )

    if any(version < min_version for version in multiscales_versions):
        logger.warning(
            "The minimum supported version of the NGFF file is %s. "
            "But the versions of the multiscales in the file are %s.",
            min_version,
            multiscales_versions,
        )
        return False

    if any(version > max_version for version in multiscales_versions):
        logger.warning(
            "The maximum supported version of the NGFF file is %s. "
            "But the versions of the multiscales in the file are %s.",
            max_version,
            multiscales_versions,
        )
        return True

    return True


def is_url(path_or_url: str | Path) -> bool:
    """Returns True if input is a URL else False."""
    parsed = urlparse(str(path_or_url))
    return parsed.scheme in {"s3", "http", "https", "ftp", "file"}


def _handle_virtual_wsi(
    last_suffix: str,
    input_path: Path,
    mpp: tuple[Number, Number] | None,
    power: Number | None,
) -> VirtualWSIReader | None:
    """Handle virtual WSI cases.

    Args:
        last_suffix (str):
            Suffix of the file to read.
        input_path (Path):
             Input path to virtual WSI.
        mpp (:obj:`tuple` or :obj:`list` or :obj:`None`, optional):
            The MPP of the WSI. If not provided, the MPP is approximated
            from the objective power.
        power (:obj:`float` or :obj:`None`, optional):
            The objective power of the WSI. If not provided, the power
            is approximated from the MPP.

    Returns:
        VirtualWSIReader | None:
            :class:`VirtualWSIReader` if input_path is valid path to virtual WSI
            otherwise None.

    """
    from .jp2 import JP2WSIReader  # noqa: PLC0415

    # Handle homogeneous cases (based on final suffix)
    def np_virtual_wsi(
        input_path: np.ndarray,
        *args: Number | tuple | str | WSIMeta | None,
        **kwargs: dict,
    ) -> VirtualWSIReader:
        """Create a virtual WSI from a numpy array."""
        return VirtualWSIReader(input_path, *args, **kwargs)

    suffix_to_reader = {
        ".npy": np_virtual_wsi,
        ".jp2": JP2WSIReader,
        ".jpeg": VirtualWSIReader,
        ".jpg": VirtualWSIReader,
        ".png": VirtualWSIReader,
        ".tif": VirtualWSIReader,
        ".tiff": VirtualWSIReader,
    }

    if last_suffix in suffix_to_reader:
        return suffix_to_reader[last_suffix](input_path, mpp=mpp, power=power)

    return None


def _handle_tiff_wsi(
    input_path: Path,
    mpp: tuple[Number, Number] | None,
    power: Number | None,
    post_proc: str | callable | None,
) -> TIFFWSIReader | OpenSlideWSIReader | None:
    """Handle TIFF WSI cases.

    Args:
        input_path (Path):
             Input path to virtual WSI.
        mpp (:obj:`tuple` or :obj:`list` or :obj:`None`, optional):
            The MPP of the WSI. If not provided, the MPP is approximated
            from the objective power.
        power (:obj:`float` or :obj:`None`, optional):
            The objective power of the WSI. If not provided, the power
            is approximated from the MPP.
        post_proc (str | callable | None):
            Post-processing function to apply to the image.

    Returns:
        OpenSlideWSIReader | TIFFWSIReader | None:
            :class:`OpenSlideWSIReader` or :class:`TIFFWSIReader` if input_path is
            valid path to tiff WSI otherwise None.

    """
    from .openslide import OpenSlideWSIReader  # noqa: PLC0415
    from .tiff import TIFFWSIReader  # noqa: PLC0415

    if openslide.OpenSlide.detect_format(input_path) is not None:
        try:
            return OpenSlideWSIReader(
                input_path, mpp=mpp, power=power, post_proc=post_proc
            )
        except openslide.OpenSlideError:
            pass
    if is_tiled_tiff(input_path):
        return TIFFWSIReader(input_path, mpp=mpp, power=power, post_proc=post_proc)

    return None


def fix_mangled_url_by_pathlib(input_path: str | Path) -> str:
    """Fix URl mangled by Path."""
    # Fix Mangled URL
    return re.sub(r"^(s3|http|https|ftp|file):/(?!/)", r"\1://", str(input_path))


class WSIReader:
    """Base whole slide image (WSI) reader class.

    This class defines functions for reading pixel data and metadata
    from whole slide image (WSI) files.

    Attributes:
        input_path (Path):
            Input path to WSI file.

    Args:
        input_img (str, :obj:`Path`, :obj:`ndarray` or :obj:`.WSIReader`):
            Input path to WSI.
        mpp (:obj:`tuple` or :obj:`list` or :obj:`None`, optional):
            The MPP of the WSI. If not provided, the MPP is approximated
            from the objective power.
        power (:obj:`float` or :obj:`None`, optional):
            The objective power of the WSI. If not provided, the power
            is approximated from the MPP.
        post_proc (str | callable | None):
            Post-processing function to apply to the image. If None,
            no post-processing is applied. If 'auto', the post-processing
            function is automatically selected based on the reader type.

    """

    @staticmethod
    def open(
        input_img: str | Path | np.ndarray | WSIReader,
        mpp: tuple[Number, Number] | None = None,
        power: Number | None = None,
        post_proc: str | callable | None = "auto",
        **kwargs: Unpack[WSIReaderParams],
    ) -> WSIReader:
        """Return an appropriate :class:`.WSIReader` object.

        Args:
            input_img (str, Path, :obj:`numpy.ndarray` or :obj:`.WSIReader`):
                Input to create a WSI object from. Supported types of
                input are: `str` and :obj:`Path` which point to the
                location on the disk where image is stored,
                :class:`numpy.ndarray` in which the input image in the
                form of numpy array (HxWxC) is stored, or :obj:`.WSIReader`
                which is an already created tiatoolbox WSI handler. In
                the latter case, the function directly passes the
                input_imge to the output.
            mpp (tuple):
                (x, y) tuple of the MPP in the units of the input image.
            power (float):
                Objective power of the input image.
            post_proc (str | callable | None):
                Post-processing function to apply to the image. If None,
                no post-processing is applied. If 'auto', the post-processing
                function is automatically selected based on the reader type.
            kwargs (dict):
                Key-word arguments.

        Returns:
            WSIReader:
                An object with base :class:`.WSIReader` as base class.

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> wsi = WSIReader.open(input_img="./sample.svs")

        When working with multi-channel images such as immunofluorescence,
        the default behaviour when post_proc is set to "auto" is to convert
        the output to RGB when reading from the slide. If you need the raw
        channel outputs, set post_proc to None:

            >>> wsi = WSIReader.open(input_img="./sample.ome.tiff", post_proc="auto")
            >>> region = wsi.read_rect((0, 0), (100, 100))
            >>> print(region.shape)
            (100, 100, 3)  # RGB output

            >>> wsi = WSIReader.open(input_img="./sample.ome.tiff", post_proc=None)
            >>> region = wsi.read_rect((0, 0), (100, 100))
            >>> print(region.shape)
            (100, 100, 5)  # raw channel outputs

        """
        WSIReader._validate_input(input_img)

        if isinstance(input_img, np.ndarray):
            return VirtualWSIReader(
                input_img, mpp=mpp, power=power, post_proc=post_proc
            )

        if isinstance(input_img, WSIReader):
            return input_img

        # Input is a string or Path, normalise to Path
        input_path = UPath(input_img)
        WSIReader.verify_supported_wsi(input_path, **kwargs)

        # Handle special cases first (DICOM, Zarr/NGFF, OME-TIFF)
        special_reader = WSIReader._handle_special_cases(
            input_path, input_img, mpp, power, post_proc, **kwargs
        )
        if special_reader is not None:
            return special_reader

        # Try openslide last
        from .openslide import OpenSlideWSIReader  # noqa: PLC0415

        return OpenSlideWSIReader(input_path, mpp=mpp, power=power, post_proc=post_proc)

    @staticmethod
    def _validate_input(input_img: str | Path | np.ndarray) -> None:
        """Validate the input image type.

        Args:
            input_img (str | Path | np.ndarray): The input image, which
            must be a path, string, numpy array, or WSIReader.

        Raises:
            TypeError: If the input is not one of the accepted types.

        """
        if not isinstance(input_img, (WSIReader, np.ndarray, str, Path)):
            msg = "Invalid input: Must be a WSIReader, numpy array, string or Path"
            raise TypeError(msg)

    @staticmethod
    def verify_supported_wsi(
        input_path: Path, **kwargs: Unpack[WSIReaderParams]
    ) -> None:
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

    @staticmethod
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
            input_path (Path): Path to the input image file.
            input_img (str | Path | np.ndarray): The input image or path.
            mpp (tuple[Number, Number] | None, optional): Microns per pixel resolution.
            power (Number | None, optional): Objective power.
            post_proc (str | callable | None, optional): Post-processing method
            or identifier.
            **kwargs (dict): Additional keyword arguments for specific reader types.

        Returns:
            WSIReader | None: An appropriate WSIReader instance if a match is found,
            otherwise None.

        Raises:
            FileNotSupportedError: If the file format is not supported for NGFF Zarr.

        """
        _, _, suffixes = utils.misc.split_path_name_ext(input_path)
        last_suffix = suffixes[-1]

        reader = (
            WSIReader.try_dicom(input_path, mpp, power, post_proc)
            or WSIReader.try_fsspec(input_img, mpp, power)
            or WSIReader.try_annotation_store(
                input_path, last_suffix, post_proc, kwargs
            )
            or WSIReader.try_ngff(
                fix_mangled_url_by_pathlib(input_path),
                last_suffix,
                mpp,
                power,
                **kwargs,
            )
            or WSIReader.try_ome_tiff(
                input_path, suffixes, last_suffix, mpp, power, post_proc
            )
            or WSIReader.try_tiff(input_path, last_suffix, mpp, power, post_proc)
            or WSIReader.try_openslide(input_path, last_suffix, mpp, power)
        )

        if reader is None:
            reader = _handle_virtual_wsi(last_suffix, input_path, mpp, power)

        return reader

    @staticmethod
    def try_openslide(
        input_path: Path,
        last_suffix: str,
        mpp: tuple[Number, Number] | None,
        power: Number | None,
    ) -> OpenSlideWSIReader | None:
        """Try to create an OpenSlideWSIReader if the input is a TIFF file."""
        if last_suffix in (".tif", ".tiff"):
            try:
                from .openslide import OpenSlideWSIReader  # noqa: PLC0415

                return OpenSlideWSIReader(input_path, mpp=mpp, power=power)
            except (
                openslide.OpenSlideUnsupportedFormatError,
                openslide.OpenSlideError,
            ):
                return None
        return None

    @staticmethod
    def try_dicom(
        input_path: Path,
        mpp: tuple[Number, Number] | None,
        power: Number | None,
        post_proc: str | callable | None,
    ) -> DICOMWSIReader | None:
        """Try to create a DICOMWSIReader if the input is a DICOM file."""
        if is_dicom(input_path):
            from .dicom import DICOMWSIReader  # noqa: PLC0415

            return DICOMWSIReader(input_path, mpp=mpp, power=power, post_proc=post_proc)
        return None

    @staticmethod
    def try_fsspec(
        input_img: str | Path | np.ndarray,
        mpp: tuple[Number, Number] | None,
        power: Number | None,
    ) -> FsspecJsonWSIReader | None:
        """Try to create a FsspecJsonWSIReader if the input is a valid Zarr fsspec."""
        from .fsspec_json import FsspecJsonWSIReader  # noqa: PLC0415

        if FsspecJsonWSIReader.is_valid_zarr_fsspec(input_img):
            return FsspecJsonWSIReader(input_img, mpp=mpp, power=power)
        return None

    @staticmethod
    def try_annotation_store(
        input_path: Path,
        last_suffix: str,
        post_proc: str | callable | None,
        kwargs: dict,
    ) -> AnnotationStoreReader | None:
        """Try to create an AnnotationStoreReader if the file is a .db."""
        if last_suffix == ".db":
            from .annotation_store import AnnotationStoreReader  # noqa: PLC0415

            kwargs["post_proc"] = post_proc
            return AnnotationStoreReader(input_path, **kwargs)
        return None

    @staticmethod
    def try_ngff(
        input_path: str | Path,
        last_suffix: str,
        mpp: tuple[Number, Number] | None,
        power: Number | None,
        **kwargs: Unpack[WSIReaderParams],
    ) -> NGFFWSIReader | None:
        """Try to create an NGFFWSIReader if the file is a valid NGFF Zarr."""
        if last_suffix == ".zarr":
            if not is_ngff(input_path, **kwargs):
                msg = f"File {input_path} does not appear to be a v0.4 NGFF zarr."
                raise FileNotSupportedError(msg)
            from .ngff import NGFFWSIReader  # noqa: PLC0415

            return NGFFWSIReader(input_path, mpp=mpp, power=power, **kwargs)
        return None

    @staticmethod
    def try_ome_tiff(
        input_path: Path,
        suffixes: list[str],
        last_suffix: str,
        mpp: tuple[Number, Number] | None,
        power: Number | None,
        post_proc: str | callable | None,
    ) -> TIFFWSIReader | None:
        """Try to create a TIFFWSIReader for OME-TIFF or QPTIFF formats."""
        if (
            suffixes[-2:] in ([".ome", ".tiff"], [".ome", ".tif"])
            or last_suffix == ".qptiff"
        ):
            from .tiff import TIFFWSIReader  # noqa: PLC0415

            return TIFFWSIReader(input_path, mpp=mpp, power=power, post_proc=post_proc)
        return None

    @staticmethod
    def try_tiff(
        input_path: Path,
        last_suffix: str,
        mpp: tuple[Number, Number] | None,
        power: Number | None,
        post_proc: str | callable | None,
    ) -> TIFFWSIReader | None:
        """Try to create a TIFFWSIReader.

        Try to create a TIFFWSIReader for standard TIFF formats,
        or fallback to virtual WSI.

        """
        if last_suffix in (".tif", ".tiff"):
            try:
                from .tiff import TIFFWSIReader  # noqa: PLC0415

                return TIFFWSIReader(
                    input_path, mpp=mpp, power=power, post_proc=post_proc
                )
            except ValueError as e:
                if "Unsupported TIFF WSI format" in str(e):
                    return _handle_virtual_wsi(last_suffix, input_path, mpp, power)
                raise
        return None

    def __init__(
        self: WSIReader,
        input_img: str | Path | np.ndarray | AnnotationStore,
        post_proc: callable | None = None,
        **kwargs: Unpack[WSIReaderParams],
    ) -> None:
        """Initialize :class:`WSIReader`."""
        if isinstance(input_img, (np.ndarray, AnnotationStore)):
            self.input_path = None
        elif is_url(path_or_url=input_img):
            self.input_path = str(input_img)
        else:
            self.input_path = Path(input_img)
            if not self.input_path.exists():
                msg = f"Input path does not exist: {self.input_path}"
                raise FileNotFoundError(msg)
        self._m_info = None

        mpp = kwargs.get("mpp")
        power = kwargs.get("power")

        # Set a manual mpp value
        if mpp is not None and isinstance(mpp, Number):
            mpp = (mpp, mpp)
        if mpp is not None and (not hasattr(mpp, "__len__") or len(mpp) != 2):  # noqa: PLR2004
            msg = "`mpp` must be a number or iterable of length 2."
            raise TypeError(msg)
        self._manual_mpp = tuple(mpp) if mpp is not None else None

        # Set a manual power value
        if power and not isinstance(power, Number):
            msg = "`power` must be a number."
            raise TypeError(msg)
        self._manual_power = power
        self.post_proc = self.get_post_proc(post_proc)

    @property
    def info(self: WSIReader) -> WSIMeta:
        """WSI metadata property.

        This property is cached and only generated on the first call.

        Returns:
            WSIMeta:
                An object containing normalized slide metadata.

        """
        if self._m_info is not None:
            return self._m_info
        self._m_info = self._info()
        if self._manual_mpp:
            self._m_info.mpp = np.array(self._manual_mpp)
        if self._manual_power:
            self._m_info.objective_power = self._manual_power
        return self._m_info

    @info.setter
    def info(self: WSIReader, meta: WSIMeta) -> None:
        """WSI metadata setter.

        Args:
            meta (WSIMeta): Metadata object.

        """
        self._m_info = meta

    def get_post_proc(self: WSIReader, post_proc: str | callable | None) -> callable:
        """Get the post-processing function.

        Args:
            post_proc (str | callable | None):
                Post-processing function to apply to the image. If auto,
                will use no post_proc unless reader is TIFF or Virtual Reader,
                in which case it will use MultichannelToRGB.

        Returns:
            callable:
                Post-processing function.

        """
        if callable(post_proc):
            return post_proc
        if post_proc is None:
            return None
        if post_proc == "auto":
            # if its TIFFWSIReader or VirtualWSIReader, return fn to
            # allow multichannel, else return None
            from .tiff import TIFFWSIReader  # noqa: PLC0415

            if isinstance(self, (TIFFWSIReader, VirtualWSIReader)):
                return postproc_defs.MultichannelToRGB()
            return None
        if isinstance(post_proc, str) and hasattr(postproc_defs, post_proc):
            return getattr(postproc_defs, post_proc)()
        msg = f"Invalid post-processing function: {post_proc}"
        raise ValueError(msg)

    def _info(self: WSIReader) -> WSIMeta:
        """WSI metadata internal getter used to update info property.

        Missing values for MPP and objective power are approximated and
        a warning raised. Objective power is calculated as the mean of
        the :func:utils.transforms.mpp2common_objective_power in x and
        y. MPP (x and y) is approximated using objective power via
        :func:utils.transforms.objective_power2mpp.

        Returns:
            WSIMeta:
                An object containing normalized slide metadata.

        """
        raise NotImplementedError

    def _find_optimal_level_and_downsample(
        self: WSIReader,
        resolution: Resolution,
        units: Units,
        precision: int = 3,
    ) -> tuple[int, np.ndarray]:
        """Find the optimal level to read at for a desired resolution and units.

        The optimal level is the most downscaled level of the image
        pyramid (or multi-resolution layer) which is larger than the
        desired target resolution. The returned scale is the downsample
        factor required, post read, to achieve the desired resolution.

        Args:
            resolution (Resolution):
                Resolution to find optimal read parameters for
            units (Units):
                Units of the scale.
            precision (int or optional):
                Decimal places to use when finding optimal scale. This
                can be adjusted to avoid errors when an unnecessary
                precision is used. E.g. 1.1e-10 > 1 is insignificant in
                most cases. Defaults to 3.

        Returns:
            tuple:
                Optimal read level and scale factor between the optimal
                level and the target scale (usually <= 1):
                - :py:obj:`int` - Optimal read level.
                - :class:`numpy.ndarray` - Scale factor in X and Y.

        """
        level_scales = self.info.relative_level_scales(resolution, units)
        level_resolution_sufficient = [
            all(np.round(x, decimals=precision) <= 1) for x in level_scales
        ]
        # Check if level 0 is lower resolution than required (scale > 1)
        if not any(level_resolution_sufficient):
            level = 0
        else:
            # Find the first level with relative scale >= 1.
            # Note: np.argmax finds the index of the first True element.
            # Here it is used on a reversed list to find the first
            # element <=1, which is the same element as the last <=1
            # element when counting forward in the regular list.
            reverse_index = np.argmax(level_resolution_sufficient[::-1])
            # Convert the index from the reversed list to the regular index (level)
            level = (len(level_scales) - 1) - reverse_index
        scale = level_scales[level]

        # Check for requested resolution > than baseline resolution
        if any(np.array(scale) > 1):
            logger.warning(
                "Read: Scale > 1."
                "This means that the desired resolution is higher"
                " than the WSI baseline (maximum encoded resolution)."
                " Interpolation of read regions may occur.",
            )
        return level, scale

    def find_read_rect_params(
        self: WSIReader,
        location: IntPair,
        size: IntPair,
        resolution: Resolution,
        units: Units,
        precision: int = 3,
    ) -> tuple[int, IntPair, IntPair, NumPair, IntPair]:
        """Find optimal parameters for reading a rect at a given resolution.

        Reading the image at full baseline resolution and re-sampling to
        the desired resolution would require a large amount of memory
        and be very slow. This function checks the other resolutions
        stored in the WSI's pyramid of resolutions to find the lowest
        resolution (the smallest level) which is higher resolution (a larger
        level) than the requested output resolution.

        In addition to finding this 'optimal level', the scale factor to
        apply after reading in order to obtain the desired resolution is
        found along with conversions of the location and size into level
        and baseline coordinates.

        Args:
            location (IntPair):
                Location in terms of the baseline image (level 0)
                resolution.
            size (IntPair):
                Desired output size in pixels (width, height) tuple.
            resolution (Resolution):
                Desired output resolution.
            units (Units):
                Units of scale, default = "level". Supported units are:
                - microns per pixel ('mpp')
                - objective power ('power')
                - pyramid / resolution level ('level')
                - pixels per baseline pixel ("baseline")
            precision (int, optional):
                Decimal places to use when finding optimal scale. See
                :func:`find_optimal_level_and_downsample` for more.

        Returns:
            tuple:
                Parameters for reading the requested region.

                - :py:obj:`int` - Optimal read level.

                - :py:obj:`tuple` - Read location in level coordinates.
                    - :py:obj:`int` - X location.
                    - :py:obj:`int` - Y location.

                - :py:obj:`tuple` - Region size in level coordinates.
                    - :py:obj:`int` - Width.
                    - :py:obj:`int` - Height.

                - :py:obj:`tuple` - Scaling to apply after level read.
                    - :py:obj:`float` - X scale factor.
                    - :py:obj:`float` - Y scale factor.

                - :py:obj:`tuple` - Region size in baseline coordinates.
                    - :py:obj:`int` - Width.
                    - :py:obj:`int` - Height.

        """
        read_level, post_read_scale_factor = self._find_optimal_level_and_downsample(
            resolution,
            units,
            precision,
        )
        info = self.info
        level_downsample = info.level_downsamples[read_level]
        baseline_read_size = np.round(
            np.array(size) * level_downsample / post_read_scale_factor,
        ).astype(int)
        level_read_size = np.round(np.array(size) / post_read_scale_factor).astype(int)
        level_location = np.round(np.array(location) / level_downsample).astype(int)
        return (
            read_level,
            level_location,
            level_read_size,
            post_read_scale_factor,
            baseline_read_size,
        )

    def _find_read_params_at_resolution(
        self: WSIReader,
        location: IntPair,
        size: IntPair,
        resolution: Resolution,
        units: Units,
    ) -> tuple[int, NumPair, IntPair, IntPair, IntPair, IntPair]:
        """Works similarly to `_find_read_rect_params`.

        Return the information necessary for scaling. While
        `_find_read_rect_params` assumes location to be at baseline.
        This function assumes location to be at requested resolution.

        Args:
            location (IntPair):
                Location in the requested resolution system.
            size (IntPair):
                Desired output size in pixels (width, height) tuple and
                in the requested resolution system.
            resolution (Resolution):
                Desired output resolution.
            units (Units):
                Units of scale, default = "level". Supported units are:
                - microns per pixel ('mpp') - objective power ('power')
                - pyramid / resolution level ('level') - pixels per
                baseline pixel ("baseline")

        Returns:
            tuple:
                Parameters for reading the requested region:
                - :py:obj:`int` - Optimal read level.
                - :py:obj:`tuple` - Scaling to apply after level read to
                  achieve desired output resolution.
                    - :py:obj:`float` - X scale factor.
                    - :py:obj:`float` - Y scale factor.
                - :py:obj:`tuple` - Region size in read level
                  coordinates.
                    - :py:obj:`int` - Width.
                    - :py:obj:`int` - Height.
                - :py:obj:`tuple` - Region location in read level
                  coordinates.
                    - :py:obj:`int` - X location.
                    - :py:obj:`int` - Y location.
                - :py:obj:`tuple` - Region size in level 0 coordinates.
                    - :py:obj:`int` - Width.
                    - :py:obj:`int` - Height.
                - :py:obj:`tuple` - Region location level 0 coordinates.
                    - :py:obj:`int` - X location.
                    - :py:obj:`int` - Y location.

        """
        (
            read_level,
            # read_level to requested resolution (full)
            read_level_to_resolution_scale_factor,
        ) = self._find_optimal_level_and_downsample(
            resolution,
            units,
        )
        info = self.info

        # Do we need sanity check for input form ?
        requested_location = np.array(location)
        requested_size = np.array(size)
        baseline_to_read_level_scale_factor = 1 / info.level_downsamples[read_level]

        baseline_to_resolution_scale_factor = (
            baseline_to_read_level_scale_factor * read_level_to_resolution_scale_factor
        )

        size_at_baseline = requested_size / baseline_to_resolution_scale_factor
        location_at_baseline = (
            requested_location.astype(np.float32) / baseline_to_resolution_scale_factor
        )
        size_at_read_level = requested_size / read_level_to_resolution_scale_factor
        location_at_read_level = (
            requested_location.astype(np.float32)
            / read_level_to_resolution_scale_factor
        )
        output = (
            size_at_read_level,
            location_at_read_level,
            size_at_baseline,
            location_at_baseline,
        )
        output = tuple(np.ceil(v).astype(np.int64) for v in output)
        return (read_level, read_level_to_resolution_scale_factor, *output)

    def bounds_at_resolution_to_baseline(
        self: WSIReader,
        bounds: Bounds,
        resolution: Resolution,
        units: Units,
    ) -> Bounds:
        """Find corresponding bounds in baseline.

        Find corresponding bounds in baseline given the input is at
        requested resolution.

        """
        bounds_at_resolution = np.array(bounds)
        tl_at_resolution = bounds_at_resolution[:2]  # is in XY
        br_at_resolution = bounds_at_resolution[2:]
        size_at_resolution = br_at_resolution - tl_at_resolution
        # Find parameters for optimal read
        (
            _,  # read_level,
            _,  # read_level_to_requested_scale_factor,
            _,  # size_at_read_level,
            _,  # location_at_read_level,
            size_at_baseline,
            location_at_baseline,
        ) = self._find_read_params_at_resolution(
            tl_at_resolution,
            size_at_resolution,
            resolution,
            units,
        )
        tl_at_baseline = location_at_baseline
        br_at_baseline = tl_at_baseline + size_at_baseline
        return np.concatenate([tl_at_baseline, br_at_baseline])  # bounds at baseline

    def slide_dimensions(
        self: WSIReader,
        resolution: Resolution,
        units: Units,
        precision: int = 3,
    ) -> IntPair:
        """Return the size of WSI at requested resolution.

        Args:
            resolution (Resolution):
                Resolution to read thumbnail at, default = 1.25
                (objective power).
            units (Units):
                resolution units, default="power".
            precision (int, optional):
                Decimal places to use when finding optimal scale. See
                :func:`find_optimal_level_and_downsample` for more.

        Returns:
            :py:obj:`tuple`:
                Size of the WSI in (width, height).

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
            >>> slide_shape = wsi.slide_dimensions(0.55, 'mpp')

        """
        wsi_shape_at_baseline = self.info.slide_dimensions
        # Find parameters for optimal read
        (
            _,
            _,
            wsi_shape_at_resolution,
            _,
        ) = self.find_read_bounds_params(
            [0, 0, *list(wsi_shape_at_baseline)],
            resolution,
            units,
            precision,
        )
        return wsi_shape_at_resolution

    def find_read_bounds_params(
        self: WSIReader,
        bounds: Bounds,
        resolution: Resolution,
        units: Units,
        precision: int = 3,
    ) -> tuple[int, IntBounds, IntPair, IntPair, np.ndarray]:
        """Find optimal parameters for reading bounds at a given resolution.

        Args:
            bounds (IntBounds):
                Tuple of (start_x, start_y, end_x, end_y) i.e. (left,
                top, right, bottom) of the region in baseline reference
                frame.
            resolution (Resolution):
                desired output resolution
            units (Units):
                units of scale, default = "level". Supported units are:
                microns per pixel (mpp), objective power (power),
                pyramid / resolution level (level), pixels per baseline
                pixel (baseline).
            precision (int, optional):
                Decimal places to use when finding optimal scale. See
                :func:`find_optimal_level_and_downsample` for more.

        Returns:
            tuple:
                Parameters for reading the requested bounds area:
                - :py:obj:`int` - Optimal read level
                - :py:obj:`tuple` - Bounds of the region in level coordinates
                    - :py:obj:`int` - Left (start x value)
                    - :py:obj:`int` - Top (start y value)
                    - :py:obj:`int` - Right (end x value)
                    - :py:obj:`int` - Bottom (end y value)
                - :py:obj:`tuple` - Expected size of the output image
                    - :py:obj:`int` - Width
                    - :py:obj:`int` - Height
                - np.ndarray - Scale factor of re-sampling to apply after reading.

        """
        start_x, start_y, end_x, end_y = bounds
        read_level, post_read_scale_factor = self._find_optimal_level_and_downsample(
            resolution,
            units,
            precision,
        )
        info = self.info
        level_downsample = info.level_downsamples[read_level]
        location = np.array([start_x, start_y])
        size = np.array([end_x - start_x, end_y - start_y])
        level_size = np.round(np.array(size) / level_downsample).astype(int)
        level_location = np.round(location / level_downsample).astype(int)
        level_bounds = (*level_location, *(level_location + level_size))
        output_size = np.round(level_size * post_read_scale_factor).astype(int)
        return (read_level, level_bounds, output_size, post_read_scale_factor)

    @staticmethod
    def _check_unit_conversion_integrity(
        input_unit: str,
        output_unit: str,
        baseline_mpp: Resolution,
        baseline_power: Resolution,
    ) -> None:
        """Checks integrity of units before unit conversion.

        Args:
            input_unit (str):
                input units
            output_unit (str):
                output units
            baseline_mpp:
                baseline microns per pixel (mpp)
            baseline_power:
                baseline magnification level.

        Raises:
            ValueError:
                If the checks on unit conversion fails.

        """
        if input_unit not in {"mpp", "power", "level", "baseline"}:
            msg = (
                "Invalid input_unit: argument accepts only one of the "
                "following options: `'mpp'`, `'power'`, `'level'`, `'baseline'`."
            )
            raise ValueError(
                msg,
            )
        if output_unit not in {"mpp", "power", "baseline", None}:
            msg = (
                "Invalid output_unit: argument accepts only one of the "
                "following options: `'mpp'`, `'power'`, `'baseline'`, "
                "or None (to return all units)."
            )
            raise ValueError(
                msg,
            )
        if baseline_mpp is None and input_unit == "mpp":
            msg = (
                "Missing 'mpp': `input_unit` has been set to 'mpp' while "
                "there is no information about 'mpp' in WSI meta data."
            )
            raise ValueError(
                msg,
            )
        if baseline_power is None and input_unit == "power":
            msg = (
                "Missing 'objective_power': `input_unit` has been set to 'power' "
                "while there is no information about 'objective_power' "
                "in WSI meta data."
            )
            raise ValueError(
                msg,
            )

    def _prepare_output_dict(
        self: WSIReader,
        input_unit: Units,
        input_res: Resolution,
        baseline_mpp: Resolution,
        baseline_power: Resolution,
    ) -> dict:
        """Calculate output_res as dictionary based on input_unit and resolution."""
        output_dict = {
            "mpp": None,
            "power": None,
            "baseline": None,
        }
        if input_unit == "mpp":
            if isinstance(input_res, (list, tuple, np.ndarray)):
                output_dict["mpp"] = np.array(input_res)
            else:
                output_dict["mpp"] = np.array([input_res, input_res])
            output_dict["baseline"] = baseline_mpp[0] / output_dict["mpp"][0]
            if baseline_power is not None:
                output_dict["power"] = output_dict["baseline"] * baseline_power
            return output_dict
        if input_unit == "power":
            output_dict["baseline"] = input_res / baseline_power
            output_dict["power"] = input_res
        elif input_unit == "level":
            level_scales = self.info.relative_level_scales(input_res, input_unit)
            output_dict["baseline"] = level_scales[0]
            if baseline_power is not None:
                output_dict["power"] = output_dict["baseline"] * baseline_power
        else:  # input_unit == 'baseline'
            output_dict["baseline"] = input_res
            if baseline_power is not None:
                output_dict["power"] = baseline_power * output_dict["baseline"]

        if baseline_mpp is not None:
            output_dict["mpp"] = baseline_mpp / output_dict["baseline"]

        return output_dict

    def convert_resolution_units(
        self: WSIReader,
        input_res: Resolution,
        input_unit: Units,
        output_unit: Units | None = None,
    ) -> Resolution:
        """Converts resolution value between different units.

        This function accepts a resolution and its units in the input
        and converts it to all other units ('mpp', 'power', 'baseline').
        To achieve resolution in 'mpp' and 'power' units in the output,
        WSI metadata should contain `mpp` and `objective_power`
        information, respectively.

        Args:
            input_res (Resolution):
                the resolution which we want to convert to the other
                units.
            input_unit (Units):
                The unit of the input resolution (`input_res`).
                Acceptable input_units are 'mpp', 'power', 'baseline',
                and 'level'. output_unit (str): the desired unit to
                which we want to convert the `input_res`. Acceptable
                values for `output_unit` are: 'mpp', 'power', and
                'baseline'. If `output_unit` is not provided, all the
                conversions to all the mentioned units will be
                returned in a dictionary.
            output_unit (Units):
                Units of scale, Supported units are:
                - microns per pixel ('mpp')
                - objective power ('power')
                - pyramid / resolution level ('level')
                - pixels per baseline pixel ("baseline")


        Returns:
            output_res (Resolution):
                Either a float which is the converted `input_res` to the
                desired `output_unit` or a dictionary containing the
                converted `input_res` to all acceptable units (`'mpp'`,
                `'power'`, `'baseline'`). If there is not enough metadata
                to calculate a unit (like `mpp` or `power`), they
                will be set to None in the dictionary.

        """
        baseline_mpp = self.info.mpp
        baseline_power = self.info.objective_power

        self._check_unit_conversion_integrity(
            input_unit,
            output_unit,
            baseline_mpp,
            baseline_power,
        )

        output_dict = self._prepare_output_dict(
            input_unit,
            input_res,
            baseline_mpp,
            baseline_power,
        )
        out_res = output_dict[output_unit] if output_unit is not None else output_dict
        if out_res is None:
            logger.warning(
                "Although unit conversion from input_unit has been done, the requested "
                "output_unit is returned as None. Probably due to missing 'mpp' or "
                "'objective_power' in slide's meta data.",
            )
        return out_res

    def _find_tile_params(
        self: WSIReader,
        tile_objective_value: Number,
    ) -> tuple[int, IntPair, int, Number]:
        """Find the params for save tiles."""
        rescale = self.info.objective_power / tile_objective_value
        if not rescale.is_integer():
            msg = (
                "Tile objective value must be an integer multiple of the "
                "objective power of the slide.",
            )
            raise ValueError(
                msg,
            )
        scale_levels_available = [
            np.log2(np.round(x, 3)) for x in self.info.level_downsamples
        ]
        try:
            level_scale = np.log2(rescale)
            if not level_scale.is_integer():
                raise ValueError  # noqa: TRY301
            level_scale = np.int_(level_scale)
            if level_scale not in scale_levels_available:
                raise IndexError  # noqa: TRY301
            level = scale_levels_available.index(level_scale)
            slide_dimension = self.info.level_dimensions[level]
            rescale = 1
        # Raise index error if desired pyramid level not embedded
        # in level_dimensions
        except IndexError:
            level = 0
            slide_dimension = self.info.level_dimensions[level]
            rescale = np.int_(rescale)
            logger.warning(
                "Reading WSI at level 0. Desired tile_objective_value %s "
                "not available.",
                str(tile_objective_value),
            )
        except ValueError:
            level = 0
            slide_dimension = self.info.level_dimensions[level]
            rescale = 1
            logger.warning(
                "Reading WSI at level 0. Reading at tile_objective_value %s "
                "not allowed.",
                str(tile_objective_value),
            )
            tile_objective_value = self.info.objective_power

        return level, slide_dimension, rescale, tile_objective_value

    def read_rect_at_resolution(
        self: WSIReader,
        location: NumPair,
        size: NumPair,
        resolution: Resolution = 0,
        units: Units = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: Number | Iterable[NumPair] = 0,
        **kwargs: dict,
    ) -> np.ndarray:
        """Helper to perform `read_rect` at resolution.

        In actuality, `read_rect` at resolution is synonymous with
        calling `read_bound` at resolution because `size` has always
        been within the resolution system.

        """
        tl = np.array(location)
        br = location + np.array(size)
        bounds = np.concatenate([tl, br])
        return self.read_bounds(
            bounds,
            resolution=resolution,
            units=units,
            interpolation=interpolation,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
            coord_space="resolution",
            **kwargs,
        )

    def read_rect(
        self: WSIReader,
        location: IntPair,
        size: IntPair,
        resolution: Resolution = 0,
        units: Units = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: int | tuple[int, int] = 0,
        coord_space: str = "baseline",
        **kwargs: dict,
    ) -> np.ndarray:
        """Read a region of the whole slide image at a location and size.

        Location is in terms of the baseline image (level 0  / maximum
        resolution), and size is the output image size.

        Reads can be performed at different resolutions by supplying a
        pair of arguments for the resolution and the units of
        resolution. If metadata does not specify `mpp` or
        `objective_power` then `baseline` units should be selected with
        resolution 1.0

        The field of view varies with resolution. For a fixed field of
        view see :func:`read_bounds`.

        Args:
            location (IntPair):
                (x, y) tuple giving the top left pixel in the baseline
                (level 0) reference frame.
            size (IntPair):
                (width, height) tuple giving the desired output image
                size.
            resolution (Resolution):
                Resolution at which to read the image, default = 0.
                Either a single number or a sequence of two numbers for
                x and y are valid. This value is in terms of the
                corresponding units. For example: resolution=0.5 and
                units="mpp" will read the slide at 0.5 microns
                per-pixel, and resolution=3, units="level" will read at
                level at pyramid level / resolution layer 3.
            units (Units):
                The units of resolution, default = "level". Supported
                units are: microns per pixel (mpp), objective power
                (power), pyramid / resolution level (level), pixels per
                baseline pixel (baseline).
            interpolation (str):
                Method to use when resampling the output image. Possible
                values are "linear", "cubic", "lanczos", "area", and
                "optimise". Defaults to 'optimise' which will use cubic
                interpolation for upscaling and area interpolation for
                downscaling to avoid moiré patterns.
            pad_mode (str):
                Method to use when padding at the edges of the image.
                Defaults to 'constant'. See :func:`numpy.pad` for
                available modes.
            pad_constant_values (int, tuple(int)):
                Constant values to use when padding with constant pad mode.
                Passed to the :func:`numpy.pad` `constant_values` argument.
                Default is 0.
            coord_space (str):
                Defaults to "baseline". This is a flag to indicate if
                the input `bounds` is in the baseline coordinate system
                ("baseline") or is in the requested resolution system
                ("resolution").
            **kwargs (dict):
                Extra key-word arguments for reader specific parameters.
                Currently only used by VirtualWSIReader. See class
                docstrings for more information.

        Returns:
            :class:`numpy.ndarray`:
                Array of size MxNx3 M=size[0], N=size[1]

        Example:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> # Load a WSI image
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
            >>> location = (0, 0)
            >>> size = (256, 256)
            >>> # Read a region at level 0 (baseline / full resolution)
            >>> img = wsi.read_rect(location, size)
            >>> # Read a region at 0.5 microns per pixel (mpp)
            >>> img = wsi.read_rect(location, size, 0.5, "mpp")
            >>> # This could also be written more verbosely as follows
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=(0.5, 0.5),
            ...     units="mpp",
            ... )

        Note: The field of view varies with resolution when using
        :func:`read_rect`.

        .. figure:: ../images/read_rect_tissue.png
            :width: 512
            :alt: Diagram illustrating read_rect

        As the location is in the baseline reference frame but the size
        (width and height) is the output image size, the field of view
        therefore changes as resolution changes.

        If the WSI does not have a resolution layer corresponding
        exactly to the requested resolution (shown above in white with a
        dashed outline), a larger resolution is downscaled to achieve
        the correct requested output resolution.

        If the requested resolution is higher than the baseline (maximum
        resultion of the image), then bicubic interpolation is applied
        to the output image.

        .. figure:: ../images/read_rect-interpolated-reads.png
            :width: 512
            :alt: Diagram illustrating read_rect interpolting between levels

        When reading between the levels stored in the WSI, the
        coordinates of the requested region are projected to the next
        highest resolution. This resolution is then decoded and
        downsampled to produce the desired output. This is a major
        source of variability in the time take to perform a read
        operation. Reads which require reading a large region before
        downsampling will be significantly slower than reading at a
        fixed level.

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> # Load a WSI image
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
            >>> location = (0, 0)
            >>> size = (256, 256)
            >>> # The resolution can be different in x and y, e.g.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=(0.5, 0.75),
            ...     units="mpp",
            ... )
            >>> # Several units can be used including: objective power,
            >>> # microns per pixel, pyramid/resolution level, and
            >>> # fraction of baseline.
            >>> # E.g. Read a region at an objective power of 10x
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=10,
            ...     units="power",
            ... )
            >>> # Read a region at pyramid / resolution level 1
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=1,
            ...     units="level",
            ... )
            >>> # Read at a fractional level, this will linearly
            >>> # interpolate the downsampling factor between levels.
            >>> # E.g. if levels 0 and 1 have a downsampling of 1x and
            >>> # 2x of baseline, then level 0.5 will correspond to a
            >>> # downsampling factor 1.5x of baseline.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.5,
            ...     units="level",
            ... )
            >>> # Read a region at half of the full / baseline
            >>> # resolution.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.5,
            ...     units="baseline",
            ... )
            >>> # Read at a higher resolution than the baseline
            >>> # (interpolation applied to output)
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=1.25,
            ...     units="baseline",
            ... )
            >>> # Assuming the image has a native mpp of 0.5,
            >>> # interpolation will be applied here.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.25,
            ...     units="mpp",
            ... )

        """
        raise NotImplementedError

    def read_bounds(
        self: WSIReader,
        bounds: Bounds,
        resolution: Resolution = 0,
        units: Units = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: Number | Iterable[NumPair] = 0,
        coord_space: str = "baseline",
        **kwargs: dict,
    ) -> np.ndarray:
        """Read a region of the whole slide image within given bounds.

        Bounds are in terms of the baseline image (level 0  / maximum
        resolution).

        Reads can be performed at different resolutions by supplying a
        pair of arguments for the resolution and the units of
        resolution. If metadata does not specify `mpp` or
        `objective_power` then `baseline` units should be selected with
        resolution 1.0

        The output image size may be different to the width and height
        of the bounds as the resolution will affect this. To read a
        region with a fixed output image size see :func:`read_rect`.

        Args:
            bounds (IntBounds):
                By default, this is a tuple of (start_x, start_y, end_x,
                end_y) i.e. (left, top, right, bottom) of the region in
                baseline reference frame. However, with
                `coord_space="resolution"`, the bound is expected to be
                at the requested resolution system.
            resolution (Resolution):
                Resolution at which to read the image, default = 0.
                Either a single number or a sequence of two numbers for
                x and y are valid. This value is in terms of the
                corresponding units. For example: resolution=0.5 and
                units="mpp" will read the slide at 0.5 microns
                per-pixel, and resolution=3, units="level" will read at
                level at pyramid level / resolution layer 3.
            units (Units):
                Units of resolution, default="level". Supported units
                are: microns per pixel (mpp), objective power (power),
                pyramid / resolution level (level), pixels per baseline
                pixel (baseline).
            interpolation (str):
                Method to use when resampling the output image. Possible
                values are "linear", "cubic", "lanczos", "area", and
                "optimise". Defaults to 'optimise' which will use cubic
                interpolation for upscaling and area interpolation for
                downscaling to avoid moiré patterns.
            pad_mode (str):
                Method to use when padding at the edges of the image.
                Defaults to 'constant'. See :func:`numpy.pad` for
                available modes.
            pad_constant_values (int, tuple(int)):
                Constant values to use when padding with constant pad mode.
                Passed to the :func:`numpy.pad` `constant_values` argument.
                Default is 0.
            coord_space (str):
                Defaults to "baseline". This is a flag to indicate if
                the input `bounds` is in the baseline coordinate system
                ("baseline") or is in the requested resolution system
                ("resolution").
            **kwargs (dict):
                Extra key-word arguments for reader specific parameters.
                Currently only used by :obj:`VirtualWSIReader`. See
                class docstrings for more information.

        Returns:
            :class:`numpy.ndarray`:
                Array of size MxNx3 M=end_h-start_h, N=end_w-start_w

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> from matplotlib import pyplot as plt
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
            >>> # Read a region at level 0 (baseline / full resolution)
            >>> bounds = [1000, 2000, 2000, 3000]
            >>> img = wsi.read_bounds(bounds)
            >>> plt.imshow(img)
            >>> # This could also be written more verbosely as follows
            >>> img = wsi.read_bounds(
            ...     bounds,
            ...     resolution=0,
            ...     units="level",
            ... )
            >>> plt.imshow(img)

        Note: The field of view remains the same as resolution is varied
        when using :func:`read_bounds`.

        .. figure:: ../images/read_bounds_tissue.png
            :width: 512
            :alt: Diagram illustrating read_bounds

        This is because the bounds are in the baseline (level 0)
        reference frame. Therefore, varying the resolution does not
        change what is visible within the output image.

        If the WSI does not have a resolution layer corresponding
        exactly to the requested resolution (shown above in white with a
        dashed outline), a larger resolution is downscaled to achieve
        the correct requested output resolution.

        If the requested resolution is higher than the baseline (maximum
        resultion of the image), then bicubic interpolation is applied
        to the output image.

        """
        raise NotImplementedError

    def read_region(
        self: WSIReader,
        location: IntPair,
        level: int,
        size: IntPair,
    ) -> np.ndarray:
        """Read a region of the whole slide image (OpenSlide format args).

        This function is to help with writing code which is backwards
        compatible with OpenSlide. As such, it has the same arguments.

        This internally calls :func:`read_rect` which should be
        implemented by any :class:`WSIReader` subclass. Therefore, some
        WSI formats which are not supported by OpenSlide, such as Omnyx
        JP2 files, may also be readable with the same syntax.

        Args:
            location (IntPair):
                (x, y) tuple giving the top left pixel in the level 0
                reference frame.
            level (int):
                The level number.
            size (IntPair):
                (width, height) tuple giving the region size.

        Returns:
            :class:`numpy.ndarray`:
                Array of size MxNx3.

        """
        return self.read_rect(
            location=location,
            size=size,
            resolution=level,
            units="level",
        )

    def slide_thumbnail(
        self: WSIReader,
        resolution: Resolution = 1.25,
        units: Units = "power",
    ) -> np.ndarray:
        """Read the whole slide image thumbnail (1.25x by default).

        For more information on resolution and units see
        :func:`read_rect`

        Args:
            resolution (Resolution):
                Resolution to read thumbnail at, default = 1.25
                (objective power)
            units (Units):
                Resolution units, default="power".

        Returns:
            :class:`numpy.ndarray`:
                Thumbnail image.

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
            >>> slide_thumbnail = wsi.slide_thumbnail()

        """
        slide_dimensions = self.info.slide_dimensions
        bounds = (0, 0, *slide_dimensions)
        return self.read_bounds(bounds, resolution=resolution, units=units)

    def tissue_mask(
        self: WSIReader,
        method: str = "otsu",
        resolution: Resolution = 1.25,
        units: Units = "power",
        **masker_kwargs: dict,
    ) -> VirtualWSIReader:
        """Create a tissue mask and wrap it in a VirtualWSIReader.

        For the morphological method, mpp is used for calculating the
        scale of the morphological operations. If no mpp is available,
        objective power is used instead to estimate a good scale. This
        can be overridden with a custom size, via passing a
        `kernel_size` key-word argument in `masker_kwargs`, see
        :class:`tissuemask.MorphologicalMasker` for more.


        Args:
            method (str):
                Method to use for creating the mask. Defaults
                to 'otsu'. Methods are: otsu, morphological.
            resolution (float):
                Resolution to produce the mask at.
                Defaults to 1.25.
            units (Units):
                Units of resolution. Defaults to "power".
            **masker_kwargs:
                Extra kwargs passed to the masker class.

        """
        from tiatoolbox.tools import tissuemask  # noqa: PLC0415

        thumbnail = self.slide_thumbnail(resolution, units)
        if method not in ["otsu", "morphological"]:
            msg = f"Invalid tissue masking method: {method}."
            raise ValueError(msg)
        if method == "morphological":
            mpp = None
            power = None
            if units == "mpp":
                mpp = resolution
            elif units == "power":
                power = resolution
            masker = tissuemask.MorphologicalMasker(
                mpp=mpp,
                power=power,
                **masker_kwargs,
            )
        elif method == "otsu":
            masker = tissuemask.OtsuTissueMasker(**masker_kwargs)
        mask_img = masker.fit_transform([thumbnail])[0]
        return VirtualWSIReader(mask_img.astype(np.uint8), info=self.info, mode="bool")

    def save_tiles(
        self: WSIReader,
        output_dir: str | Path = "tiles",
        tile_objective_value: int = 20,
        tile_read_size: tuple[int, int] = (5000, 5000),
        tile_format: str = ".jpg",
        *,
        verbose: bool = False,
    ) -> None:
        """Generate image tiles from whole slide images.

        Args:
            output_dir(str or :obj:`Path`):
                Output directory to save the tiles.
            tile_objective_value (int):
                Objective value at which tile is generated, default = 20
            tile_read_size (tuple(int)):
                Tile (width, height), default = (5000, 5000).
            tile_format (str):
                File format to save image tiles, defaults = ".jpg".
            verbose (bool):
                Print output, default=False

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
            >>> wsi.save_tiles(output_dir='./dev_test',
            ...     tile_objective_value=10,
            ...     tile_read_size=(2000, 2000))

            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
            >>> slide_param = wsi.info

        """
        if verbose:
            logger.setLevel(logging.DEBUG)

        logger.debug("Processing %s.", self.input_path.name)

        output_dir = Path(output_dir, self.input_path.name)

        level, slide_dimension, rescale, tile_objective_value = self._find_tile_params(
            tile_objective_value,
        )

        tile_read_size = np.multiply(tile_read_size, rescale)
        slide_h = slide_dimension[1]
        slide_w = slide_dimension[0]
        tile_h = tile_read_size[1]
        tile_w = tile_read_size[0]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)
        data = []

        vertical_tiles = math.ceil((slide_h - tile_h) / tile_h + 1)
        horizontal_tiles = math.ceil((slide_w - tile_w) / tile_w + 1)
        for iter_tot, (h, w) in enumerate(np.ndindex(vertical_tiles, horizontal_tiles)):
            start_h = h * tile_h
            end_h = (h * tile_h) + tile_h
            start_w = w * tile_w
            end_w = (w * tile_w) + tile_w

            end_h = min(end_h, slide_h)
            end_w = min(end_w, slide_w)

            # convert to baseline reference frame
            bounds = start_w, start_h, end_w, end_h
            baseline_bounds = tuple(
                bound * int(np.round(self.info.level_downsamples[level], 3))
                for bound in bounds
            )
            # Read image region
            im = self.read_bounds(baseline_bounds, level)

            logger.debug(
                "Tile %d:  start_w: %d, end_w: %d, start_h: %d, end_h: %d, "
                "width: %d, height: %d",
                iter_tot,
                start_w,
                end_w,
                start_h,
                end_h,
                end_w - start_w,
                end_h - start_h,
            )

            # Rescale to the correct objective value
            if rescale != 1:
                im = utils.transforms.imresize(img=im, scale_factor=1 / rescale)

            img_save_name = (
                "_".join(
                    [
                        "Tile",
                        str(tile_objective_value),
                        str(int(start_w / rescale)),
                        str(int(start_h / rescale)),
                    ],
                )
                + tile_format
            )

            utils.imwrite(image_path=output_dir / img_save_name, img=im)

            data.append(
                [
                    iter_tot,
                    img_save_name,
                    int(start_w / rescale),
                    int(end_w / rescale),
                    int(start_h / rescale),
                    int(end_h / rescale),
                    im.shape[0],
                    im.shape[1],
                ],
            )

        # Save information on each slide to relate to the whole slide image
        save_tiles_df = pd.DataFrame(
            data,
            columns=[
                "iter",
                "Tile_Name",
                "start_w",
                "end_w",
                "start_h",
                "end_h",
                "size_w",
                "size_h",
            ],
        )
        save_tiles_df.to_csv(output_dir / "Output.csv", index=False)

        # Save slide thumbnail
        slide_thumb = self.slide_thumbnail()
        utils.imwrite(output_dir / f"slide_thumbnail{tile_format}", img=slide_thumb)

        if verbose:
            logger.setLevel(logging.INFO)

    @staticmethod
    def _estimate_mpp_objective_power(
        objective_power: float | None,
        mpp: float | tuple[float] | tuple[float, float] | None,
    ) -> tuple[
        float | None,
        float | tuple[float] | tuple[float, float] | np.ndarray | None,
    ]:
        """Estimate objective power or mpp if one of these is available."""
        if objective_power is not None and mpp is not None:
            return objective_power, mpp  # use slide metadata

        if objective_power is None and mpp is None:
            logger.warning(
                "Metadata: Unable to determine objective power "
                "or microns-per-pixel (MPP)."
            )
            return objective_power, mpp  # Unable to determine

        if objective_power is None:
            objective_power = utils.misc.mpp2common_objective_power(
                mpp=float(np.mean(mpp)),
            )
            logger.warning(
                "Metadata: Objective power inferred from microns-per-pixel (MPP).",
            )
            return objective_power, mpp  # estimate objective power from mpp

        # mpp is None
        mpp = utils.misc.objective_power2mpp(
            objective_power=np.asarray(objective_power),  # ensures ndarray output
        )
        # float ensures expected output type
        mpp = np.array([float(mpp), float(mpp)]) if mpp.ndim == 0 else mpp
        logger.warning(
            "Metadata: microns-per-pixel (MPP) inferred from Objective power.",
        )
        return objective_power, mpp  # estimate objective power from mpp


class ArrayView:
    """An object for viewing a zarr array with a different index ordering.

    Used to allow YXS index order reads for arrays with axes in other
    orders such as SYX. Currently supported axes are:
    - YXS
    - SYX

    """

    def __init__(self: ArrayView, array: zarr.Array, axes: str) -> None:
        """Initialise the view object.

        Args:
            array (zarr.Array):
                Zarr Array to read from.
            axes (str):
                Axes ordering string. Allowed values are YXS and SYX.

        """
        self.array = array
        self.axes = axes
        self._shape = dict(zip(self.axes, self.array.shape, strict=False))

    @property
    def shape(self: ArrayView) -> tuple:
        """Return array shape."""
        try:
            return tuple(self._shape[c] for c in "YXC")
        except KeyError:
            return tuple(self._shape[c] for c in "YXS")

    def __getitem__(self: ArrayView, index: int) -> np.ndarray:
        """Get an item from the dataset."""
        # Normalize to a tuple of length = len(self.axes)
        if not isinstance(index, tuple):
            index = (index,)
        while len(index) < len(self.axes):
            index = (*index, slice(None))

        if self.axes in ("YXS", "YXC"):
            return self.array[index]
        if self.axes in ("SYX", "CYX"):
            y, x, s = index
            index = (s, y, x)
            return np.rollaxis(self.array[index], 0, 3)
        msg = f"Unsupported axes `{self.axes}`."
        raise IndexError(msg)


class VirtualWSIReader(WSIReader):
    """Class for reading non-pyramidal images e.g., visual fields.

    Supported formats:

    - .jpg
    - .png
    - :class:`numpy.ndarray`

    This reader uses :func:`tiatoolbox.utils.image.sub_pixel_read` to
    allow reading low resolution images as if they are larger i.e. with
    'virtual' pyramid resolutions. This is useful for reading low
    resolution masks as if they were stretched to overlay a higher
    resolution WSI.

    Extra key-word arguments given to :func:`~WSIReader.read_region` and
    :func:`~WSIReader.read_bounds` will be passed to
    :func:`~tiatoolbox.utils.image.sub_pixel_read`.

    Attributes:
        img (:class:`numpy.ndarray`):
            Input image as :class:`numpy.ndarray`.
        mode (str):
            Mode of the input image. Default is 'rgb'. Allowed values
            are: rgb, bool, feature. "rgb" mode supports bright-field color images.
            "bool" mode supports binary masks,
            interpolation in this case will be "nearest" instead of "bicubic".
            "feature" mode allows multichannel features.

    Args:
        input_img (str, :obj:`Path`, :class:`numpy.ndarray`):
            Input path to WSI.
        info (WSIMeta):
            Metadata for the virtual wsi.
        mode (str):
            Mode of the input image. Default is 'rgb'. Allowed values
            are: rgb, bool, feature. "rgb" mode supports bright-field color images.
            "bool" mode supports binary masks,
            interpolation in this case will be "nearest" instead of "bicubic".
            "feature" mode allows multichannel features.
        post_proc (str, callable):
            Post-processing function to apply to the output image.

    """

    def __init__(
        self: VirtualWSIReader,
        input_img: str | Path | np.ndarray,
        mpp: tuple[Number, Number] | None = None,
        power: Number | None = None,
        info: WSIMeta | None = None,
        mode: str = "rgb",
        post_proc: str | callable | None = "auto",
    ) -> None:
        """Initialize :class:`VirtualWSIReader`."""
        super().__init__(
            input_img=input_img,
            mpp=mpp,
            power=power,
            post_proc=post_proc,
        )
        if mode.lower() not in ["rgb", "bool", "feature"]:
            msg = "Invalid mode."
            raise ValueError(msg)

        if isinstance(input_img, np.ndarray):
            self.img = input_img
        else:
            self.img = utils.imread(self.input_path)

        _min_image_ndims = (
            2  # Minimum number of dimensions required for an image (H, W)
        )
        # Reject 1D (or otherwise <2D) inputs early with a clear message.
        if self.img.ndim < _min_image_ndims:
            msg = "Input image must be 2D (H, W) or 3D (H, W, C). Got a 1D array."
            raise ValueError(msg)

        if mode != "bool" and (
            self.img.ndim == 2 or self.img.shape[2] not in [3, 4]  # noqa: PLR2004
        ):
            logger.warning(
                "The image mode is set to 'feature' as the input"
                " dimensions do not match with binary mask or RGB/RGBA.",
            )
            mode = "feature"

        self.mode = mode.lower()

        if info is not None:
            self._m_info = info

    def _info(self: VirtualWSIReader) -> WSIMeta:
        """Visual Field metadata getter.

        This generates a WSIMeta object for the slide if none exists.
        There is 1 level with dimensions equal to the image and no mpp,
        objective power, or vendor data.


        Returns:
            WSIMeta:
                Metadata information.

        """
        param = WSIMeta(
            file_path=self.input_path,
            axes="YSX",
            objective_power=None,
            # align to XY to match with OpenSlide
            slide_dimensions=self.img.shape[:2][::-1],
            level_count=1,
            level_dimensions=(self.img.shape[:2][::-1],),
            level_downsamples=[1.0],
            vendor=None,
            mpp=None,
            raw=None,
        )
        if self._m_info is None:
            self._m_info = param
        return self._m_info

    def _find_params_from_baseline(
        self: VirtualWSIReader,
        location: IntPair,
        baseline_read_size: IntPair,
    ) -> tuple[IntPair, IntPair]:
        """Convert read parameters from (virtual) baseline coordinates.

        Args:
            location (IntPair):
                Location of the location to read in (virtual) baseline
                coordinates.
            baseline_read_size (IntPair):
                Size of the region to read in (virtual) baseline
                coordinates.

        Returns:
            tuple(IntPair, IntPair):
                Baseline image location and read size.

        """
        baseline_size = np.array(self.info.slide_dimensions)
        image_size = np.array(self.img.shape[:2][::-1])
        size_ratio = image_size / baseline_size
        image_location = np.array(location, dtype=np.float32) * size_ratio
        read_size = np.array(baseline_read_size) * size_ratio
        return image_location, read_size

    def read_rect(
        self: VirtualWSIReader,
        location: IntPair,
        size: IntPair,
        resolution: Resolution = 0,
        units: Units = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: int | IntPair = 0,
        coord_space: str = "baseline",
        **kwargs: dict,
    ) -> np.ndarray:
        """Read a region of the whole slide image at a location and size.

        Location is in terms of the baseline image (level 0  / maximum
        resolution), and size is the output image size.

        Reads can be performed at different resolutions by supplying a
        pair of arguments for the resolution and the units of
        resolution. If metadata does not specify `mpp` or
        `objective_power` then `baseline` units should be selected with
        resolution 1.0

        The field of view varies with resolution. For a fixed field of
        view see :func:`read_bounds`.

        Args:
            location (IntPair):
                (x, y) tuple giving the top left pixel in the baseline
                (level 0) reference frame.
            size (IntPair):
                (width, height) tuple giving the desired output image
                size.
            resolution (Resolution):
                Resolution at which to read the image, default = 0.
                Either a single number or a sequence of two numbers for
                x and y are valid. This value is in terms of the
                corresponding units. For example: resolution=0.5 and
                units="mpp" will read the slide at 0.5 microns
                per-pixel, and resolution=3, units="level" will read at
                level at pyramid level / resolution layer 3.
            units (Units):
                The units of resolution, default = "level". Supported
                units are: microns per pixel (mpp), objective power
                (power), pyramid / resolution level (level), pixels per
                baseline pixel (baseline).
            interpolation (str):
                Method to use when resampling the output image. Possible
                values are "linear", "cubic", "lanczos", "area", and
                "optimise". Defaults to 'optimise' which will use cubic
                interpolation for upscaling and area interpolation for
                downscaling to avoid moiré patterns.
            pad_mode (str):
                Method to use when padding at the edges of the image.
                Defaults to 'constant'. See :func:`numpy.pad` for
                available modes.
            pad_constant_values (int, tuple(int)):
                Constant values to use when padding with constant pad mode.
                Passed to the :func:`numpy.pad` `constant_values` argument.
                Default is 0.
            coord_space (str):
                Defaults to "baseline". This is a flag to indicate if
                the input `bounds` is in the baseline coordinate system
                ("baseline") or is in the requested resolution system
                ("resolution").
            **kwargs (dict):
                Extra key-word arguments for reader specific parameters.
                Currently, only used by VirtualWSIReader. See class
                docstrings for more information.

        Returns:
            :class:`numpy.ndarray`:
                Array of size MxNx3 M=size[0], N=size[1]

        Example:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> # Load a WSI image
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
            >>> location = (0, 0)
            >>> size = (256, 256)
            >>> # Read a region at level 0 (baseline / full resolution)
            >>> img = wsi.read_rect(location, size)
            >>> # Read a region at 0.5 microns per pixel (mpp)
            >>> img = wsi.read_rect(location, size, 0.5, "mpp")
            >>> # This could also be written more verbosely as follows
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=(0.5, 0.5),
            ...     units="mpp",
            ... )

        Note: The field of view varies with resolution when using
        :func:`read_rect`.

        .. figure:: ../images/read_rect_tissue.png
            :width: 512
            :alt: Diagram illustrating read_rect

        As the location is in the baseline reference frame but the size
        (width and height) is the output image size, the field of view
        therefore changes as resolution changes.

        If the WSI does not have a resolution layer corresponding
        exactly to the requested resolution (shown above in white with a
        dashed outline), a larger resolution is downscaled to achieve
        the correct requested output resolution.

        If the requested resolution is higher than the baseline (maximum
        resultion of the image), then bicubic interpolation is applied
        to the output image.

        .. figure:: ../images/read_rect-interpolated-reads.png
            :width: 512
            :alt: Diagram illustrating read_rect interpolting between levels

        When reading between the levels stored in the WSI, the
        coordinates of the requested region are projected to the next
        highest resolution. This resolution is then decoded and
        downsampled to produce the desired output. This is a major
        source of variability in the time take to perform a read
        operation. Reads which require reading a large region before
        downsampling will be significantly slower than reading at a
        fixed level.

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> # Load a WSI image
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
            >>> location = (0, 0)
            >>> size = (256, 256)
            >>> # The resolution can be different in x and y, e.g.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=(0.5, 0.75),
            ...     units="mpp",
            ... )
            >>> # Several units can be used including: objective power,
            >>> # microns per pixel, pyramid/resolution level, and
            >>> # fraction of baseline.
            >>> # E.g. Read a region at an objective power of 10x
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=10,
            ...     units="power",
            ... )
            >>> # Read a region at pyramid / resolution level 1
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=1,
            ...     units="level",
            ... )
            >>> # Read at a fractional level, this will linearly
            >>> # interpolate the downsampling factor between levels.
            >>> # E.g. if levels 0 and 1 have a downsampling of 1x and
            >>> # 2x of baseline, then level 0.5 will correspond to a
            >>> # downsampling factor 1.5x of baseline.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.5,
            ...     units="level",
            ... )
            >>> # Read a region at half of the full / baseline
            >>> # resolution.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.5,
            ...     units="baseline",
            ... )
            >>> # Read at a higher resolution than the baseline
            >>> # (interpolation applied to output)
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=1.25,
            ...     units="baseline",
            ... )
            >>> # Assuming the image has a native mpp of 0.5,
            >>> # interpolation will be applied here.
            >>> img = wsi.read_rect(
            ...     location,
            ...     size,
            ...     resolution=0.25,
            ...     units="mpp",
            ... )

        """
        if coord_space == "resolution":
            return self.read_rect_at_resolution(
                location,
                size,
                resolution=resolution,
                units=units,
                interpolation=interpolation,
                pad_mode=pad_mode,
                pad_constant_values=pad_constant_values,
            )

        # Find parameters for optimal read
        (
            _,
            _,
            _,
            _,
            baseline_read_size,
        ) = self.find_read_rect_params(
            location=location,
            size=size,
            resolution=resolution,
            units=units,
        )

        image_location, image_read_size = self._find_params_from_baseline(
            location,
            baseline_read_size,
        )

        bounds = utils.transforms.locsize2bounds(
            location=image_location,
            size=image_read_size,
        )

        output_size = None if interpolation in [None, "none"] else size
        im_region = utils.image.sub_pixel_read(
            self.img,
            bounds,
            output_size=output_size,
            interpolation=interpolation,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
            read_kwargs=kwargs,
            pad_at_baseline=False,
        )

        if self.mode == "rgb":
            if self.post_proc is not None:
                im_region = self.post_proc(im_region)
            return utils.transforms.background_composite(image=im_region, alpha=False)
        return im_region

    def read_bounds(
        self: VirtualWSIReader,
        bounds: IntBounds,
        resolution: Resolution = 0,
        units: Units = "level",
        interpolation: str = "optimise",
        pad_mode: str = "constant",
        pad_constant_values: int | IntPair = 0,
        coord_space: str = "baseline",
        **kwargs: dict,
    ) -> np.ndarray:
        """Read a region of the whole slide image within given bounds.

        Bounds are in terms of the baseline image (level 0  / maximum
        resolution).

        Reads can be performed at different resolutions by supplying a
        pair of arguments for the resolution and the units of
        resolution. If metadata does not specify `mpp` or
        `objective_power` then `baseline` units should be selected with
        resolution 1.0

        The output image size may be different to the width and height
        of the bounds as the resolution will affect this. To read a
        region with a fixed output image size see :func:`read_rect`.

        Args:
            bounds (IntBounds):
                By default, this is a tuple of (start_x, start_y, end_x,
                end_y) i.e. (left, top, right, bottom) of the region in
                baseline reference frame. However, with
                `coord_space="resolution"`, the bound is expected to be
                at the requested resolution system.
            resolution (Resolution):
                Resolution at which to read the image, default = 0.
                Either a single number or a sequence of two numbers for
                x and y are valid. This value is in terms of the
                corresponding units. For example: resolution=0.5 and
                units="mpp" will read the slide at 0.5 microns
                per-pixel, and resolution=3, units="level" will read at
                level at pyramid level / resolution layer 3.
            units (Units):
                Units of resolution, default="level". Supported units
                are: microns per pixel (mpp), objective power (power),
                pyramid / resolution level (level), pixels per baseline
                pixel (baseline).
            interpolation (str):
                Method to use when resampling the output image. Possible
                values are "linear", "cubic", "lanczos", "area", and
                "optimise". Defaults to 'optimise' which will use cubic
                interpolation for upscaling and area interpolation for
                downscaling to avoid moiré patterns.
            pad_mode (str):
                Method to use when padding at the edges of the image.
                Defaults to 'constant'. See :func:`numpy.pad` for
                available modes.
            pad_constant_values (int, tuple(int)):
                Constant values to use when padding with constant pad mode.
                Passed to the :func:`numpy.pad` `constant_values` argument.
                Default is 0.
            coord_space (str):
                Defaults to "baseline". This is a flag to indicate if
                the input `bounds` is in the baseline coordinate system
                ("baseline") or is in the requested resolution system
                ("resolution").
            **kwargs (dict):
                Extra key-word arguments for reader specific parameters.
                Currently only used by :obj:`VirtualWSIReader`. See
                class docstrings for more information.

        Returns:
            :class:`numpy.ndarray`:
                Array of size MxNx3 M=end_h-start_h, N=end_w-start_w

        Examples:
            >>> from tiatoolbox.wsicore.wsireader import WSIReader
            >>> from matplotlib import pyplot as plt
            >>> wsi = WSIReader.open(input_img="./CMU-1.ndpi")
            >>> # Read a region at level 0 (baseline / full resolution)
            >>> bounds = [1000, 2000, 2000, 3000]
            >>> img = wsi.read_bounds(bounds)
            >>> plt.imshow(img)
            >>> # This could also be written more verbosely as follows
            >>> img = wsi.read_bounds(
            ...     bounds,
            ...     resolution=0,
            ...     units="level",
            ... )
            >>> plt.imshow(img)

        Note: The field of view remains the same as resolution is varied
        when using :func:`read_bounds`.

        .. figure:: ../images/read_bounds_tissue.png
            :width: 512
            :alt: Diagram illustrating read_bounds

        This is because the bounds are in the baseline (level 0)
        reference frame. Therefore, varying the resolution does not
        change what is visible within the output image.

        If the WSI does not have a resolution layer corresponding
        exactly to the requested resolution (shown above in white with a
        dashed outline), a larger resolution is downscaled to achieve
        the correct requested output resolution.

        If the requested resolution is higher than the baseline (maximum
        resultion of the image), then bicubic interpolation is applied
        to the output image.

        """
        # convert from requested to `baseline`
        bounds_at_baseline = bounds
        if coord_space == "resolution":
            bounds_at_baseline = self.bounds_at_resolution_to_baseline(
                bounds,
                resolution,
                units,
            )
            _, size_at_requested = utils.transforms.bounds2locsize(bounds)
            # * Find parameters for optimal read
            # don't use the `output_size` (`size_at_requested`) here
            # because the rounding error at `bounds_at_baseline` leads to
            # different `size_at_requested` (keeping same read resolution
            # but base image is of different scale)
            _, _, _, post_read_scale = self.find_read_bounds_params(
                bounds_at_baseline,
                resolution=resolution,
                units=units,
            )
        else:
            # * Find parameters for optimal read
            _, _, size_at_requested, post_read_scale = self.find_read_bounds_params(
                bounds_at_baseline,
                resolution=resolution,
                units=units,
            )

        location_at_read, size_at_read = self._find_params_from_baseline(
            *utils.transforms.bounds2locsize(bounds_at_baseline),
        )
        bounds_at_read = utils.transforms.locsize2bounds(location_at_read, size_at_read)

        if interpolation in [None, "none"]:
            interpolation = None

        if interpolation == "optimise" and self.mode == "bool":
            interpolation = "nearest"

        im_region = utils.image.sub_pixel_read(
            self.img,
            bounds_at_read,
            output_size=size_at_requested,
            interpolation=interpolation,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
            read_kwargs=kwargs,
            pad_at_baseline=False,
        )

        if coord_space == "resolution":
            # do this to enforce output size is as defined by input bounds
            im_region = utils.transforms.imresize(
                img=im_region,
                output_size=size_at_requested,
            )
        else:
            im_region = utils.transforms.imresize(
                img=im_region,
                scale_factor=post_read_scale,
                output_size=size_at_requested,
            )

        if self.mode == "rgb":
            if self.post_proc is not None:
                im_region = self.post_proc(im_region)
            return utils.transforms.background_composite(image=im_region, alpha=False)
        return im_region
