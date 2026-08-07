"""Helper functions to detect WSI type."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Unpack
from urllib.parse import urlparse

import fsspec
import tifffile
import zarr
from packaging.version import Version
from wsidicom import WsiDicom
from wsidicom.errors import WsiDicomNotFoundError

from tiatoolbox import logger

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from tiatoolbox.wsicore.metadata.ngff import Multiscales

    from .types import WSIReaderParams

MIN_NGFF_VERSION = Version("0.4")
MAX_NGFF_VERSION = Version("0.4")


def is_dicom(path: Path) -> bool:
    """Check if the input is a DICOM file.

    Args:
        path (Path): Path to the file to check.

    Returns:
        bool: True if the file is a DICOM file.

    """
    try:
        WsiDicom.open(path)
        return True
    except WsiDicomNotFoundError:
        return False
    else:
        return False


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


def is_zarr(path: str | Path, **kwargs: Unpack[WSIReaderParams]) -> bool:
    """Check if the input is a Zarr file.

    Args:
        path (Path):
            Path to the file to check.

    Returns:
        bool:
            True if the file is a Zarr file.

    """
    try:
        zarr_kwargs = {k: v for k, v in kwargs.items() if k == "storage_options"}
        _ = zarr.open(path, mode="r", **zarr_kwargs)
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
        min_version (Version):
            Minimum version of the NGFF file to be considered valid.
        max_version (Version):
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

    return is_zarr(path, **zarr_kwargs)


def is_url(path_or_url: str | Path) -> bool:
    """Returns True if input is a URL else False."""
    parsed = urlparse(str(path_or_url))
    return parsed.scheme in {"s3", "http", "https", "ftp", "file"}


def is_valid_zarr_fsspec(file_path: str | Path | np.ndarray) -> bool:
    """Check if the input path is a valid Zarr fsspec JSON file.

    Checks if the file_path is a valid Zarr fsspec JSON file generated by:
    tiatoolbox/utils/tiff_to_fsspec.py

    Args:
        file_path: str Path to the file to check.

    Returns:
        bool: True if the file is a valid Zarr fsspec JSON file

    """
    if not isinstance(file_path, (str, Path)):
        return False

    path = Path(file_path)

    if path.suffix.lower() != ".json":
        return False

    try:
        with fsspec.open(str(file_path), "r") as file:
            data = json.load(file)

        # Basic validation for fsspec Zarr JSON structure
        if ".zattrs" not in data:
            logger.error("Field .zattrs missing in '%s'.", file_path)
            return False

        return True  # noqa: TRY300

    except json.JSONDecodeError as e:
        logger.error("Invalid JSON file: %s", e)
        return False
    except (OSError, ValueError) as e:
        logger.error("An error occurred: %s", e)
        return False
