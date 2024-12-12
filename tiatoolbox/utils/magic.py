"""Detection of file type via magic numbers / signatures.

Checks here are based on the file signature, not the file extension.
They are all intented to be fast and lightweight, and should not require
parsing the entire file. There may occationally be false positives which
should be caught when attemping to parse the file.

"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from typing import BinaryIO


def _normalize_binaryio(
    file: str | Path | bytes | BinaryIO | BytesIO,
    *,
    must_exist: bool,
) -> BinaryIO:
    """Normalize the input to a BinaryIO object.

    To be used in a context manager so that the io is closed after use.

    Args:
        file (str or Path or bytes or BinaryIO):
            The file to normalize.
        must_exist (bool, optional):
            Whether the file must exist. Defaults to False.

    Returns:
        BinaryIO
            The file as a BinaryIO object.

    """
    if isinstance(file, (str, Path)):
        path = Path(file)
        if not path.exists():
            if must_exist:
                msg = f"File {path} does not exist."
                raise FileNotFoundError(msg)
            return BytesIO()
        return Path.open(path, mode="rb")  # -- intentional
    if isinstance(file, (BinaryIO, BytesIO)):
        return file
    if isinstance(file, bytes):
        return BytesIO(file)
    msg = (
        f"Input must be a str, Path, bytes, or BinaryIO. "
        f"Received {type(file).__name__}."
    )
    raise TypeError(
        msg,
    )


def is_dir(file: str | Path | bytes | BinaryIO | BytesIO) -> bool:
    """Check if file is a directory.

    Thin wrapper around `Path.is_dir()` to handle multiple input types.

    Args:
        file (Union[str, Path, bytes]):
            The file to check.

    Returns:
        bool:
            A boolean indicating whether file is a directory.

    """
    return Path(file).is_dir() if isinstance(file, (str, Path)) else False


def is_sqlite3(file: str | Path | bytes | BinaryIO | BytesIO) -> bool:
    """Check if a file is a SQLite database.

    Args:
        file (Union[str, Path, bytes]):
            The file to check.

    Returns:
        bool:
            A boolean indicating whether file is a SQLite database.

    """
    if is_dir(file):
        return False
    with _normalize_binaryio(file, must_exist=False) as io:
        return io.read(16) == b"SQLite format 3\x00"


def is_zip(file: str | Path | bytes | BytesIO) -> bool:
    """Check if a file is a ZIP archive.

    Args:
        file (Union[str, Path, bytes]):
            The file to check.

    """
    if is_dir(file):
        return False
    with _normalize_binaryio(file, must_exist=False) as io:
        return zipfile.is_zipfile(io)


def is_dcm(file: str | Path | bytes | BytesIO) -> bool:
    """Determines whether the given file is a DICOM file.

    Checks if the first 128 bytes of the file contain the 'DICM'
    preamble. Returns True if it is a DCM file, False otherwise.

    This intentionally does not parse the file with `pydicom.dcmread()`
    to avoid the overhead of parsing the entire file. Parsing .dcm files
    can be slow for VL Whole Slide Images in some cases (e.g. sparse
    tiling).

    Args:
        file (Union[str, Path, bytes, BytesIO]):
            A string, Path, bytes, or BytesIO object representing the
            path or binary data of the file.

    Returns:
        bool:
            A boolean indicating whether the file is a .dcm file or not.

    """
    if is_dir(file):
        return False
    with _normalize_binaryio(file, must_exist=False) as io:
        io.seek(128)  # Preamble should be ignored for security reasons
        return io.read(4) == b"DICM"
