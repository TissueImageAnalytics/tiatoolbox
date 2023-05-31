"""Detection of file type via magic numbers / signatures."""
import zipfile
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Union


def _normalize_binaryio(file: Union[str, Path, bytes, BinaryIO]) -> BinaryIO:
    """Normalize the input to a BinaryIO object.

    To be used in a context manager so that the io is closed after use.

    Args:
        file (str or Path or bytes or BinaryIO):
            The file to normalize.

    Returns:
        BinaryIO
            The file as a BinaryIO object.

    """
    if isinstance(file, (str, Path)):
        return open(file, "rb")  # noqa: SIM115 -- intentional
    if isinstance(file, BinaryIO):
        return file
    if isinstance(file, bytes):
        return BytesIO(file)
    raise TypeError(
        "Input must be a str, Path, bytes, or BinaryIO. "
        f"Recieved {type(file).__name__}."
    )


def is_dir(file: Union[str, Path, bytes, BinaryIO]) -> bool:
    """Check if file is a directory.

    Thin wrapper around `pathlib.Path.is_dir()` to handle multiple input types.

    Args:
        file (Union[str, Path, bytes]):
            The file to check.

    """
    return Path(file).is_dir() if isinstance(file, (str, Path)) else False


def is_sqlite3(file: Union[str, Path, bytes, BinaryIO]) -> bool:
    """Check if a file is a SQLite database.

    Args:
        file (Union[str, Path, bytes]):
            The file to check.

    """
    if is_dir(file):
        return False
    with _normalize_binaryio(file) as io:
        return io.read(16) == b"SQLite format 3\x00"


def is_zip(file: Union[str, Path, bytes, BytesIO]) -> bool:
    """Check if a file is a ZIP archive.

    Args:
        file (Union[str, Path, bytes]):
            The file to check.

    """
    if is_dir(file):
        return False
    with _normalize_binaryio(file) as io:
        return zipfile.is_zipfile(io)
