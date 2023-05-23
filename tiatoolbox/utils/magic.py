"""Detection of file type via magic numbers / signatures."""
from pathlib import Path
from typing import Union


def _normalize_file(file: Union[str, Path, bytes], num_bytes: int = 256) -> bytes:
    """Normalize the input to bytes.

    Args:
        file (str or Path or bytes):
            The file to normalize.
        num_bytes : (int, optional)
            The number of bytes to read from the file, by default 256.

    Returns:
        bytes
            The file as bytes.

    """
    if isinstance(file, (str, Path)):
        with open(file, "rb") as f:
            return f.read(num_bytes)
    if isinstance(file, bytes):
        return file[:num_bytes]
    raise TypeError(f"file must be a str, Path or bytes, not {type(file).__name__}")


def is_sqlite3(file: Union[str, Path, bytes]) -> bool:
    """Check if a file is a SQLite database.

    Args:
        file (Union[str, Path, bytes]):
            The file to check.

    """
    header = _normalize_file(file, 16)

    return header == b"SQLite format 3\x00"


def is_zip(file: Union[str, Path, bytes]) -> bool:
    """Check if a file is a ZIP archive.

    Args:
        file (Union[str, Path, bytes]):
            The file to check.

    """
    header = _normalize_file(file, 4)

    return header in {b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"}
