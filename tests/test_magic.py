"""Test for detecting magic numbers and signatures in files."""

import sqlite3
import zipfile
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Callable

import pytest

from tiatoolbox.utils.magic import _normalize_binaryio, is_dcm, is_sqlite3, is_zip


def test_is_sqlite3(tmp_path: Path) -> None:
    """Create a dummy sqlite database and use tiatoolbox.magic.is_sqlite3()."""
    db = sqlite3.connect(tmp_path / "test.db")
    db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT);")
    db.execute("INSERT INTO test (name) VALUES ('test');")
    db.commit()
    db.close()

    (tmp_path / "test.txt").write_text("test")

    assert is_sqlite3(tmp_path / "test.db")
    assert not is_sqlite3(tmp_path / "test.txt")


def test_is_dcm(remote_sample: Callable) -> None:
    """Get a dummy DICOM file and use tiatoolbox.magic.is_dicom()."""
    path = remote_sample("dicom-1")
    for subpath in path.glob("**/*.dcm"):
        assert is_dcm(subpath)


def test_is_zip(tmp_path: Path) -> None:
    """Create a dummy zip file and use tiatoolbox.magic.is_zip()."""
    (tmp_path / "test.txt").write_text("test")
    with zipfile.ZipFile(tmp_path / "test.zip", "w") as zf:
        zf.write(tmp_path / "test.txt")

    assert is_zip(tmp_path / "test.zip")
    assert not is_zip(tmp_path / "test.txt")


def test_normalize_must_exist() -> None:
    """Test that must_exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _normalize_binaryio("nonexistent", must_exist=True)


def test_normalize_bytes() -> None:
    """Test that _normalize_binaryio() returns BytesIO for bytes."""
    assert isinstance(
        _normalize_binaryio(b"test", must_exist=False),
        (BytesIO, BinaryIO),
    )


def test_normalize_binaryio() -> None:
    """Test that _normalize_binaryio() returns BinaryIO for BinaryIO."""
    assert isinstance(
        _normalize_binaryio(BytesIO(b"test"), must_exist=False),
        (BytesIO, BinaryIO),
    )


def test_normalize_type_error() -> None:
    """Test that _normalize_binaryio() raises TypeError for invalid types."""
    with pytest.raises(TypeError):
        _normalize_binaryio(1, must_exist=False)


def test_normalize_non_existent(tmp_path: Path) -> None:
    """Test that _normalize_binaryio() returns empty BinaryIO for non-existant file."""
    assert isinstance(
        _normalize_binaryio(tmp_path / "foo", must_exist=False),
        (BytesIO, BinaryIO),
    )


def test_is_sqlite3_dir(tmp_path: Path) -> None:
    """Test that is_sqlite3() returns False for directories."""
    assert not is_sqlite3(tmp_path)


def test_is_dcm_dir(tmp_path: Path) -> None:
    """Test that is_dcm() returns False for directories."""
    assert not is_dcm(tmp_path)


def test_is_zip_dir(tmp_path: Path) -> None:
    """Test that is_zip() returns False for directories."""
    assert not is_zip(tmp_path)
