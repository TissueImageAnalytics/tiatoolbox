"""Tests for detecting magic numbers and signatures in files."""
import sqlite3

import pytest

from tiatoolbox.utils.magic import _normalize_binaryio, is_dcm, is_sqlite3


def test_is_sqlite3(tmp_path):
    """Create a dummy sqlite database and use tiatoolbox.magic.is_sqlite3()."""
    db = sqlite3.connect(tmp_path / "test.db")
    db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT);")
    db.execute("INSERT INTO test (name) VALUES ('test');")
    db.commit()
    db.close()

    (tmp_path / "test.txt").write_text("test")

    assert is_sqlite3(tmp_path / "test.db")
    assert not is_sqlite3(tmp_path / "test.txt")


def test_is_dcm(remote_sample):
    """Get a dummy DICOM file and use tiatoolbox.magic.is_dicom()."""
    path = remote_sample("dicom-1")
    for subpath in path.glob("**/*.dcm"):
        assert is_dcm(subpath)


def test_normalize_must_exist():
    """Test that must_exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _normalize_binaryio("nonexistent", must_exist=True)
