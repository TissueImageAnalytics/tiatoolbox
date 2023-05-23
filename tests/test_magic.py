"""Tests for detecting magic numbers and signatures in files."""
import sqlite3

from tiatoolbox.utils.magic import is_sqlite3


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
