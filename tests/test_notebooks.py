"""Test that notbooks execute without error."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

NOTEBOOKS = (
    "examples/01-wsi-reading.ipynb",
    "examples/02-stain-normalization.ipynb",
)


@pytest.fixture()
def root_path(request):
    return Path(request.config.rootdir)


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_notebook(root_path, notebook):
    """Test that notbooks execute without exception."""
    path = root_path / notebook
    process = subprocess.Popen(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--stdout",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _, err = process.communicate()
    # Fail the test and print the tail of stderr if the process failed
    if process.returncode != 0:
        clean_err = re.sub(
            r"\x1b\[[0-9;]*[mGKHF]", "", err, flags=re.IGNORECASE | re.MULTILINE
        )
        lines = clean_err.split("\n")
        tail = "\n\t".join(lines[-10:])
        pytest.fail(
            f"Notebook failed to execute: {notebook}"
            f" Tail of notebook output:"
            f"\n\t{tail}"
        )

    # Remove the tmp directory created by nbconvert
    tmp = path.parent / "tmp"
    shutil.rmtree(tmp, ignore_errors=True)
