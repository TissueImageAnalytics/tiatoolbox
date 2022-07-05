"""Test that notbooks execute without error."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

NOTEBOOKS = (
    "examples/01-wsi-reading.ipynb",
    "examples/02-stain-normalization.ipynb",
    "examples/03-tissue-masking.ipynb",
)


@pytest.fixture()
def root_path(request):
    """Get the root path of the project."""
    return Path(request.config.rootdir)


# -------------------------------------------------------------------------------------
# Generate Parameterized Tests
# -------------------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    """Generate (parameterize) test scenarios.

    Adapted from pytest documentation. For more information on
    parameterized tests see:
    https://docs.pytest.org/en/6.2.x/example/parametrize.html#a-quick-port-of-testscenarios

    """
    # Return if the test is not part of a class
    if metafunc.cls is None:
        return
    idlist = []
    argvalues = []
    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append([x[1] for x in items])
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


class TestNotebook:
    scenarios = [
        (
            "WSI Reading",
            {"notebook": "examples/01-wsi-reading.ipynb"},
        ),
        (
            "Stain Normalization",
            {"notebook": "examples/02-stain-normalization.ipynb"},
        ),
        (
            "Tissue Masking",
            {"notebook": "examples/03-tissue-masking.ipynb"},
        ),
    ]

    @staticmethod
    def test_notebook(root_path, tmp_path, notebook):
        """Test that the notebook execute without exception."""
        path = root_path / notebook
        process = subprocess.Popen(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--output-dir",
                str(tmp_path),
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
            # Remove control codes (e.g. colors) from the output
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
        tmp = tmp_path / "tmp"
        shutil.rmtree(tmp, ignore_errors=True)
