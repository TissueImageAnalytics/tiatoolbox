"""Test that notbooks execute without error."""
import json
import re
import subprocess
from pathlib import Path

import pytest


@pytest.fixture()
def root_path(request):
    """Get the root path of the project."""
    return Path(request.config.rootdir)


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
        (
            "Patch Extraction",
            {"notebook": "examples/04-patch-extraction.ipynb"},
        ),
    ]

    @staticmethod
    def test_notebook(root_path, tmp_path, notebook):
        """Test that the notebook execute without exception."""
        nb_path = root_path / notebook
        tmp_nb_path = tmp_path / nb_path.name

        # Load the notebook JSON
        with nb_path.open() as fh:
            nb_json = json.load(fh)

        # Filter out lines which start with "!apt-get" or "!pip" installs
        for cell in nb_json["cells"]:
            if cell.get("cell_type") != "code":
                continue
            cell["source"] = [
                line
                for line in cell["source"]
                if not re.match(r"^!\s*(apt(-get)?|pip).*install", line)
            ]

        # Write the notebook JSON to a temporary file
        with tmp_nb_path.open("w") as fh:
            json.dump(nb_json, fh)

        process = subprocess.Popen(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--stdout",
                str(tmp_nb_path),
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
