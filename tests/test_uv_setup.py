"""Tests verifying the uv sync development setup is consistent and functional.

These tests confirm that pyproject.toml is well-formed and that a uv-synced
environment can actually import and exercise key tiatoolbox functionality.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


# ---------------------------------------------------------------------------
# Version consistency
# ---------------------------------------------------------------------------


class TestVersionConsistency:
    """Version must be identical across all authoritative locations."""

    @pytest.fixture(scope="class")
    def pyproject_version(self) -> str:
        """Return the version declared in pyproject.toml [project]."""
        return tomllib.loads(PYPROJECT.read_text())["project"]["version"]

    @staticmethod
    def test_pyproject_version_matches_package(pyproject_version: str) -> None:
        """Installed package version must match pyproject.toml [project].version."""
        installed = version("tiatoolbox")
        assert installed == pyproject_version, (
            f"Installed tiatoolbox version ({installed!r}) != "
            f"pyproject.toml version ({pyproject_version!r}). "
            f"Re-install with `uv sync` or `pip install -e .`."
        )

    @staticmethod
    def test_pyproject_version_matches_init(pyproject_version: str) -> None:
        """tiatoolbox.__version__ must match pyproject.toml [project].version."""
        import tiatoolbox  # noqa: PLC0415

        assert tiatoolbox.__version__ == pyproject_version, (
            f"tiatoolbox.__version__ ({tiatoolbox.__version__!r}) != "
            f"pyproject.toml version ({pyproject_version!r}). "
            f"Run `poetry version` to bump both simultaneously."
        )

    @staticmethod
    def test_pyproject_version_matches_setup_py(pyproject_version: str) -> None:
        """setup.py version= must match pyproject.toml [project].version."""
        setup_py = (REPO_ROOT / "setup.py").read_text()
        m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', setup_py)
        assert m, "Could not find version= in setup.py"
        assert m.group(1) == pyproject_version, (
            f"setup.py version ({m.group(1)!r}) != "
            f"pyproject.toml version ({pyproject_version!r})."
        )


# ---------------------------------------------------------------------------
# uv infrastructure files
# ---------------------------------------------------------------------------


def test_pyproject_has_project_table() -> None:
    """pyproject.toml must contain a [project] table for uv sync to work."""
    data = tomllib.loads(PYPROJECT.read_text())
    assert "project" in data, "[project] table missing from pyproject.toml"
    for field in ("name", "version", "dependencies", "requires-python"):
        assert field in data["project"], (
            f"[project].{field} missing from pyproject.toml"
        )


# ---------------------------------------------------------------------------
# Functional: key imports and CLI entry point
# ---------------------------------------------------------------------------


CRITICAL_IMPORTS = [
    "tiatoolbox",
    "tiatoolbox.wsicore.wsireader",
    "tiatoolbox.models.engine.semantic_segmentor",
    "tiatoolbox.tools.patchextraction",
    "tiatoolbox.annotation.storage",
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "shapely",
    "openslide",
    "skimage",
    "scipy",
]


@pytest.mark.parametrize("module", CRITICAL_IMPORTS)
def test_critical_module_importable(module: str) -> None:
    """Each critical module must be importable in the active environment."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, (
        f"Failed to import {module!r}.\n"
        f"stderr: {result.stderr.strip()}\n"
        f"This module may be missing from the uv-synced environment."
    )


def test_cli_entry_point() -> None:
    """The `tiatoolbox` CLI entry point must respond to --help."""
    result = subprocess.run(
        [sys.executable, "-m", "tiatoolbox", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, (
        f"`python -m tiatoolbox --help` failed.\nstderr: {result.stderr.strip()}"
    )
    assert "tiatoolbox" in result.stdout.lower(), (
        "Unexpected --help output from tiatoolbox CLI"
    )
