"""Tests verifying the uv sync development setup is consistent and functional.

These tests guard against drift between pyproject.toml [project] dependencies
and the requirements/*.txt files, and confirm that a uv-synced environment
can actually import and exercise key tiatoolbox functionality.
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
REQUIREMENTS_TXT = REPO_ROOT / "requirements" / "requirements.txt"
REQUIREMENTS_DEV_TXT = REPO_ROOT / "requirements" / "requirements_dev.txt"
REQUIREMENTS_DOCS_TXT = REPO_ROOT / "docs" / "requirements.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(name: str) -> str:
    """Normalize a package name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name.lower())


def _parse_req_line(line: str) -> tuple[str, str] | None:
    """Parse one requirements.txt line into (normalized_name, specifier).

    Returns None for blank lines, comments, and -r/-i includes.
    """
    line = line.split("#")[0].strip()
    if not line or line.startswith("-"):
        return None
    # Match name (with optional extras) then version specifier
    m = re.match(r"^([A-Za-z0-9_.-]+)(\[.*?])?([\s,><=!~^*]+.*)?$", line)
    if not m:
        return None
    name = _normalize(m.group(1))
    spec = re.sub(r"\s+", "", m.group(3) or "")
    return name, spec


def _parse_requirements_file(path: Path) -> dict[str, str]:
    """Return {normalized_name: specifier} from a requirements file."""
    result = {}
    for line in path.read_text().splitlines():
        parsed = _parse_req_line(line)
        if parsed:
            result[parsed[0]] = parsed[1]
    return result


def _parse_pyproject_deps(deps: list[str]) -> dict[str, str]:
    """Return {normalized_name: specifier} from a pyproject.toml dep list.

    Skips self-referential entries like ``tiatoolbox[docs]``.
    """
    result = {}
    for dep in deps:
        dep = dep.strip()
        if _normalize(dep.split("[")[0]) == "tiatoolbox":
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)(\[.*?])?(.*)?$", dep)
        if m:
            name = _normalize(m.group(1))
            spec = re.sub(r"\s+", "", m.group(3) or "")
            result[name] = spec
    return result


# ---------------------------------------------------------------------------
# Dependency parity: pyproject.toml ↔ requirements/*.txt
# ---------------------------------------------------------------------------


class TestDependencyParity:
    """Guard against drift between pyproject.toml and requirements files."""

    @pytest.fixture(scope="class")
    def pyproject(self) -> dict:
        """Load pyproject.toml once for the class."""
        return tomllib.loads(PYPROJECT.read_text())

    def test_runtime_deps_match_requirements_txt(self, pyproject: dict) -> None:
        """Every package in requirements.txt must appear in [project].dependencies."""
        req_deps = _parse_requirements_file(REQUIREMENTS_TXT)
        proj_deps = _parse_pyproject_deps(pyproject["project"]["dependencies"])

        missing = set(req_deps) - set(proj_deps)
        assert not missing, (
            f"Packages in requirements.txt but missing from pyproject.toml "
            f"[project].dependencies: {sorted(missing)}\n"
            f"Run `uv sync` users will not get these packages."
        )

        extra = set(proj_deps) - set(req_deps)
        assert not extra, (
            f"Packages in pyproject.toml [project].dependencies but missing from "
            f"requirements.txt: {sorted(extra)}\n"
            f"Conda/pip users installing from requirements.txt will not get these."
        )

    def test_runtime_dep_version_specs_match(self, pyproject: dict) -> None:
        """Version specifiers must be identical in both sources (after normalization)."""
        req_deps = _parse_requirements_file(REQUIREMENTS_TXT)
        proj_deps = _parse_pyproject_deps(pyproject["project"]["dependencies"])

        mismatches = {
            name: (req_deps[name], proj_deps[name])
            for name in req_deps
            if name in proj_deps and req_deps[name] != proj_deps[name]
        }
        assert not mismatches, (
            "Version specifier mismatches between requirements.txt and "
            "pyproject.toml [project].dependencies:\n"
            + "\n".join(
                f"  {name}: requirements.txt={r!r}  pyproject.toml={p!r}"
                for name, (r, p) in sorted(mismatches.items())
            )
        )

    def test_dev_deps_match_requirements_dev_txt(self, pyproject: dict) -> None:
        """Dev packages must match between pyproject.toml [dev] and requirements_dev.txt.

        requirements_dev.txt uses ``-r requirements.txt`` and
        ``-r ../docs/requirements.txt`` for transitive deps; those are excluded
        from comparison (covered by tiatoolbox[docs] self-reference in pyproject).
        """
        req_deps = _parse_requirements_file(REQUIREMENTS_DEV_TXT)
        proj_deps = _parse_pyproject_deps(
            pyproject["project"]["optional-dependencies"]["dev"]
        )

        missing = set(req_deps) - set(proj_deps)
        assert not missing, (
            f"Packages in requirements_dev.txt but missing from pyproject.toml "
            f"[project.optional-dependencies.dev]: {sorted(missing)}"
        )

        extra = set(proj_deps) - set(req_deps)
        assert not extra, (
            f"Packages in pyproject.toml [dev] but missing from "
            f"requirements_dev.txt: {sorted(extra)}"
        )

    def test_docs_deps_match_requirements_docs_txt(self, pyproject: dict) -> None:
        """Docs packages must match between pyproject.toml [docs] and docs/requirements.txt."""
        req_deps = _parse_requirements_file(REQUIREMENTS_DOCS_TXT)
        proj_deps = _parse_pyproject_deps(
            pyproject["project"]["optional-dependencies"]["docs"]
        )

        missing = set(req_deps) - set(proj_deps)
        assert not missing, (
            f"Packages in docs/requirements.txt but missing from pyproject.toml "
            f"[project.optional-dependencies.docs]: {sorted(missing)}"
        )

        extra = set(proj_deps) - set(req_deps)
        assert not extra, (
            f"Packages in pyproject.toml [docs] but missing from "
            f"docs/requirements.txt: {sorted(extra)}"
        )


# ---------------------------------------------------------------------------
# Version consistency
# ---------------------------------------------------------------------------


class TestVersionConsistency:
    """Version must be identical across all authoritative locations."""

    @pytest.fixture(scope="class")
    def pyproject_version(self) -> str:
        """Return the version declared in pyproject.toml [project]."""
        return tomllib.loads(PYPROJECT.read_text())["project"]["version"]

    def test_pyproject_version_matches_package(self, pyproject_version: str) -> None:
        """Installed package version must match pyproject.toml [project].version."""
        installed = version("tiatoolbox")
        assert installed == pyproject_version, (
            f"Installed tiatoolbox version ({installed!r}) != "
            f"pyproject.toml version ({pyproject_version!r}). "
            f"Re-install with `uv sync` or `pip install -e .`."
        )

    def test_pyproject_version_matches_init(self, pyproject_version: str) -> None:
        """tiatoolbox.__version__ must match pyproject.toml [project].version."""
        import tiatoolbox  # noqa: PLC0415

        assert tiatoolbox.__version__ == pyproject_version, (
            f"tiatoolbox.__version__ ({tiatoolbox.__version__!r}) != "
            f"pyproject.toml version ({pyproject_version!r}). "
            f"Run `poetry version` to bump both simultaneously."
        )

    def test_pyproject_version_matches_setup_py(self, pyproject_version: str) -> None:
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


def test_python_version_file_exists() -> None:
    """.python-version must exist and declare Python 3.x."""
    pv_file = REPO_ROOT / ".python-version"
    assert pv_file.exists(), ".python-version not found — run `echo '3.11' > .python-version`"
    content = pv_file.read_text().strip()
    assert re.match(r"^3\.\d+", content), (
        f".python-version contains {content!r}, expected e.g. '3.11'"
    )


def test_uv_lockfile_exists() -> None:
    """uv.lock must be committed so `uv sync --frozen` works in CI."""
    assert (REPO_ROOT / "uv.lock").exists(), (
        "uv.lock not found. Generate it with `uv lock` and commit the result."
    )


def test_pyproject_has_project_table() -> None:
    """pyproject.toml must contain a [project] table for uv sync to work."""
    data = tomllib.loads(PYPROJECT.read_text())
    assert "project" in data, "[project] table missing from pyproject.toml"
    for field in ("name", "version", "dependencies", "requires-python"):
        assert field in data["project"], f"[project].{field} missing from pyproject.toml"


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
        [sys.executable, "-c", f"import {module}"],  # noqa: S603
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Failed to import {module!r}.\n"
        f"stderr: {result.stderr.strip()}\n"
        f"This module may be missing from the uv-synced environment."
    )


def test_cli_entry_point() -> None:
    """The `tiatoolbox` CLI entry point must respond to --help."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "tiatoolbox", "--help"],  # noqa: S603
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"`python -m tiatoolbox --help` failed.\nstderr: {result.stderr.strip()}"
    )
    assert "tiatoolbox" in result.stdout.lower(), (
        "Unexpected --help output from tiatoolbox CLI"
    )
