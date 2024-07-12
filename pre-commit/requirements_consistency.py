"""Test for ensuring that requirements files are valid and consistent."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import NoReturn

import yaml
from pkg_resources import Requirement

REQUIREMENTS_FILES = [
    ("requirements/requirements.txt", "requirements/requirements_dev.txt"),
    ("requirements/requirements.conda.yml", "requirements/requirements.dev.conda.yml"),
    ("requirements/requirements.win64.conda.yml", None),
    ("docs/requirements.txt", None),
    ("setup.py", None),
]


def parse_pip(
    file_path: Path | None = None,
    lines: list[str] | None = None,
) -> dict[str, Requirement]:
    """Parse a pip requirements file.

    Note that package names are case insensitive and underscores and
    dashes are considered equivalent.

    Args:
        file_path (Path):
            Path to the requirements file.
        lines (list):
            List of lines to parse.

    Returns:
        dict:
            A dictionary mapping package names to
            :class:`pkg_resources.Requirement`.

    """
    if lines and file_path:
        msg = "Only one of file_path or lines may be specified"
        raise ValueError(msg)
    if not lines:
        with file_path.open("r") as fh:
            lines = fh.readlines()
    packages = {}
    for line in lines:
        line_ = line.strip()
        # Skip comment lines
        if line_.startswith("#"):
            continue
        # Skip blank lines
        if not line_:
            continue
        # Parse requirement
        requirement = Requirement.parse(line_)
        # Check for duplicate packages
        if requirement.key in packages:
            msg = f"Duplicate dependency: {requirement.name} in {file_path.name}"
            raise ValueError(
                msg,
            )
        packages[requirement.key] = requirement
    return packages


def parse_conda(file_path: Path) -> dict[str, Requirement]:
    """Parse a conda requirements file.

    Args:
        file_path (Path):
            Path to the requirements file.

    Returns:
        dict:
            A dictionary mapping package names to
            :class:`pkg_resources.Requirement`.

    """
    with file_path.open("r") as fh:
        config = yaml.safe_load(fh)
    dependencies: list[str] = config["dependencies"]
    packages = {}
    for dependency in dependencies:
        # pip-style dependency
        if isinstance(dependency, dict):
            pip = parse_pip(lines=dependency["pip"])
            packages = dict(pip.items())

            continue
        requirement = Requirement.parse(dependency)
        # Check for duplicate packages
        if requirement.key in packages:
            msg = f"Duplicate dependency: {requirement.key} in {file_path.name}"
            raise ValueError(
                msg,
            )
        packages[requirement.key] = requirement
    return packages


def parse_setup_py(file_path: Path) -> dict[str, Requirement]:
    """Parse a setup.py file.

    Args:
        file_path (Path):
            Path to the setup.py file.

    Returns:
        dict:
            A dictionary mapping package names to
            pkg_resources.Requirement.
    """
    mock_setup = {}
    import setuptools

    setuptools.setup = lambda **kw: mock_setup.update(kw)
    spec = importlib.util.spec_from_file_location("setup", str(file_path))
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)

    del setup, setuptools  # skipcq

    requirements = mock_setup.get("install_requires", [])
    return parse_pip(lines=requirements)


def test_files_exist(root_dir: Path) -> None:
    """Test that all requirements files exist.

    Args:
        root_dir (Path):
            Path to the root directory of the project.

    Raises:
        FileNotFoundError:
            If a requirements file is missing.

    """
    for sub_name, super_name in REQUIREMENTS_FILES:
        sub_path = root_dir / sub_name
        if not sub_path.exists():
            msg = f"Missing file: {sub_path}"
            raise FileNotFoundError(msg)
        if super_name:
            super_path = root_dir / super_name
            if not super_path.exists():
                msg = f"Missing file: {super_path}"
                raise FileNotFoundError(msg)


def parse_requirements(
    file_path: Path | None = None,
    lines: list[str] | None = None,
) -> dict[str, tuple[str, tuple[str, ...], str]]:
    """Parse a requirements file (pip or conda).

    Args:
        file_path (Path):
            Path to the requirements file.
        lines (list):
            List of lines to parse.

    Returns:
        dict:
            A dictionary mapping package names to
            pkg_resources.Requirement.
    """
    if lines and file_path:
        msg = "Only one of file_path or lines may be specified"
        raise ValueError(msg)
    if file_path.name == "setup.py":
        return parse_setup_py(file_path)
    if file_path.suffix == ".yml":
        return parse_conda(file_path)
    if file_path.suffix == ".txt":
        return parse_pip(file_path, lines)

    msg = f"Unsupported file type: {file_path.suffix}"
    raise ValueError(msg)


def in_common_consistent(all_requirements: dict[Path, dict[str, Requirement]]) -> bool:
    """Test that in-common requirements are consistent.

    Args:
        all_requirements (dict):
            Dictionary mapping requirements files to
            dictionaries mapping package names to
            pkg_resources.Requirement.

    Returns:
        bool:
            True if the requirements are consistent.

    """
    consistent = True
    # Check that requirements are consistent across files
    # First find a set of all requirement keys
    requirement_key_sets = [set(x.keys()) for x in all_requirements.values()]
    requirement_keys = requirement_key_sets[0].union(*requirement_key_sets[1:])

    # Iterate over the keys
    for key in requirement_keys:
        # Find the specs for the requirement and which files it is in
        zipped_file_specs = [
            (
                path,
                *(  # Unpack the (constraint, version) tuple
                    requirements[key].specs[0]  # Get the first spec
                    if requirements[key].specs  # Check that there are specs
                    else ("", "None")  # Default if no specs
                ),
            )
            for path, requirements in all_requirements.items()
            if key in requirements  # Filter out files that don't have the key
        ]

        # Unzip the specs to get a list of constraints and versions
        _, constraints, versions = zip(*zipped_file_specs)

        # Check that the constraints and versions are the same across files
        formatted_reqs = [f"{c}{v} ({p.name})" for p, c, v in zipped_file_specs]
        if any(x != constraints[0] for x in constraints):
            print(
                f"{key} has inconsistent constraints: {', '.join(formatted_reqs)}.",
            )
            consistent = False
        if any(x != versions[0] for x in versions):
            print(f"{key} has inconsistent versions: {', '.join(formatted_reqs)}.")
            consistent = False
    return consistent


def main() -> NoReturn:
    """Main entry point for the hook."""
    root = Path(__file__).parent.parent
    test_files_exist(root)

    passed = True

    # Keep track of all parsed files
    all_requirements: dict[Path, dict[str, Requirement]] = {}

    # Check that packages in main are also in super (dev)
    for sub_name, super_name in REQUIREMENTS_FILES:
        # Get the main requirements
        sub_path = root / sub_name
        sub_reqs = parse_requirements(sub_path)
        all_requirements[sub_path] = sub_reqs

        # Skip comparison if there is no superset (dev) file
        if not super_name:
            continue

        # Get the superset of (dev) requirements
        super_path = root / super_name
        super_reqs = parse_requirements(super_path)
        all_requirements[super_path] = super_reqs

        # Check that all sub requirements are in the super (dev) file
        sub_keys = set(sub_reqs.keys())
        super_keys = set(super_reqs.keys())
        super_missing = sub_keys - super_keys
        if super_missing:  # sub is not a subset of super
            print(f"{super_name} is missing {', '.join(super_missing)} from {sub_name}")
            passed = False

    passed &= in_common_consistent(all_requirements)

    if not passed:
        sys.exit(1)
    print("All tests passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
