"""Test for ensuring that requirements files are valid and consistent."""
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import yaml
from pkg_resources import Requirement

REQUIREMENTS_FILES = [
    ("requirements.txt", "requirements_dev.txt"),
    ("requirements.conda.yml", "requirements.dev.conda.yml"),
    ("requirements.win64.conda.yml", None),
]


def parse_pip(
    file_path: Path = None, lines: List[str] = None
) -> Dict[str, Requirement]:
    """Parse a pip requirements file.

    Not that package names are case insensitive and underscores and
    dashes are considered equivalent.

    Args:
        file_path (pathlib.Path):
            Path to the requirements file.
        lines (list):
            List of lines to parse.

    Returns:
        dict:
            A dictionary mapping package names to Requirement objects.
    """
    if lines and file_path:
        raise ValueError("Only one of file_path or lines may be specified")
    if not lines:
        with file_path.open("r") as fh:
            lines = fh.readlines()
    packages = {}
    for line in lines:
        line = line.strip()
        # Skip comment lines
        if line.startswith("#"):
            continue
        # Skip blank lines
        if not line:
            continue
        # Parse requirement
        requirement = Requirement.parse(line)
        # Check for duplicate packages
        if requirement.key in packages:
            raise ValueError(
                f"Duplicate dependency: {requirement.name} in {file_path.name}"
            )
        packages[requirement.key] = requirement
    return packages


def parse_conda(file_path: Path) -> Dict[str, Requirement]:
    """Parse a conda requirements file.

    Args:
        file_path (pathlib.Path):
            Path to the requirements file.

    Returns:
        dict:
            A dictionary mapping package names to
            pkg_resources.Requirement.
    """
    with file_path.open("r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    dependencies: List[str] = config["dependencies"]
    packages = {}
    for dependency in dependencies:
        # pip-style dependency
        if isinstance(dependency, dict):
            pip = parse_pip(lines=dependency["pip"])
            for package_name, requirement in pip.items():
                packages[package_name] = requirement
            continue
        requirement = Requirement.parse(dependency)
        # Check for duplicate packages
        if requirement.key in packages:
            raise ValueError(
                f"Duplicate dependency: {requirement.key} in {file_path.name}"
            )
        packages[requirement.key] = requirement
    return packages


@pytest.fixture(scope="session", autouse=True)
def root_dir(request) -> Path:
    """Return the root directory of the project."""
    return Path(str(request.config.rootdir))


def test_files_exist(root_dir):
    """Test that all requirements files exist."""
    for main, dev in REQUIREMENTS_FILES:
        main_path = root_dir / main
        assert main_path.exists()
        if dev:
            dev_path = root_dir / dev
            assert dev_path.exists()


def parse_requirements(
    file_path: Path = None, lines: List[str] = None
) -> Dict[str, Tuple[str, Tuple[str, ...], str]]:
    """Parse a requirements file (pip or conda).

    Args:
        file_path (pathlib.Path):
            Path to the requirements file.
        lines (list):
            List of lines to parse.

    Returns:
        dict:
            A dictionary mapping package names to
            pkg_resources.Requirement.
    """
    if lines and file_path:
        raise ValueError("Only one of file_path or lines may be specified")
    if file_path.suffix == ".yml":
        return parse_conda(file_path)
    if file_path.suffix == ".txt":
        return parse_pip(file_path, lines)

    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def test_requirements_consistent(root_dir):
    """Test that dev requirements are consistent.

    1. A dev file should contain all the same requirements as the main
       file.
    2. Versions in dev file should be match those in the main file.
    3. A package in all files should have the same constraint and version.

    """
    # Keep track of all parsed files
    all_requirements: Dict[Path, Dict[str, Requirement]] = {}

    # Check that main/dev pairs match
    for main_name, dev_name in REQUIREMENTS_FILES:
        # Get the main requirements
        main_path = root_dir / main_name
        main = parse_requirements(main_path)
        all_requirements[main_path] = main

        # Skip comparison if there is no dev file
        if not dev_name:
            continue

        # Get the dev requirements
        dev_path = root_dir / dev_name
        dev = parse_requirements(dev_path)
        all_requirements[dev_path] = dev

        for name, requirement in main.items():
            # Check that the package is in the dev file
            assert name in dev, f"{name} not in dev requirements ({dev_path.name})"
            main_specs = requirement.specs or [("", "")]
            dev_specs = dev[name].specs or [("", "")]
            main_constraint, main_version = main_specs[0]
            dev_constraint, dev_version = dev_specs[0]

            # Check that the constraint is the same
            assert main_constraint == dev_constraint, (
                f"{name} has different constraints:"
                f" {main_constraint}{main_version} ({main_path.name})"
                f" vs {dev_constraint}{dev_version} ({dev_path.name})."
            )

            # Check that the version spec is the same
            assert main_version == dev_version, (
                f"{name} has different versions:"
                f" {main_version} ({main_path.name})"
                f" vs {dev_version} ({dev_path.name})."
            )

    # Check that packages which are in all files match
    key_sets = [set(x.keys()) for x in all_requirements.values()]
    common_packages = key_sets[0].intersection(*key_sets[1:])
    file_names = [path.name for path in all_requirements]
    for package in common_packages:
        specs = [
            x[package].specs[0] if x[package].specs else ("?", "None")
            for x in all_requirements.values()
        ]
        constraints, versions = zip(*specs)
        formatted_deps = [
            f"{c}{v} ({n})" for n, c, v in zip(file_names, constraints, versions)
        ]
        assert all(x == constraints[0] for x in constraints), (
            f"{package} has inconsistent constraints:" f" {', '.join(formatted_deps)}."
        )
        assert all(x == versions[0] for x in versions), (
            f"{package} has inconsistent versions:" f" {', '.join(formatted_deps)}."
        )
