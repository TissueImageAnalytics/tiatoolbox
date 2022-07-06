"""Test for ensuring that requirements files are valid and consistent."""
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from pkg_resources import Requirement

REQUIREMENTS_FILES = [
    ("requirements.txt", "requirements_dev.txt"),
    ("requirements.conda.yml", "requirements.dev.conda.yml"),
    ("requirements.win64.conda.yml", None),
    ("docs/requirements.txt", None),
    ("setup.py", None),
]


def parse_pip(
    file_path: Path = None, lines: List[str] = None
) -> Dict[str, Requirement]:
    """Parse a pip requirements file.

    Note that package names are case insensitive and underscores and
    dashes are considered equivalent.

    Args:
        file_path (pathlib.Path):
            Path to the requirements file.
        lines (list):
            List of lines to parse.

    Returns:
        dict:
            A dictionary mapping package names to
            :class:`pkg_resources.Requirement`.

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
            :class:`pkg_resources.Requirement`.

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


def test_files_exist(root_dir):
    """Test that all requirements files exist."""
    for main, dev in REQUIREMENTS_FILES:
        main_path = root_dir / main
        if not main_path.exists():
            raise FileNotFoundError(f"Missing file: {main_path}")
        assert main_path.exists()
        if dev:
            dev_path = root_dir / dev
            if not dev_path.exists():
                raise FileNotFoundError(f"Missing file: {dev_path}")


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
    if file_path.name == "setup.py":
        mock_setup = {}
        import setuptools

        setuptools.setup = lambda **kw: mock_setup.update(kw)
        spec = importlib.util.spec_from_file_location("setup", str(file_path))
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)

        del setup, setuptools

        requirements = mock_setup.get("install_requires", [])
        return parse_pip(lines=requirements)
    if file_path.suffix == ".yml":
        return parse_conda(file_path)
    if file_path.suffix == ".txt":
        return parse_pip(file_path, lines)

    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def are_requirements_consistent(root_dir: Path) -> bool:
    """Test that dev requirements are consistent.

    1. A dev file should contain all the same requirements as the main
       file.
    2. A package in any two or more files should have the same
       constraint and version.

    """
    consistent = True

    # Keep track of all parsed files
    all_requirements: Dict[Path, Dict[str, Requirement]] = {}

    # Check that packages in main are also in dev
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

        # Check that all main packages are in the dev file
        for name in main:
            assert name in dev, f"{name} not in dev requirements ({dev_path.name})"

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
        if not all(x == constraints[0] for x in constraints):
            print(
                f"{key} has inconsistent constraints:" f" {', '.join(formatted_reqs)}."
            )
            consistent = False
        if not all(x == versions[0] for x in versions):
            print(f"{key} has inconsistent versions:" f" {', '.join(formatted_reqs)}.")
            consistent = False
    return consistent  # noqa: R504


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    test_files_exist(root)
    consistent = are_requirements_consistent(root)
    if not consistent:
        sys.exit(1)
    print("All tests passed")
    sys.exit(0)
