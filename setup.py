#!/usr/bin/env python

"""The setup script."""

import sys
from pathlib import Path

from setuptools import find_packages, setup

with Path("README.md").open(encoding="utf-8") as readme_file:
    readme = readme_file.read()

with Path("HISTORY.md").open(encoding="utf-8") as history_file:
    history = history_file.read()

import tomllib

with Path("pyproject.toml").open("rb") as _f:
    _pyproject = tomllib.load(_f)
install_requires = _pyproject["project"]["dependencies"]

# Optional model dependencies, imported lazily by the architecture that needs them.
# DeepSpot-M is packaged separately because it ships its own gene vocabulary assets
# and its weights are gated, so it is not pulled in for every install.
extras_require = {
    "deepspotm": ["deepspotm>=1.0.0"],
}

dependency_links = []

if sys.platform != "darwin":
    dependency_links = ["https://download.pytorch.org/whl/cu126"]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="TIA Centre",
    author_email="TIA@warwick.ac.uk",
    python_requires=">=3.12, <3.15",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    description="Computational pathology toolbox developed by TIA Centre.",
    dependency_links=dependency_links,
    entry_points={
        "console_scripts": [
            "tiatoolbox=tiatoolbox.cli:main",
        ],
    },
    install_requires=install_requires,
    extras_require=extras_require,
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="tiatoolbox",
    name="tiatoolbox",
    packages=find_packages(include=["tiatoolbox", "tiatoolbox.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/TissueImageAnalytics/tiatoolbox",
    version="2.1.3",
    zip_safe=False,
)
