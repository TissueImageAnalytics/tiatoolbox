#!/usr/bin/env python

"""The setup script."""
import sys
from pathlib import Path

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

install_requires = [
    line
    for line in Path("requirements.txt").read_text().splitlines()
    if line and line[0] not in ("-", "#")
]

dependency_links = []

if sys.platform != "darwin":
    dependency_links = ["https://download.pytorch.org/whl/cu113"]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="TIA Centre",
    author_email="tia@dcs.warwick.ac.uk",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="Computational pathology toolbox developed by TIA Centre.",
    dependency_links=dependency_links,
    entry_points={
        "console_scripts": [
            "tiatoolbox=tiatoolbox.cli:main",
        ],
    },
    install_requires=install_requires,
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
    version="1.3.3",
    zip_safe=False,
)
