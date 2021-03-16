#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "numpy<=1.19.3",
    "pillow",
    "matplotlib",
    "opencv-python>=4.0",
    "openslide-python==1.1.2",
    "pyyaml",
    "pandas",
    "glymur",
    "scikit-learn==0.23.2",
    "scikit-image>=0.17",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="TIA Lab",
    author_email="tialab@dcs.warwick.ac.uk",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Computational pathology toolbox developed by TIA Lab.",
    entry_points={
        "console_scripts": [
            "tiatoolbox=tiatoolbox.cli:main",
        ],
    },
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="tiatoolbox",
    name="tiatoolbox",
    packages=find_packages(include=["tiatoolbox", "tiatoolbox.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/tialab/tiatoolbox",
    version="0.5.2",
    zip_safe=False,
)
