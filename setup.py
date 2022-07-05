#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = [
    "albumentations>=1.0.3",
    "Click>=8.1.2",
    "defusedxml>=0.7.1",
    "glymur>=0.9.9",
    "imagecodecs>=2021.11.20",
    "matplotlib>==3.5.1",
    "numpy>=1.21.5",
    "opencv-python>=4.5.5",
    "openslide-python>=1.1.2",
    "pandas>=1.3.5",
    "pillow>=9.0.1",
    "pydicom>=2.3.0",
    "pyyaml>=6.0",
    "requests>=2.27.1",
    "scikit-image>=0.19.2",
    "scikit-learn>=1.0.2",
    "scipy>=1.7.3",
    "shapely>=1.8.1",
    "tifffile>=2021.11.2",
    "torch>=1.11.0",
    "torchvision>=0.12.0",
    "tqdm>=4.63.1",
    "umap-learn>=0.5.2",
    "wsidicom>=0.2.0",
    "zarr>=2.11.1",
]

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
    url="https://github.com/TissueImageAnalytics/tiatoolbox",
    version="1.2.0",
    zip_safe=False,
)
