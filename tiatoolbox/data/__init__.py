# skipcq: PTC-W6004  # noqa: ERA001
"""Package to define datasets available to download via TIAToolbox."""
from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if sys.version_info >= (3, 9):  # pragma: no cover
    import importlib.resources as importlib_resources
else:  # pragma: no cover
    import importlib_resources  # To support Python 3.8

import requests

from tiatoolbox import logger, read_registry_files

if TYPE_CHECKING:
    import numpy as np

# Load a dictionary of sample files data (names and urls)
SAMPLE_FILES = read_registry_files("data/remote_samples.yaml")["files"]

__all__ = ["stain_norm_target"]


def _fetch_remote_sample(
    key: str,
    tmp_path: str | Path | None = None,
) -> Path:
    """Get the path to a sample file, after downloading from remote if required.

    Loads remote resources by name. This is done by looking up files in
    `tiatoolbox/data/remote_samples.yaml`.

    Args:
        key (str):
            The name of the resource to fetch.
        tmp_path (str or Path):
            The directory to use for local caching. Defaults to the OS
            tmp path, see `tempfile.gettempdir` for more information.
            During testing, `tmp_path` should be set to a temporary test
            location using `tmp_path_factory.mktemp()`.

    Returns:
        Path:
            The local path to the cached sample file after downloading.

    """
    tmp_path = Path(tmp_path) if tmp_path else Path(tempfile.gettempdir())
    if not tmp_path.is_dir():
        msg = "tmp_path must be a directory."
        raise ValueError(msg)
    sample = SAMPLE_FILES[key]
    url = "/".join(sample["url"])
    url_filename = Path(urlparse(url).path).name
    # Get the filename from SAMPLE_FILES, else use the URL filename
    filename = SAMPLE_FILES[key].get("filename", url_filename)
    file_path = tmp_path / filename
    # Download the file if it doesn't exist
    if not file_path.is_file():
        logger.info(f"Downloading sample file {filename}")
        # Start the connection with a 5s timeout to avoid hanging forever
        response = requests.get(url, stream=True, timeout=5)
        # Raise an exception for status codes != 200
        response.raise_for_status()
        # Write the file in blocks of 1024 bytes to avoid running out of memory
        with Path.open(file_path, "wb") as handle:
            for block in response.iter_content(1024):
                handle.write(block)
        # Extract the (zip) archive contents if required
        if sample.get("extract"):
            logger.info(f"Extracting sample file {filename}")
            extract_path = tmp_path / filename.replace(".zip", "")
            with zipfile.ZipFile(file_path, "r") as zip_handle:
                zip_handle.extractall(path=extract_path)
            file_path = extract_path
        return file_path
    logger.info(f"Skipping download of sample file {filename}.")
    if sample.get("extract"):
        file_path = tmp_path / filename.replace(".zip", "")
    return file_path


def _local_sample_path(path: str | Path) -> Path:
    """Get the path to a data file bundled with the package.

    Args:
        path (str or Path):
            Relative path to the package data file.

    Returns:
        Path:
            Path within the package to the data file.


    Example:
        >>> # Get the path to a sample target image for performing
        >>> # stain normalization.
        >>> from tiatoolbox.data import stain_norm_target
        >>> img = stain_norm_target()

    """
    file_path = importlib_resources.files("tiatoolbox") / str(Path("data") / path)
    with importlib_resources.as_file(file_path) as path:
        return path


def stain_norm_target() -> np.ndarray:
    """Target image for stain normalization."""
    from tiatoolbox.utils import imread

    return imread(_local_sample_path("target_image.png"))


def small_svs() -> Path:
    """Small SVS file for testing."""
    return _fetch_remote_sample("svs-1-small")
