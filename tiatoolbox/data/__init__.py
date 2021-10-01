import pathlib
import tempfile
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import pkg_resources
import requests
import yaml

from tiatoolbox.utils.misc import imread

# Load a dictionary of sample files data (names and urls)
SAMPLE_FILES_REGISTRY_PATH = pkg_resources.resource_filename(
    "tiatoolbox", "data/remote_samples.yaml"
)
with open(SAMPLE_FILES_REGISTRY_PATH) as registry_handle:
    SAMPLE_FILES = yaml.safe_load(registry_handle)["files"]

__all__ = ["stainnorm_target"]


def _fetch_remote_sample(
    key: str, tmp_path: Optional[pathlib.Path] = None
) -> pathlib.Path:
    """Get the path to a sample file, after downloading from remote if required.

    Loads remote resources by name. This is done by looking up files in
    `tiatoolbox/data/remote_samples.yaml`.

    Args:
        key (str): The name of the resource to fetch.
        tmp_path (pathlib.Path): The directory to use for local caching.
            Defaults to the OS tmp path, see `tempfile.gettempdir` for more
            information. During testing, `tmp_path` should be set to a
            temporary test location using `tmp_path_factory.mketmp()`.

    Returns:
        pathlib.Path: The local path to the cached sample file after downloading.

    """
    if tmp_path is None:
        tmp_path = tempfile.gettempdir()
    if not tmp_path.is_dir():
        raise ValueError("tmp_path must be a directory.")
    url = "/".join(SAMPLE_FILES[key]["url"])
    url_filename = pathlib.Path(urlparse(url).path).name
    # Get the filename from SAMPLE_FILES, else use the URL filename
    filename = SAMPLE_FILES[key].get("filename", url_filename)
    file_path = tmp_path / filename
    if not pathlib.Path(file_path).is_file():
        print(f"Downloading sample file {filename}")
        # Start the connection with a 5s timeout to avoid hanging forever
        response = requests.get(url, stream=True, timeout=5)
        # Raise an exception for status codes != 200
        response.raise_for_status()
        # Write the file in blocks of 1024 bytes to avoid running out of memory
        with open(file_path, "wb") as handle:
            for block in response.iter_content(1024):
                handle.write(block)
        return file_path
    print(f"Skipping {filename}")
    return file_path


def _local_sample_path(path: pathlib.Path) -> pathlib.Path:
    """Get the path to a data file bundled with the package.

    Args:
        path (pathlib.Path): Relative path to the package data file.

    Returns:
        pathlib.Path: Path within the package to the data file.


    Example:
        >>> from tiatoolbox.data import stainnorm_target
        >>> img = stainnorm_target()

    """
    return pkg_resources.resource_filename(
        "tiatoolbox", str(pathlib.Path("data") / path)
    )


def stainnorm_target() -> np.ndarray:
    """Target image for stain normalisation."""
    return imread(_local_sample_path("target_image.png"))
