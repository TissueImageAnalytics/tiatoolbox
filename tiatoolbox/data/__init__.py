# skipcq: PTC-W6004
"""Package to define datasets available to download via TIAToolbox."""
import pathlib
import tempfile
from typing import Optional, Union
from urllib.parse import urlparse

import numpy as np
import pkg_resources
import yaml

from tiatoolbox.utils import download_data

# Load a dictionary of sample files data (names and urls)
SAMPLE_FILES_REGISTRY_PATH = pkg_resources.resource_filename(
    "tiatoolbox",
    "data/remote_samples.yaml",
)
with open(SAMPLE_FILES_REGISTRY_PATH) as registry_handle:
    SAMPLE_FILES = yaml.safe_load(registry_handle)["files"]

__all__ = ["stain_norm_target"]


def _fetch_remote_sample(
    key: str,
    tmp_path: Optional[Union[str, pathlib.Path]] = None,
) -> pathlib.Path:
    """Get the path to a sample file, after downloading from remote if required.

    Loads remote resources by name. This is done by looking up files in
    `tiatoolbox/data/remote_samples.yaml`.

    Args:
        key (str):
            The name of the resource to fetch.
        tmp_path (str or pathlib.Path):
            The directory to use for local caching. Defaults to the OS
            tmp path, see `tempfile.gettempdir` for more information.
            During testing, `tmp_path` should be set to a temporary test
            location using `tmp_path_factory.mktemp()`.

    Returns:
        pathlib.Path:
            The local path to the cached sample file after downloading.

    """
    tmp_path = (
        pathlib.Path(tmp_path) if tmp_path else pathlib.Path(tempfile.gettempdir())
    )
    if not tmp_path.is_dir():
        raise ValueError("tmp_path must be a directory.")
    sample = SAMPLE_FILES[key]
    url = "/".join(sample["url"])
    url_filename = pathlib.Path(urlparse(url).path).name
    # Get the filename from SAMPLE_FILES, else use the URL filename
    filename = SAMPLE_FILES[key].get("filename", url_filename)
    file_path = tmp_path / filename
    # Download the file if it doesn't exist

    return download_data(url, save_path=file_path, unzip=sample.get("extract", False))


def _local_sample_path(path: Union[str, pathlib.Path]) -> pathlib.Path:
    """Get the path to a data file bundled with the package.

    Args:
        path (str or pathlib.Path):
            Relative path to the package data file.

    Returns:
        pathlib.Path:
            Path within the package to the data file.


    Example:
        >>> # Get the path to a sample target image for performing
        >>> # stain normalization.
        >>> from tiatoolbox.data import stain_norm_target
        >>> img = stain_norm_target()

    """
    return pkg_resources.resource_filename(
        "tiatoolbox",
        str(pathlib.Path("data") / path),
    )


def stain_norm_target() -> np.ndarray:
    """Target image for stain normalization."""
    from tiatoolbox.utils import imread

    return imread(_local_sample_path("target_image.png"))


def small_svs() -> pathlib.Path:
    """Small SVS file for testing."""
    return _fetch_remote_sample("svs-1-small")
