# skipcq: PTC-W6004  # noqa: ERA001
"""Package to define datasets available to download via TIAToolbox."""

from __future__ import annotations

import importlib.resources as importlib_resources
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from huggingface_hub import hf_hub_download

from tiatoolbox import logger, read_registry_files
from tiatoolbox.utils import imread

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

from tiatoolbox.utils import download_data

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
    tmp_path = Path(tmp_path) if tmp_path else None
    if tmp_path is not None and not tmp_path.is_dir():
        msg = "tmp_path must be a directory."
        raise ValueError(msg)

    file_path = hf_hub_download(
        repo_id=SAMPLE_FILES[key]["hf_repo_id"],
        filename=SAMPLE_FILES[key]["filename"],
        subfolder=SAMPLE_FILES[key]["subfolder"],
        local_dir=tmp_path,
        repo_type="dataset",
    )
    return Path(file_path)


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
    with importlib_resources.as_file(file_path) as file_path_:
        return file_path_


def stain_norm_target() -> np.ndarray:
    """Target image for stain normalization."""
    return imread(_local_sample_path("target_image.png"))


def small_svs() -> Path:
    """Small SVS file for testing."""
    return _fetch_remote_sample("svs-1-small")
