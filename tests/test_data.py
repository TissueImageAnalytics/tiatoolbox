"""Tests for handling toolbox remote data."""

import os
import pathlib

import numpy as np
import pytest

from tiatoolbox.data import _fetch_remote_sample, small_svs, stain_norm_target
from tiatoolbox.wsicore.wsireader import WSIReader


def test_fetch_sample(tmp_path):
    """Test for fetching sample via code name."""
    # Load a dictionary of sample files data (names and urls)
    # code name retrieved from TOOLBOX_ROOT/data/remote_samples.yaml
    tmp_path = pathlib.Path(tmp_path)
    path = _fetch_remote_sample("stainnorm-source")
    assert os.path.exists(path)
    # Test if corrupted
    WSIReader.open(path)

    path = _fetch_remote_sample("stainnorm-source", tmp_path)
    # Assuming Path has no trailing '/'
    assert os.path.exists(path)
    assert str(tmp_path) in str(path)

    # Test not directory path
    test_path = pathlib.Path(f"{tmp_path}/dummy.npy")
    np.save(test_path, np.zeros([3, 3, 3]))
    with pytest.raises(ValueError, match=r".*tmp_path must be a directory.*"):
        path = _fetch_remote_sample("wsi1_8k_8k_svs", test_path)

    # Very tiny so temporary hook here also
    arr = stain_norm_target()
    assert isinstance(arr, np.ndarray)


def test_fetch_sample_skip(tmp_path):
    """Test skipping fetching sample via code name if already downloaded."""
    # Fetch the remote file twice
    tmp_path = pathlib.Path(tmp_path)
    path_a = _fetch_remote_sample("wsi1_8k_8k_svs")
    path_b = _fetch_remote_sample("wsi1_8k_8k_svs")
    assert os.path.exists(path_a)
    assert path_a == path_b
    # Test if corrupted
    WSIReader.open(path_a)

    path = _fetch_remote_sample("wsi1_8k_8k_svs", tmp_path)
    # Assuming Path has no trailing '/'
    assert os.path.exists(path)
    assert str(tmp_path) in str(path)

    # Test not directory path
    test_path = pathlib.Path(f"{tmp_path}/dummy.npy")
    np.save(test_path, np.zeros([3, 3, 3]))
    with pytest.raises(ValueError, match=r".*tmp_path must be a directory.*"):
        path = _fetch_remote_sample("wsi1_8k_8k_svs", test_path)

    # Very tiny so temporary hook here also
    arr = stain_norm_target()
    assert isinstance(arr, np.ndarray)

    _ = _fetch_remote_sample("stainnorm-source", tmp_path)


def test_small_svs():
    """Test for fetching small SVS (CMU-1-Small-Region) sample image."""
    path = small_svs()
    # Check it exists
    assert path.exists()
    assert path.is_file()
    # Test if corrupted
    WSIReader.open(path)
