"""Tests for handling toolbox remote data."""

import os
import pathlib

import numpy as np
import pytest

from tiatoolbox.data import _fetch_remote_sample, stainnorm_target
from tiatoolbox.wsicore.wsireader import get_wsireader


def test_fetch_sample(tmp_path):
    """Test for fetching sample via code name."""
    # Load a dictionary of sample files data (names and urls)
    # code name retrieved from TOOLBOX_ROOT/data/remote_samples.yaml
    tmp_path = pathlib.Path(tmp_path)
    path = _fetch_remote_sample("stainnorm-source")
    assert os.path.exists(path)
    # test if corrupted
    get_wsireader(path)

    path = _fetch_remote_sample("stainnorm-source", tmp_path)
    # assuming Path has no trailing '/'
    assert os.path.exists(path)
    assert str(tmp_path) in str(path)

    # test not directory path
    test_path = pathlib.Path(f"{tmp_path}/dummy.npy")
    np.save(test_path, np.zeros([3, 3, 3]))
    with pytest.raises(ValueError, match=r".*tmp_path must be a directory.*"):
        path = _fetch_remote_sample("wsi1_8k_8k_svs", test_path)

    #  very tiny so temporary hook here also
    arr = stainnorm_target()
    assert isinstance(arr, np.ndarray)

    _ = _fetch_remote_sample("stainnorm-source", tmp_path)
