"""Tests for NucleusDetector."""

import pathlib
import shutil

import pandas as pd
import pytest

from tiatoolbox.models.engine.nucleus_detector import (
    IONucleusDetectorConfig,
    NucleusDetector,
)
from tiatoolbox.utils import env_detection as toolbox_env

ON_GPU = not toolbox_env.running_on_ci() and toolbox_env.has_gpu()


def _rm_dir(path):
    """Helper func to remove directory."""
    if pathlib.Path(path).exists():
        shutil.rmtree(path, ignore_errors=True)


def check_output(path):
    """Check NucleusDetector output."""
    coordinates = pd.read_csv(path)
    assert coordinates.x[0] == pytest.approx(53, abs=2)
    assert coordinates.x[1] == pytest.approx(55, abs=2)
    assert coordinates.y[0] == pytest.approx(107, abs=2)
    assert coordinates.y[1] == pytest.approx(127, abs=2)


def test_nucleus_detector_engine(remote_sample, tmp_path):
    """Test for nucleus detection engine."""
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))

    nucleus_detector = NucleusDetector(pretrained_model="mapde-conic")
    _ = nucleus_detector.predict(
        [mini_wsi_svs],
        mode="wsi",
        save_dir=tmp_path / "output",
        on_gpu=ON_GPU,
    )

    check_output(tmp_path / "output" / "0.locations.0.csv")

    _rm_dir(tmp_path / "output")

    ioconfig = IONucleusDetectorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 0.5}],
        output_resolutions=[{"units": "mpp", "resolution": 0.5}],
        save_resolution=None,
        patch_input_shape=[252, 252],
        patch_output_shape=[252, 252],
        stride_shape=[150, 150],
    )

    nucleus_detector = NucleusDetector(pretrained_model="mapde-conic")
    _ = nucleus_detector.predict(
        [mini_wsi_svs],
        mode="wsi",
        save_dir=tmp_path / "output",
        on_gpu=ON_GPU,
        ioconfig=ioconfig,
    )

    check_output(tmp_path / "output" / "0.locations.0.csv")
