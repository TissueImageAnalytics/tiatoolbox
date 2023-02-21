"""Tests for NucleusDetector."""

import pathlib

import pandas as pd
import pytest

from tiatoolbox.models.engine.nucleus_detector import NucleusDetector


def test_nucleus_detector_engine(remote_sample, tmp_path):
    """Test for nucleus detection engine."""
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))

    nucleus_detector = NucleusDetector(pretrained_model="mapde-conic")
    _ = nucleus_detector.predict(
        [mini_wsi_svs], mode="wsi", save_dir=tmp_path / "output"
    )

    coordinates = pd.read_csv(tmp_path / "output" / "0.locations.0.csv")
    assert coordinates.x[0] == pytest.approx(53, abs=2)
    assert coordinates.x[1] == pytest.approx(55, abs=2)
    assert coordinates.y[0] == pytest.approx(107, abs=2)
    assert coordinates.y[1] == pytest.approx(127, abs=2)
