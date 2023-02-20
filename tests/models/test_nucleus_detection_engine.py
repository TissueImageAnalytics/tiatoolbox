"""Tests for NucleusDetector."""

from tiatoolbox.models.engine.nucleus_detection import NucleusDetector


def test_nucleus_detector_engine(sample_svs):
    """Test for nucleus detection engine."""
    _ = NucleusDetector(pretrained_model="mapde-conic")
