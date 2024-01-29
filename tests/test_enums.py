"""Tests for the enumerated types used by TIAToolbox."""

import pytest

from tiatoolbox.enums import GeometryType


def test_geometrytype_missing() -> None:
    """Test that GeometryType.MISSING is returned when given None."""
    with pytest.raises(ValueError, match="not a valid GeometryType"):
        GeometryType(None)


def test_geometrytype_point_from_string() -> None:
    """Init GeometryType.POINT from string."""
    assert GeometryType("Point") == GeometryType.POINT


def test_geometrytype_linestring_from_string() -> None:
    """Init GeometryType.LINE_STRING from string."""
    assert GeometryType("LineString") == GeometryType.LINE_STRING


def test_geometrytype_polygon_from_string() -> None:
    """Init GeometryType.POLYGON from string."""
    assert GeometryType("Polygon") == GeometryType.POLYGON


def test_geometrytype_multipoint_from_string() -> None:
    """Init GeometryType.MULTI_POINT from string."""
    assert GeometryType("MultiPoint") == GeometryType.MULTI_POINT


def test_geometrytype_multilinestring_from_string() -> None:
    """Init GeometryType.MULTI_LINE_STRING from string."""
    assert GeometryType("MultiLineString") == GeometryType.MULTI_LINE_STRING


def test_geometrytype_multipolygon_from_string() -> None:
    """Init GeometryType.MULTI_POLYGON from string."""
    assert GeometryType("MultiPolygon") == GeometryType.MULTI_POLYGON
