"""Enumerated types used by TIAToolbox."""

from __future__ import annotations

import contextlib
import enum
from typing import Any


class GeometryType(enum.IntEnum):
    """Integer enumerated type for different kinds of geometry.

    Initialize with an integer or string representation of the geometry type:
        1 or "Point" -> POINT
        2 or "LineString" -> LINE_STRING
        3 or "Polygon" -> POLYGON
        4 or "MultiPoint" -> MULTI_POINT
        5 or "MultiLineString" -> MULTI_LINE_STRING
        6 or "MultiPolygon" -> MULTI_POLYGON

    """

    POINT = 1
    LINE_STRING = 2
    POLYGON = 3
    MULTI_POINT = 4
    MULTI_LINE_STRING = 5
    MULTI_POLYGON = 6

    def __str__(self):
        """Return the string representation of the GeometryType."""
        return self.name.title().replace("_", " ")

    @classmethod
    def _missing_(cls, value: object) -> Any:
        """Return the GeometryType corresponding to the value."""
        if isinstance(value, str):
            with contextlib.suppress(KeyError):
                # Replace UpperCamelCase with UPPER_CAMEL_CASE
                name = "".join(
                    f"_{c}" if c.isupper() else c.upper() for c in value
                ).lstrip("_")
                return cls[name]
        return super()._missing_(value)

    @classmethod
    def from_string(cls: type[GeometryType], string: str) -> GeometryType:
        """Return the GeometryType corresponding to the string."""
        return cls[string.upper().replace("", "_")]
