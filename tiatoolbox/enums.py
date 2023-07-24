"""Enumerated types used by TIAToolbox."""

from __future__ import annotations

import contextlib
import enum
from typing import Any


class GeometryType(enum.IntEnum):
    """Enumerated type for geometry types."""

    POINT = 1
    LINE = 2
    POLYGON = 3
    MULTI_POINT = 4
    MULTI_LINE = 5
    MULTI_POLYGON = 6

    def __str__(self):
        """Return the string representation of the GeometryType."""
        return self.name.title().replace("_", " ")

    @classmethod
    def _missing_(cls, value: object) -> Any:
        """Return the GeometryType corresponding to the value."""
        if isinstance(value, str):
            with contextlib.suppress(KeyError):
                return cls[value.upper().replace(" ", "_")]
        return super()._missing_(value)

    @classmethod
    def from_string(cls: type[GeometryType], string: str) -> GeometryType:
        """Return the GeometryType corresponding to the string."""
        return cls[string.upper().replace(" ", "_")]
