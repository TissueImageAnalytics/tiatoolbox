"""Define Variable types for TIAToolbox."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, SupportsFloat

import numpy as np
from shapely.geometry import LineString, Point, Polygon  # type: ignore[import-untyped]

# Proper type annotations for shapely is not yet available.


JSON = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
NumPair = tuple[SupportsFloat, SupportsFloat]
IntPair = tuple[int, int]

# WSIReader
Resolution = SupportsFloat | NumPair | np.ndarray | Sequence[SupportsFloat]
Units = Literal["mpp"], Literal["power"], Literal["baseline"], Literal["level"]
Bounds = tuple[SupportsFloat, SupportsFloat, SupportsFloat, SupportsFloat]
IntBounds = tuple[int, int, int, int]

# Annotation Store
Geometry = Point | LineString | Polygon
Properties = JSON  # Could define this using a TypedDict
QueryGeometry = Bounds | Geometry
CallablePredicate = Callable[[Properties], bool]
CallableSelect = Callable[[Properties], Properties]
Predicate = str | bytes | CallablePredicate
Select = str | bytes | CallableSelect
NumpyPadLiteral = Literal[
    "constant",
    "edge",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "reflect",
    "symmetric",
    "wrap",
    "empty",
]
