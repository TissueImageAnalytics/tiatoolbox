"""Define Variable types for TIAToolbox."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Literal, SupportsFloat, Union

import numpy as np
from shapely.geometry import LineString, Point, Polygon  # type: ignore[import-untyped]

# Proper type annotations for shapely is not yet available.


JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]
NumPair = tuple[SupportsFloat, SupportsFloat]
IntPair = tuple[int, int]

# WSIReader
Resolution = Union[SupportsFloat, NumPair, np.ndarray, Sequence[SupportsFloat]]
Units = Literal["mpp", "power", "baseline", "level"]
Bounds = tuple[SupportsFloat, SupportsFloat, SupportsFloat, SupportsFloat]
IntBounds = tuple[int, int, int, int]

# Annotation Store
Geometry = Union[Point, LineString, Polygon]
Properties = JSON  # Could define this using a TypedDict
QueryGeometry = Union[Bounds, Geometry]
CallablePredicate = Callable[[Properties], bool]
CallableSelect = Callable[[Properties], Properties]
Predicate = Union[str, bytes, CallablePredicate]
Select = Union[str, bytes, CallableSelect]
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
