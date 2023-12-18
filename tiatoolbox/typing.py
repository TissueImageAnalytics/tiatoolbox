"""Define Variable types for TIAToolbox."""
from __future__ import annotations

from typing import Callable, Dict, List, Literal, Sequence, SupportsFloat, Tuple, Union

import numpy as np
from shapely.geometry import LineString, Point, Polygon  # type: ignore[import-untyped]

# Proper type annotations for shapely is not yet available.


JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]
NumPair = Tuple[SupportsFloat, SupportsFloat]
IntPair = Tuple[int, int]

# WSIReader
Resolution = Union[SupportsFloat, NumPair, np.ndarray, Sequence[SupportsFloat]]
Units = Literal["mpp", "power", "baseline", "level"]
Bounds = Tuple[SupportsFloat, SupportsFloat, SupportsFloat, SupportsFloat]
IntBounds = Tuple[int, int, int, int]

# Annotation Store
Geometry = Union[Point, LineString, Polygon]
Properties = JSON
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
