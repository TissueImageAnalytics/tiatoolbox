"""Define Variable types for TIAToolbox."""
from typing import Callable, Literal, Sequence, SupportsFloat

import numpy as np
from shapely.geometry import LineString, Point, Polygon

JSON = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
NumPair = tuple[SupportsFloat, SupportsFloat]
IntPair = tuple[int, int]

# WSIReader
Resolution = SupportsFloat | NumPair | np.ndarray | Sequence[SupportsFloat]
Units = Literal["mpp", "power", "baseline", "level"]
Bounds = tuple[SupportsFloat, SupportsFloat, SupportsFloat, SupportsFloat]
IntBounds = tuple[int, int, int, int]

# Annotation Store
Geometry = Point | LineString | Polygon
Properties = JSON
QueryGeometry = Bounds | Geometry
CallablePredicate = Callable[[Properties], bool]
CallableSelect = Callable[[Properties], Properties]
Predicate = str | bytes | CallablePredicate
Select = str | bytes | CallableSelect
