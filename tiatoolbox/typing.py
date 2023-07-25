"""Define Variable types for TIAToolbox."""
from typing import Callable, Literal, Sequence, SupportsFloat, TypeAlias

import numpy as np
from shapely.geometry import LineString, Point, Polygon

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
NumPair: TypeAlias = tuple[SupportsFloat, SupportsFloat]
IntPair: TypeAlias = tuple[int, int]

# WSIReader
Resolution: TypeAlias = SupportsFloat | NumPair | np.ndarray | Sequence[SupportsFloat]
Units: TypeAlias = Literal["mpp", "power", "baseline", "level"]
Bounds: TypeAlias = tuple[SupportsFloat, SupportsFloat, SupportsFloat, SupportsFloat]
IntBounds: TypeAlias = tuple[int, int, int, int]

# Annotation Store
Geometry: TypeAlias = Point | LineString | Polygon
Properties: TypeAlias = JSON
QueryGeometry = Bounds | Geometry
CallablePredicate = Callable[[Properties], bool]
CallableSelect = Callable[[Properties], Properties]
Predicate = str | bytes | CallablePredicate
Select = str | bytes | CallableSelect
