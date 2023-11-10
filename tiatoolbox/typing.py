"""Define Variable types for TIAToolbox."""
from __future__ import annotations

from typing import Callable, Dict, List, Literal, Sequence, SupportsFloat, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon  # type: ignore[import-untyped]

# Proper type annotations for shapely is not yet avaliable

JSON = Dict[str, "JSON"] | List["JSON"] | str | int | float | bool | None
NumPair = Tuple[SupportsFloat, SupportsFloat]
IntPair = Tuple[int, int]

# WSIReader
Resolution = SupportsFloat | NumPair | np.ndarray | Sequence[SupportsFloat]
Units = Literal["mpp", "power", "baseline", "level"]
Bounds = Tuple[SupportsFloat, SupportsFloat, SupportsFloat, SupportsFloat]
IntBounds = Tuple[int, int, int, int]

# Annotation Store
Geometry = Point | LineString | Polygon
Properties = JSON
QueryGeometry = Bounds | Geometry
CallablePredicate = Callable[[Properties], bool]
CallableSelect = Callable[[Properties], Properties]
Predicate = str | bytes | CallablePredicate
Select = str | bytes | CallableSelect
