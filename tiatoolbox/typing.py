"""Define Variable types for TIAToolbox."""
from typing import Literal
from typing import SupportsFloat as Numeric
from typing import Tuple, Union

import numpy as np

Resolution = Union[Numeric, Tuple[Numeric, Numeric], np.ndarray]
Units = Literal["mpp", "power", "baseline", "level"]
NumPair = Tuple[Numeric, Numeric]
IntPair = Tuple[int, int]
Bounds = Tuple[Numeric, Numeric, Numeric, Numeric]
IntBounds = Tuple[int, int, int, int]
