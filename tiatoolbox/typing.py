from numbers import Number
from typing import Literal, Tuple, Union

import numpy as np

Resolution = Union[Number, Tuple[Number, Number], np.ndarray]
Units = Literal["mpp", "power", "baseline", "level"]
