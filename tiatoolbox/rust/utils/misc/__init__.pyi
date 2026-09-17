# tiatoolbox/rust/utils/misc.pyi

import numpy as np
from numpy.typing import NDArray

def contrast_enhancer(
    img: NDArray[np.uint8],
    low_p: int,
    high_p: int,
) -> NDArray[np.uint8]: ...
