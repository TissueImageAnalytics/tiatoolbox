# Type stubs for the Rust extension built from rust/misc.

import numpy as np
from numpy.typing import NDArray

def contrast_enhancer(
    img: NDArray[np.uint8],
    low_p: int,
    high_p: int,
) -> NDArray[np.uint8]: ...
