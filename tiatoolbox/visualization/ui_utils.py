"""Some utilities for bokeh ui."""

from __future__ import annotations

from cmath import pi

import numpy as np

scale_factor = 2
init_res = 40211.5 * scale_factor * (2 / (100 * pi))
min_zoom = 0
max_zoom = 10
resolutions = [init_res / 2**lev for lev in range(min_zoom, max_zoom + 1)]


def get_level_by_extent(extent: tuple[float, float, float, float]) -> int:
    """Replicate the Bokeh tile renderer `get_level_by_extent` function."""
    x_rs = (extent[2] - extent[0]) / 1700
    y_rs = (extent[3] - extent[1]) / 1000
    resolution = np.maximum(x_rs, y_rs)

    i = 0
    for r in resolutions:
        if resolution > r:
            if i == 0:
                return 0
            return i - 1
        i += 1

    # Otherwise return the highest available resolution
    return i - 1
