"""Regression tests for non-square PointsPatchExtractor patches."""

from __future__ import annotations

import numpy as np

from tiatoolbox.tools.patchextraction import PointsPatchExtractor


def test_points_patch_extractor_nonsquare_center() -> None:
    """PointsPatchExtractor centers non-square patches on each requested point."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[50, 50] = [255, 0, 0]

    extractor = PointsPatchExtractor(
        input_img=img,
        locations_list=np.array([[50, 50]]),
        patch_size=(11, 41),
    )

    assert extractor.locations_df.loc[0, "x"] == 45
    assert extractor.locations_df.loc[0, "y"] == 30

    patch = extractor[0]
    assert patch.shape == (41, 11, 3)
    assert np.array_equal(patch[20, 5], [255, 0, 0])
