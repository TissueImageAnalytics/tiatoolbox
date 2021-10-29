"""Tests for visualization."""

import copy
import pathlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tiatoolbox.utils.visualization import (
    overlay_instance_prediction,
    overlay_patch_prediction,
)
from tiatoolbox.wsicore.wsireader import get_wsireader


def test_overlay_patch_prediction(sample_wsi_dict):
    """Test for overlaying merged patch prediction of wsi."""
    mini_wsi_svs = pathlib.Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    mini_wsi_pred = pathlib.Path(sample_wsi_dict["wsi2_4k_4k_pred"])
    reader = get_wsireader(mini_wsi_svs)

    raw, merged = joblib.load(mini_wsi_pred)

    thumb = reader.slide_thumbnail(resolution=2.77, units="mpp")
    with pytest.raises(ValueError, match=r".*Mismatch shape.*"):
        _ = overlay_patch_prediction(thumb, merged)

    label_info_full = {
        0: ("BACKGROUND", (0, 0, 0)),
        1: ("01_TUMOR", (255, 0, 0)),
        2: ("02_STROMA", (0, 255, 0)),
        3: ("03_COMPLEX", (0, 0, 255)),
        4: ("04_LYMPHO", (0, 255, 255)),
        5: ("05_DEBRIS", (255, 0, 255)),
        6: ("06_MUCOSA", (255, 255, 0)),
        7: ("07_ADIPOSE", (125, 255, 255)),
        8: ("08_EMPTY", (255, 125, 255)),
    }

    thumb = reader.slide_thumbnail(resolution=raw["resolution"], units=raw["units"])
    with pytest.raises(ValueError, match=r".*float `img` outside.*"):
        _ = overlay_patch_prediction(thumb.astype(np.float32), merged)

    label_info_fail = copy.deepcopy(label_info_full)
    del label_info_fail[1]
    with pytest.raises(ValueError, match=r".*Missing label.*"):
        _ = overlay_patch_prediction(thumb, merged, label_info=label_info_fail)

    label_info_fail = copy.deepcopy(label_info_full)
    label_info_fail[1] = (1, (255, 255, 255))
    with pytest.raises(ValueError, match=r".*Wrong `label_info` format.*"):
        _ = overlay_patch_prediction(thumb, merged, label_info=label_info_fail)

    label_info_fail = copy.deepcopy(label_info_full)
    label_info_fail["ABC"] = ("ABC", (255, 255, 255))
    with pytest.raises(ValueError, match=r".*Wrong `label_info` format.*"):
        _ = overlay_patch_prediction(thumb, merged, label_info=label_info_fail)

    label_info_fail = copy.deepcopy(label_info_full)
    label_info_fail[1] = ("ABC", "ABC")
    with pytest.raises(ValueError, match=r".*Wrong `label_info` format.*"):
        _ = overlay_patch_prediction(thumb, merged, label_info=label_info_fail)

    label_info_fail = copy.deepcopy(label_info_full)
    label_info_fail[1] = ("ABC", (255, 255))
    with pytest.raises(ValueError, match=r".*Wrong `label_info` format.*"):
        _ = overlay_patch_prediction(thumb, merged, label_info=label_info_fail)

    # Test normal run, should not crash.
    thumb_float = thumb / 255.0
    _ = overlay_patch_prediction(thumb_float, merged, label_info=label_info_full)
    _ = overlay_patch_prediction(thumb, merged, label_info=label_info_full)
    ax = plt.subplot(1, 2, 1)
    _ = overlay_patch_prediction(thumb, merged, ax=ax)


def test_overlay_instance_prediction():
    """Test for overlaying instance predictions on canvas."""

    inst_map = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 2, 2, 0],
            [0, 0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    # dummy instance dict
    type_colour = {
        0: ("A", (1, 0, 1)),
        1: ("B", (2, 0, 2)),
    }
    inst_dict = {
        0: {
            "centroid": [1, 1],
            "type": 0,
            "contour": [[1, 1], [1, 2], [2, 2], [2, 1]],
        },
        1: {
            "centroid": [3, 3],
            "type": 1,
            "contour": [[3, 3], [3, 4], [4, 4], [4, 3]],
        },
    }
    canvas = np.zeros(inst_map.shape + (3,), dtype=np.uint8)
    canvas = overlay_instance_prediction(
        canvas, inst_dict, draw_dot=False, type_colour=type_colour, line_thickness=1
    )
    assert np.sum(canvas[..., 0].astype(np.int32) - inst_map) == 0
    assert np.sum(canvas[..., 1].astype(np.int32) - inst_map) == -12
    assert np.sum(canvas[..., 2].astype(np.int32) - inst_map) == 0
    canvas = overlay_instance_prediction(
        canvas, inst_dict, draw_dot=True, type_colour=None, line_thickness=1
    )
