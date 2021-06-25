"""Tests for visualization."""

import copy
import pathlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tiatoolbox.utils.visualization import overlay_patch_prediction
from tiatoolbox.wsicore.wsireader import get_wsireader


def test_overlay_patch_prediction(_sample_wsi_dict):
    """Test for overlaying merged patch prediction of wsi."""
    _mini_wsi_svs = pathlib.Path(_sample_wsi_dict['wsi2_4k_4k_svs'])
    _mini_wsi_pred = pathlib.Path(_sample_wsi_dict['wsi2_4k_4k_pred'])
    reader = get_wsireader(_mini_wsi_svs)

    raw, merged = joblib.load(_mini_wsi_pred)

    with pytest.raises(ValueError, match=r".*Mismatch shape.*"):
        thumb = reader.slide_thumbnail(resolution=2.77, units='mpp')
        _ = overlay_patch_prediction(thumb, merged)

    label_info_full = {
        0: ("BACKGROUND", (0, 0, 0)),
        1: ("01_TUMOR"  , (255,   0,   0)),
        2: ("02_STROMA" , (0, 255,   0)),
        3: ("03_COMPLEX", (0,   0, 255)),
        4: ("04_LYMPHO" , (0, 255, 255)),
        5: ("05_DEBRIS" , (255,   0, 255)),
        6: ("06_MUCOSA" , (255, 255,   0)),
        7: ("07_ADIPOSE", (125, 255, 255)),
        8: ("08_EMPTY"  , (255, 125, 255)),
    }

    thumb = reader.slide_thumbnail(resolution=raw['resolution'], units=raw['units'])
    with pytest.raises(ValueError, match=r".*float `img` outside.*"):
        _ = overlay_patch_prediction(thumb.astype(np.float32), merged)

    with pytest.raises(ValueError, match=r".*Missing label.*"):
        label_info_fail = copy.deepcopy(label_info_full)
        del label_info_fail[1]
        _ = overlay_patch_prediction(thumb, merged, label_info=label_info_fail)
    with pytest.raises(ValueError, match=r".*Wrong `label_info` format.*"):
        label_info_fail = copy.deepcopy(label_info_full)
        label_info_fail[1] = (1, (255, 255, 255))
        _ = overlay_patch_prediction(thumb, merged, label_info=label_info_fail)
    with pytest.raises(ValueError, match=r".*Wrong `label_info` format.*"):
        label_info_fail = copy.deepcopy(label_info_full)
        label_info_fail['ABC'] = ('ABC', (255, 255, 255))
        _ = overlay_patch_prediction(thumb, merged, label_info=label_info_fail)
    with pytest.raises(ValueError, match=r".*Wrong `label_info` format.*"):
        label_info_fail = copy.deepcopy(label_info_full)
        label_info_fail[1] = ('ABC', 'ABC')
        _ = overlay_patch_prediction(thumb, merged, label_info=label_info_fail)
    with pytest.raises(ValueError, match=r".*Wrong `label_info` format.*"):
        label_info_fail = copy.deepcopy(label_info_full)
        label_info_fail[1] = ('ABC', (255, 255))
        _ = overlay_patch_prediction(thumb, merged, label_info=label_info_fail)

    # test normal run, should not crash
    thumb_float = thumb / 255.0
    _ = overlay_patch_prediction(thumb_float, merged, label_info=label_info_full)
    _ = overlay_patch_prediction(thumb, merged, label_info=label_info_full)
    ax = plt.subplot(1, 2, 1)
    _ = overlay_patch_prediction(thumb, merged, ax=ax)
