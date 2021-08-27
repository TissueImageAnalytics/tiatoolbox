import copy
import os
import pathlib
import shutil
from time import time

import numpy as np
import pytest
import torch
import torch.multiprocessing as torch_mp
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')

from tiatoolbox import rcParam
from tiatoolbox.models.abc import ModelABC
from tiatoolbox.models.segmentation import (
    IOConfigSegmentor,
    WSIStreamDataset,
    NucleusInstanceSegmentor
)
from tiatoolbox.wsicore.wsireader import get_wsireader
from tiatoolbox.models.backbone.hovernet import HoVerNet

ON_GPU = True
# ----------------------------------------------------


def _rm_dir(path):
    """Helper func to remove directory."""
    shutil.rmtree(path, ignore_errors=True)


def _get_temp_folder_path():
    """Return unique temp folder path"""
    new_dir = os.path.join(
        rcParam["TIATOOLBOX_HOME"], f"test_model_patch_{int(time())}"
    )
    return new_dir


# ----------------------------------------------------


def test_get_tile_info():
    """Test for getting tile info."""

    predictor = NucleusInstanceSegmentor(model='A')
    # ! assuming the tiles organized as following (coming out from
    # ! PatchExtractor). If this is broke, need to check back
    # ! PatchExtractor output ordering first

    # left to right, top to bottom
    # ---------------------
    # |  0 |  1 |  2 |  3 |
    # ---------------------
    # |  4 |  5 |  6 |  7 |
    # ---------------------
    # |  8 |  9 | 10 | 11 |
    # ---------------------
    # | 12 | 13 | 14 | 15 |
    # ---------------------
    # ! assume flag index ordering: left right top bottom

    info = predictor._get_tile_info(1, [16, 16], [4, 4])
    boxes, flag = info[0]  # index 0 should be full grid, removal
    # removal flag at top edges
    assert np.sum(np.nonzero(flag[:, 0]) != np.array([
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    ])) == 0, 'Fail Top'
    # removal flag at bottom edges
    assert np.sum(np.nonzero(flag[:, 1]) != np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    ])) == 0, 'Fail Bottom'
    # removal flag at left edges
    assert np.sum(np.nonzero(flag[:, 2]) != np.array([
        1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15
    ])) == 0, 'Fail Left'
    # removal flag at right edges
    assert np.sum(np.nonzero(flag[:, 3]) != np.array([
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14
    ])) == 0, 'Fail Right'

    # test for verical boundary boxes
    _boxes = np.array([
        [3,  0, 5,  4], [7,  0, 9,  4], [11,  0, 13,  4],
        [3,  4, 5,  8], [7,  4, 9,  8], [11,  4, 13,  8],
        [3,  8, 5, 12], [7,  8, 9, 12], [11,  8, 13, 12],
        [3, 12, 5, 16], [7, 12, 9, 16], [11, 12, 13, 16],
    ])
    boxes, flag = info[1]
    assert np.sum(_boxes - boxes) == 0, 'Wrong Vertical Bounds'

    # test for horizontal boundary boxes
    _boxes = np.array([
        [0,  3,  4,  5], [4,  3, 8,  5], [8,  3, 12,  5], [12,  3, 16,  5],
        [0,  7,  4,  9], [4,  7, 8,  9], [8,  7, 12,  9], [12,  7, 16,  9],
        [0, 11,  4, 13], [4, 11, 8, 13], [8, 11, 12, 13], [12, 11, 16, 13],
    ])
    boxes, flag = info[2]
    assert np.sum(_boxes - boxes) == 0, 'Wrong Horizontal Bounds'

    # print(info[2][0])


def test_infer():
    """"""
    model = HoVerNet(num_types=6, mode='fast')
    pretrained = '/home/dang/storage_1/workspace/pretrained/hovernet_fast_pannuke_pytorch.tar'
    pretrained = torch.load(pretrained)['desc']
    model.load_state_dict(pretrained)

    ioconfig = IOConfigSegmentor(
        input_resolutions=[{"units": "mpp", "resolution": 0.25}],
        output_resolutions=[
            {"units": "mpp", "resolution": 0.25},
            {"units": "mpp", "resolution": 0.25},
            {"units": "mpp", "resolution": 0.25}
        ],
        margin=128,
        tile_shape=[1024, 1024],
        patch_input_shape=[256, 256],
        patch_output_shape=[164, 164],
        stride_shape=[164, 164],
    )
    # ioconfig.to_baseline()
    # print(ioconfig.input_resolutions)
    # print(ioconfig.output_resolutions)

    sample_wsi = 'local/samples/wsi1_2k_2k.svs'
    save_dir = 'local/test/'
    _rm_dir(save_dir)

    predictor = NucleusInstanceSegmentor(
        model=model, batch_size=4, num_postproc_worker=0)

    predictor.predict(
        [sample_wsi],
        mode='wsi',
        ioconfig=ioconfig,
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir)

# test_get_tile_info()]
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
test_infer()
print('here')
