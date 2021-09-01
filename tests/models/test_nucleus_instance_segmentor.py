import os
import shutil
from time import time

import joblib
import numpy as np
import torch

# import sys
# sys.path.append(".")

from tiatoolbox import rcParam
from tiatoolbox.utils.metrics import f1_detection
from tiatoolbox.models.backbone.hovernet import HoVerNet
from tiatoolbox.models.segmentation import (
    IOConfigSegmentor,
    SemanticSegmentor,
    NucleusInstanceSegmentor,
)

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

    predictor = NucleusInstanceSegmentor(model="A")
    # ! assuming the tiles organized as following (coming out from
    # ! PatchExtractor). If this is broken, need to check back
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
    ioconfig = IOConfigSegmentor(
        input_resolutions=[{"units": "mpp", "resolution": 0.25}],
        output_resolutions=[
            {"units": "mpp", "resolution": 0.25},
            {"units": "mpp", "resolution": 0.25},
            {"units": "mpp", "resolution": 0.25},
        ],
        margin=1,
        tile_shape=[4, 4],
        stride_shape=[4, 4],
        patch_input_shape=[4, 4],
        patch_output_shape=[4, 4],
    )
    info = predictor._get_tile_info([16, 16], ioconfig)
    boxes, flag = info[0]  # index 0 should be full grid, removal
    # removal flag at top edges
    assert (
        np.sum(
            np.nonzero(flag[:, 0])
            != np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        )
        == 0
    ), "Fail Top"
    # removal flag at bottom edges
    assert (
        np.sum(
            np.nonzero(flag[:, 1]) != np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        )
        == 0
    ), "Fail Bottom"
    # removal flag at left edges
    assert (
        np.sum(
            np.nonzero(flag[:, 2])
            != np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15])
        )
        == 0
    ), "Fail Left"
    # removal flag at right edges
    assert (
        np.sum(
            np.nonzero(flag[:, 3]) != np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
        )
        == 0
    ), "Fail Right"

    # test for verical boundary boxes
    _boxes = np.array(
        [
            [3, 0, 5, 4],
            [7, 0, 9, 4],
            [11, 0, 13, 4],
            [3, 4, 5, 8],
            [7, 4, 9, 8],
            [11, 4, 13, 8],
            [3, 8, 5, 12],
            [7, 8, 9, 12],
            [11, 8, 13, 12],
            [3, 12, 5, 16],
            [7, 12, 9, 16],
            [11, 12, 13, 16],
        ]
    )
    _flag = np.array(
        [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
    )
    boxes, flag = info[1]
    assert np.sum(_boxes - boxes) == 0, "Wrong Vertical Bounds"
    assert np.sum(flag - _flag) == 0, "Fail Vertical Flag"

    # test for horizontal boundary boxes
    _boxes = np.array(
        [
            [0, 3, 4, 5],
            [4, 3, 8, 5],
            [8, 3, 12, 5],
            [12, 3, 16, 5],
            [0, 7, 4, 9],
            [4, 7, 8, 9],
            [8, 7, 12, 9],
            [12, 7, 16, 9],
            [0, 11, 4, 13],
            [4, 11, 8, 13],
            [8, 11, 12, 13],
            [12, 11, 16, 13],
        ]
    )
    _flag = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
        ]
    )
    boxes, flag = info[2]
    assert np.sum(_boxes - boxes) == 0, "Wrong Horizontal Bounds"
    assert np.sum(flag - _flag) == 0, "Fail Horizontal Flag"

    # test for cross-section boundary boxes
    _boxes = np.array(
        [
            [2, 2, 6, 6],
            [6, 2, 10, 6],
            [10, 2, 14, 6],
            [2, 6, 6, 10],
            [6, 6, 10, 10],
            [10, 6, 14, 10],
            [2, 10, 6, 14],
            [6, 10, 10, 14],
            [10, 10, 14, 14],
        ]
    )
    _flag = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    boxes, flag = info[3]
    assert np.sum(boxes - _boxes) == 0, "Wrong Cross Section Bounds"
    assert np.sum(flag - _flag) == 0, "Fail Cross Section Flag"


def test_functionality():
    """Functionality test for nuclei instance segmentor."""
    model = HoVerNet(num_types=6, mode="fast")
    pretrained = (
        "/home/dang/storage_1/workspace/pretrained/hovernet_fast_pannuke_pytorch.tar"
    )
    pretrained = torch.load(pretrained)["desc"]
    model.load_state_dict(pretrained)

    ioconfig = IOConfigSegmentor(
        input_resolutions=[{"units": "mpp", "resolution": 0.25}],
        output_resolutions=[
            {"units": "mpp", "resolution": 0.25},
            {"units": "mpp", "resolution": 0.25},
            {"units": "mpp", "resolution": 0.25},
        ],
        margin=128,
        tile_shape=[1024, 1024],
        patch_input_shape=[256, 256],
        patch_output_shape=[164, 164],
        stride_shape=[164, 164],
    )

    sample_wsi = "local/samples/wsi1_2k_2k.svs"
    ROOT_SAVE_DIR = "local/test/"

    save_dir = f"{ROOT_SAVE_DIR}/instance/"
    # test run without worker first
    _rm_dir(save_dir)
    inst_segmentor = NucleusInstanceSegmentor(
        model=model, batch_size=4, num_postproc_worker=0
    )

    output = inst_segmentor.predict(
        [sample_wsi],
        mode="wsi",
        ioconfig=ioconfig,
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    inst_dictA = joblib.load(f"{output[0][1]}.dat")

    # **
    # then est run when using worker, will then compare results
    # to ensure the predictions are the same
    _rm_dir(save_dir)
    inst_segmentor = NucleusInstanceSegmentor(
        model=model, batch_size=4, num_postproc_worker=2
    )
    output = inst_segmentor.predict(
        [sample_wsi],
        mode="wsi",
        ioconfig=ioconfig,
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    inst_dictB = joblib.load(f"{output[0][1]}.dat")
    inst_coords_A = np.array([v["centroid"] for v in inst_dictA.values()])
    inst_coords_B = np.array([v["centroid"] for v in inst_dictB.values()])
    score = f1_detection(inst_coords_B, inst_coords_A, radius=1.0)
    assert score > 0.95, "Heavy loss of precision!"

    # **
    # To evaluate the precision of doing post processing on tile
    # then re-assemble without using full image prediction maps,
    # we compare its output with the output when doing
    # post processing on the entire images
    save_dir = f"{ROOT_SAVE_DIR}/semantic/"
    _rm_dir(save_dir)
    semantic_segmentor = SemanticSegmentor(
        model=model, batch_size=4, num_postproc_worker=2
    )
    output = semantic_segmentor.predict(
        [sample_wsi],
        mode="wsi",
        ioconfig=ioconfig,
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    raw_maps = [np.load(f"{output[0][1]}.raw.{head_idx}.npy") for head_idx in range(3)]
    _, inst_dictB = model.postproc(raw_maps)

    inst_coords_A = np.array([v["centroid"] for v in inst_dictA.values()])
    inst_coords_B = np.array([v["centroid"] for v in inst_dictB.values()])
    score = f1_detection(inst_coords_B, inst_coords_A, radius=1.0)
    assert score > 0.9, "Heavy loss of precision!"

    # # this is for manual debugging
    # from tiatoolbox.utils.visualization import overlay_instance_prediction
    # from tiatoolbox.utils.misc import imwrite
    # from tiatoolbox.wsicore.wsireader import get_wsireader
    # wsi_reader = get_wsireader(sample_wsi)
    # thumb = wsi_reader.slide_thumbnail(resolution=0.25, units='mpp')
    # thumb = overlay_instance_prediction(thumb, inst_dictB)
    # imwrite('dump.png', thumb)
