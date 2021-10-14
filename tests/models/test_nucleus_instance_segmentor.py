import pathlib
import shutil

import joblib
import numpy as np

from tiatoolbox.models import (
    IOSegmentorConfig,
    NucleusInstanceSegmentor,
    SemanticSegmentor,
)
from tiatoolbox.utils.metrics import f1_detection

BATCH_SIZE = 16
ON_GPU = True
# ----------------------------------------------------


def _rm_dir(path):
    """Helper func to remove directory."""
    shutil.rmtree(path, ignore_errors=True)


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
    ioconfig = IOSegmentorConfig(
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

    # test for vertical boundary boxes
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


def test_functionality(remote_sample, tmp_path):
    """Functionality test for nuclei instance segmentor."""
    root_save_dir = pathlib.Path(tmp_path)
    sample_wsi = pathlib.Path(remote_sample("wsi1_2k_2k_svs"))

    ioconfig = IOSegmentorConfig(
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

    save_dir = f"{root_save_dir}/instance/"
    # test run without worker first
    _rm_dir(save_dir)
    inst_segmentor = NucleusInstanceSegmentor(
        batch_size=BATCH_SIZE,
        num_postproc_workers=0,
        pretrained_model="hovernet_fast-pannuke",
    )

    output = inst_segmentor.predict(
        [sample_wsi],
        mode="wsi",
        ioconfig=ioconfig,
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    inst_dict_a = joblib.load(f"{output[0][1]}.dat")

    # **
    # then test run when using workers, will then compare results
    # to ensure the predictions are the same
    _rm_dir(save_dir)
    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        batch_size=BATCH_SIZE,
        num_postproc_workers=2,
    )
    assert inst_segmentor.num_postproc_workers == 2
    output = inst_segmentor.predict(
        [sample_wsi],
        mode="wsi",
        ioconfig=ioconfig,
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    inst_dict_b = joblib.load(f"{output[0][1]}.dat")
    inst_coords_a = np.array([v["centroid"] for v in inst_dict_a.values()])
    inst_coords_b = np.array([v["centroid"] for v in inst_dict_b.values()])
    score = f1_detection(inst_coords_b, inst_coords_a, radius=1.0)
    assert score > 0.95, "Heavy loss of precision!"

    # **
    # To evaluate the precision of doing post processing on tile
    # then re-assemble without using full image prediction maps,
    # we compare its output with the output when doing
    # post processing on the entire images
    save_dir = f"{root_save_dir}/semantic/"
    _rm_dir(save_dir)
    semantic_segmentor = SemanticSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        batch_size=BATCH_SIZE,
        num_postproc_workers=2,
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
    _, inst_dict_b = semantic_segmentor.model.postproc(raw_maps)

    inst_coords_a = np.array([v["centroid"] for v in inst_dict_a.values()])
    inst_coords_b = np.array([v["centroid"] for v in inst_dict_b.values()])
    score = f1_detection(inst_coords_b, inst_coords_a, radius=1.0)
    assert score > 0.9, "Heavy loss of precision!"

    # # this is for manual debugging
    from tiatoolbox.utils.misc import imwrite
    from tiatoolbox.utils.visualization import overlay_instance_prediction
    from tiatoolbox.wsicore.wsireader import get_wsireader

    wsi_reader = get_wsireader(sample_wsi)
    thumb = wsi_reader.slide_thumbnail(resolution=0.25, units="mpp")
    thumb = overlay_instance_prediction(thumb, inst_dict_b)
    imwrite("dump.png", thumb)
