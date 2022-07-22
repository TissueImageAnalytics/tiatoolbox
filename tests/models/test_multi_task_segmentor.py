"""Unit test package for HoVerNet+."""

# ! The garbage collector
import gc
import multiprocessing
import pathlib
import shutil

import joblib
import numpy as np
import pytest

from tiatoolbox.models import IOSegmentorConfig, MultiTaskSegmentor
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.misc import imwrite

ON_GPU = toolbox_env.has_gpu()
# The batch size value here is based on two TitanXP, each with 12GB
BATCH_SIZE = 1 if not ON_GPU else 16
try:
    NUM_POSTPROC_WORKERS = multiprocessing.cpu_count()
except NotImplementedError:
    NUM_POSTPROC_WORKERS = 2

# ----------------------------------------------------


def _rm_dir(path):
    """Helper func to remove directory."""
    shutil.rmtree(path, ignore_errors=True)


def _crash_func(x):
    """Helper to induce crash."""
    raise ValueError("Propataion Crash.")


def helper_tile_info():
    """Helper function for tile information."""
    predictor = MultiTaskSegmentor(model="A")
    # ! assuming the tiles organized as follows (coming out from
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

    return predictor._get_tile_info([16, 16], ioconfig)


# ----------------------------------------------------


def test_get_tile_info():
    """Test for getting tile info."""
    info = helper_tile_info()
    _, flag = info[0]  # index 0 should be full grid, removal
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


def test_vertical_boundary_boxes():
    """Test for vertical boundary boxes."""
    info = helper_tile_info()
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


def test_horizontal_boundary_boxes():
    """Test for horizontal boundary boxes."""
    info = helper_tile_info()
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


def test_cross_section_boundary_boxes():
    """Test for cross-section boundary boxes."""
    info = helper_tile_info()
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


def test_crash_segmentor(remote_sample, tmp_path):
    """Test engine crash when given malformed input."""
    root_save_dir = pathlib.Path(tmp_path)
    sample_wsi_svs = pathlib.Path(remote_sample("svs-1-small"))
    sample_wsi_msk = remote_sample("small_svs_tissue_mask")
    sample_wsi_msk = np.load(sample_wsi_msk).astype(np.uint8)
    imwrite(f"{tmp_path}/small_svs_tissue_mask.jpg", sample_wsi_msk)
    sample_wsi_msk = tmp_path.joinpath("small_svs_tissue_mask.jpg")

    save_dir = f"{root_save_dir}/instance/"

    # resolution for travis testing, not the correct ones
    resolution = 4.0
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": resolution}],
        output_resolutions=[
            {"units": "mpp", "resolution": resolution},
            {"units": "mpp", "resolution": resolution},
            {"units": "mpp", "resolution": resolution},
        ],
        margin=128,
        tile_shape=[512, 512],
        patch_input_shape=[256, 256],
        patch_output_shape=[164, 164],
        stride_shape=[164, 164],
    )
    instance_segmentor = MultiTaskSegmentor(
        batch_size=BATCH_SIZE,
        num_postproc_workers=2,
        pretrained_model="hovernet_fast-pannuke",
    )

    # * Test crash propagation when parallelize post processing
    _rm_dir(save_dir)
    instance_segmentor.model.postproc_func = _crash_func
    with pytest.raises(ValueError, match=r"Propataion Crash."):
        instance_segmentor.predict(
            [sample_wsi_svs],
            masks=[sample_wsi_msk],
            mode="wsi",
            ioconfig=ioconfig,
            on_gpu=ON_GPU,
            crash_on_exception=True,
            save_dir=save_dir,
        )
    _rm_dir(tmp_path)


@pytest.mark.skipif(
    toolbox_env.running_on_ci() or not toolbox_env.has_gpu(),
    reason="Local test on machine with GPU.",
)
def test_functionality_local(remote_sample, tmp_path):
    """Local functionality test for multi task segmentor."""
    gc.collect()
    root_save_dir = pathlib.Path(tmp_path)
    mini_wsi_svs = pathlib.Path(remote_sample("svs-1-small"))

    save_dir = f"{root_save_dir}/multitask/"
    _rm_dir(save_dir)
    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="hovernetplus-oed",
        batch_size=BATCH_SIZE,
        num_postproc_workers=NUM_POSTPROC_WORKERS,
    )
    output = multi_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir,
    )

    inst_dict = joblib.load(f"{output[0][1]}.0.dat")
    layer_map = np.load(f"{output[0][1]}.1.npy")

    assert len(inst_dict) > 0, "Must have some nuclei"
    assert layer_map is not None, "Must have some layers."
    _rm_dir(tmp_path)


def test_functionality_hovernetplus_travis(remote_sample, tmp_path):
    """Functionality test for multi task segmentor."""
    root_save_dir = pathlib.Path(tmp_path)
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))
    required_dims = (258, 258)
    # above image is 512 x 512 at 0.252 mpp resolution. This is 258 x 258 at 0.500 mpp.

    save_dir = f"{root_save_dir}/multi/"
    _rm_dir(save_dir)

    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="hovernetplus-oed",
        batch_size=BATCH_SIZE,
        num_postproc_workers=NUM_POSTPROC_WORKERS,
    )
    output = multi_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir,
    )

    inst_dict = joblib.load(f"{output[0][1]}.0.dat")
    layer_map = np.load(f"{output[0][1]}.1.npy")

    assert len(inst_dict) > 0, "Must have some nuclei."
    assert layer_map is not None, "Must have some layers."
    assert (
        layer_map.shape == required_dims
    ), "Output layer map dimensions must be same as the expected output shape"
    _rm_dir(tmp_path)


def test_functionality_hovernet_travis(remote_sample, tmp_path):
    """Functionality test for multi task segmentor."""
    root_save_dir = pathlib.Path(tmp_path)
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))

    save_dir = f"{root_save_dir}/multi/"
    _rm_dir(save_dir)

    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        batch_size=BATCH_SIZE,
        num_postproc_workers=NUM_POSTPROC_WORKERS,
    )
    output = multi_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir,
    )

    inst_dict = joblib.load(f"{output[0][1]}.0.dat")

    assert len(inst_dict) > 0, "Must have some nuclei."
    _rm_dir(tmp_path)
