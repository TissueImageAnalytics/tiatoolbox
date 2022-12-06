"""Tests for Nucleus Instance Segmentor."""

import copy

# ! The garbage collector
import gc
import pathlib
import shutil

import joblib
import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from tiatoolbox import cli
from tiatoolbox.models import (
    IOSegmentorConfig,
    NucleusInstanceSegmentor,
    SemanticSegmentor,
)
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.engine.nucleus_instance_segmentor import (
    _process_tile_predictions,
)
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.metrics import f1_detection
from tiatoolbox.utils.misc import imwrite
from tiatoolbox.wsicore.wsireader import WSIReader

ON_GPU = toolbox_env.has_gpu()
# The value is based on 2 TitanXP each with 12GB
BATCH_SIZE = 1 if not ON_GPU else 16

# ----------------------------------------------------


def _rm_dir(path):
    """Helper func to remove directory."""
    shutil.rmtree(path, ignore_errors=True)


def _crash_func(x):
    """Helper to induce crash."""
    raise ValueError("Propataion Crash.")


def helper_tile_info():
    """Helper function for tile information."""
    predictor = NucleusInstanceSegmentor(model="A")
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
    instance_segmentor = NucleusInstanceSegmentor(
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


def test_functionality_travis(remote_sample, tmp_path):
    """Functionality test for nuclei instance segmentor."""
    gc.collect()
    root_save_dir = pathlib.Path(tmp_path)
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))

    resolution = 2.0

    reader = WSIReader.open(mini_wsi_svs)
    thumb = reader.slide_thumbnail(resolution=resolution, units="mpp")
    mini_wsi_jpg = f"{tmp_path}/mini_svs.jpg"
    imwrite(mini_wsi_jpg, thumb)

    save_dir = f"{root_save_dir}/instance/"

    # * test run on wsi, test run with worker
    # resolution for travis testing, not the correct ones
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": resolution}],
        output_resolutions=[
            {"units": "mpp", "resolution": resolution},
            {"units": "mpp", "resolution": resolution},
        ],
        margin=128,
        tile_shape=[1024, 1024],
        patch_input_shape=[256, 256],
        patch_output_shape=[164, 164],
        stride_shape=[164, 164],
    )

    _rm_dir(save_dir)

    inst_segmentor = NucleusInstanceSegmentor(
        batch_size=1,
        num_loader_workers=0,
        num_postproc_workers=0,
        pretrained_model="hovernet_fast-pannuke",
    )
    inst_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        ioconfig=ioconfig,
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir,
    )

    # clean up
    _rm_dir(tmp_path)


def test_functionality_merge_tile_predictions_travis(remote_sample, tmp_path):
    """Functional tests for merging tile predictions."""
    gc.collect()  # Force clean up everything on hold
    save_dir = pathlib.Path(f"{tmp_path}/output")
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))

    resolution = 0.5
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

    # mainly to hook the merge prediction function
    inst_segmentor = NucleusInstanceSegmentor(
        batch_size=BATCH_SIZE,
        num_postproc_workers=0,
        pretrained_model="hovernet_fast-pannuke",
    )

    _rm_dir(save_dir)
    semantic_segmentor = SemanticSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        batch_size=BATCH_SIZE,
        num_postproc_workers=0,
    )

    output = semantic_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        on_gpu=ON_GPU,
        ioconfig=ioconfig,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    raw_maps = [np.load(f"{output[0][1]}.raw.{head_idx}.npy") for head_idx in range(3)]
    raw_maps = [[v] for v in raw_maps]  # mask it as patch output

    dummy_reference = {i: {"box": np.array([0, 0, 32, 32])} for i in range(1000)}
    dummy_flag_mode_list = [
        [[1, 1, 0, 0], 1],
        [[0, 0, 1, 1], 2],
        [[1, 1, 1, 1], 3],
        [[0, 0, 0, 0], 0],
    ]

    inst_segmentor._wsi_inst_info = copy.deepcopy(dummy_reference)
    inst_segmentor._futures = [[dummy_reference, dummy_reference.keys()]]
    inst_segmentor._merge_post_process_results()
    assert len(inst_segmentor._wsi_inst_info) == 0

    blank_raw_maps = [np.zeros_like(v) for v in raw_maps]
    _process_tile_predictions(
        ioconfig=ioconfig,
        tile_bounds=np.array([0, 0, 512, 512]),
        tile_flag=dummy_flag_mode_list[0][0],
        tile_mode=dummy_flag_mode_list[0][1],
        tile_output=[[np.array([0, 0, 512, 512]), blank_raw_maps]],
        ref_inst_dict=dummy_reference,
        postproc=semantic_segmentor.model.postproc_func,
        merge_predictions=semantic_segmentor.merge_prediction,
    )

    for tile_flag, tile_mode in dummy_flag_mode_list:
        _process_tile_predictions(
            ioconfig=ioconfig,
            tile_bounds=np.array([0, 0, 512, 512]),
            tile_flag=tile_flag,
            tile_mode=tile_mode,
            tile_output=[[np.array([0, 0, 512, 512]), raw_maps]],
            ref_inst_dict=dummy_reference,
            postproc=semantic_segmentor.model.postproc_func,
            merge_predictions=semantic_segmentor.merge_prediction,
        )

    # test exception flag
    tile_flag = [0, 0, 0, 0]
    with pytest.raises(ValueError, match=r".*Unknown tile mode.*"):
        _process_tile_predictions(
            ioconfig=ioconfig,
            tile_bounds=np.array([0, 0, 512, 512]),
            tile_flag=tile_flag,
            tile_mode=-1,
            tile_output=[[np.array([0, 0, 512, 512]), raw_maps]],
            ref_inst_dict=dummy_reference,
            postproc=semantic_segmentor.model.postproc_func,
            merge_predictions=semantic_segmentor.merge_prediction,
        )
    _rm_dir(tmp_path)


@pytest.mark.skipif(
    toolbox_env.running_on_ci() or not ON_GPU,
    reason="Local test on machine with GPU.",
)
def test_functionality_local(remote_sample, tmp_path):
    """Local functionality test for nuclei instance segmentor."""
    root_save_dir = pathlib.Path(tmp_path)
    save_dir = pathlib.Path(f"{tmp_path}/output")
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_1k_1k_svs"))

    # * generate full output w/o parallel post processing worker first
    _rm_dir(save_dir)
    inst_segmentor = NucleusInstanceSegmentor(
        batch_size=8,
        num_postproc_workers=0,
        pretrained_model="hovernet_fast-pannuke",
    )
    output = inst_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        on_gpu=True,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    inst_dict_a = joblib.load(f"{output[0][1]}.dat")

    # * then test run when using workers, will then compare results
    # * to ensure the predictions are the same
    _rm_dir(save_dir)
    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        batch_size=BATCH_SIZE,
        num_postproc_workers=2,
    )
    assert inst_segmentor.num_postproc_workers == 2
    output = inst_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        on_gpu=True,
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
        [mini_wsi_svs],
        mode="wsi",
        on_gpu=True,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    raw_maps = [np.load(f"{output[0][1]}.raw.{head_idx}.npy") for head_idx in range(3)]
    _, inst_dict_b = semantic_segmentor.model.postproc(raw_maps)

    inst_coords_a = np.array([v["centroid"] for v in inst_dict_a.values()])
    inst_coords_b = np.array([v["centroid"] for v in inst_dict_b.values()])
    score = f1_detection(inst_coords_b, inst_coords_a, radius=1.0)
    assert score > 0.9, "Heavy loss of precision!"
    _rm_dir(tmp_path)


def test_cli_nucleus_instance_segment_ioconfig(remote_sample, tmp_path):
    """Test for nucleus segmentation with IOconfig."""
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))
    output_path = tmp_path / "output"

    resolution = 2.0

    reader = WSIReader.open(mini_wsi_svs)
    thumb = reader.slide_thumbnail(resolution=resolution, units="mpp")
    mini_wsi_jpg = f"{tmp_path}/mini_svs.jpg"
    imwrite(mini_wsi_jpg, thumb)

    fetch_pretrained_weights(
        "hovernet_fast-pannuke", str(tmp_path.joinpath("hovernet_fast-pannuke.pth"))
    )

    # resolution for travis testing, not the correct ones
    config = {
        "input_resolutions": [{"units": "mpp", "resolution": resolution}],
        "output_resolutions": [
            {"units": "mpp", "resolution": resolution},
            {"units": "mpp", "resolution": resolution},
            {"units": "mpp", "resolution": resolution},
        ],
        "margin": 128,
        "tile_shape": [512, 512],
        "patch_input_shape": [256, 256],
        "patch_output_shape": [164, 164],
        "stride_shape": [164, 164],
        "save_resolution": {"units": "mpp", "resolution": 8.0},
    }

    with open(tmp_path.joinpath("config.yaml"), "w") as fptr:
        yaml.dump(config, fptr)

    runner = CliRunner()
    nucleus_instance_segment_result = runner.invoke(
        cli.main,
        [
            "nucleus-instance-segment",
            "--img-input",
            str(mini_wsi_jpg),
            "--pretrained-weights",
            str(tmp_path.joinpath("hovernet_fast-pannuke.pth")),
            "--num-loader-workers",
            str(0),
            "--num-postproc-workers",
            str(0),
            "--mode",
            "tile",
            "--output-path",
            str(output_path),
            "--yaml-config-path",
            tmp_path.joinpath("config.yaml"),
        ],
    )

    assert nucleus_instance_segment_result.exit_code == 0
    assert output_path.joinpath("0.dat").exists()
    assert output_path.joinpath("file_map.dat").exists()
    assert output_path.joinpath("results.json").exists()
    _rm_dir(tmp_path)


def test_cli_nucleus_instance_segment(remote_sample, tmp_path):
    """Test for nucleus segmentation."""
    mini_wsi_svs = pathlib.Path(remote_sample("wsi4_512_512_svs"))
    output_path = tmp_path / "output"

    runner = CliRunner()
    nucleus_instance_segment_result = runner.invoke(
        cli.main,
        [
            "nucleus-instance-segment",
            "--img-input",
            str(mini_wsi_svs),
            "--mode",
            "wsi",
            "--num-loader-workers",
            str(0),
            "--num-postproc-workers",
            str(0),
            "--output-path",
            str(output_path),
        ],
    )

    assert nucleus_instance_segment_result.exit_code == 0
    assert output_path.joinpath("0.dat").exists()
    assert output_path.joinpath("file_map.dat").exists()
    assert output_path.joinpath("results.json").exists()
    _rm_dir(tmp_path)
