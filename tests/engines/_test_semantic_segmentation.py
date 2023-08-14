"""Test for Semantic Segmentor."""
# ! The garbage collector
import gc
import multiprocessing
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch
import torch.multiprocessing as torch_mp
import torch.nn.functional as F  # noqa: N812
import yaml
from click.testing import CliRunner
from torch import nn

from tiatoolbox import cli
from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.architecture.utils import centre_crop
from tiatoolbox.models.engine.semantic_segmentor import WSIStreamDataset
from tiatoolbox.models.models_abc import ModelABC
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils import imread, imwrite
from tiatoolbox.wsicore.wsireader import WSIReader

ON_GPU = toolbox_env.has_gpu()
# The value is based on 2 TitanXP each with 12GB
BATCH_SIZE = 1 if not ON_GPU else 16
try:
    NUM_POSTPROC_WORKERS = multiprocessing.cpu_count()
except NotImplementedError:
    NUM_POSTPROC_WORKERS = 2

# ----------------------------------------------------


def _crash_func(_x) -> None:
    """Helper to induce crash."""
    msg = "Propagation Crash."
    raise ValueError(msg)


class _CNNTo1(ModelABC):
    """Contains a convolution.

    Simple model to test functionality, this contains a single
    convolution layer which has weight=0 and bias=1.

    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1)
        self.conv.weight.data.fill_(0)
        self.conv.bias.data.fill_(1)

    def forward(self, img):
        """Define how to use layer."""
        return self.conv(img)

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o
        aggregation for a single data batch.

        Args:
            model (nn.Module): PyTorch defined model.
            batch_data (ndarray): A batch of data generated by
                torch.utils.data.DataLoader.
            on_gpu (bool): Whether to run inference on a GPU.

        """
        device = "cuda" if on_gpu else "cpu"
        ####
        model.eval()  # infer mode

        ####
        img_list = batch_data

        img_list = img_list.to(device).type(torch.float32)
        img_list = img_list.permute(0, 3, 1, 2)  # to NCHW

        hw = np.array(img_list.shape[2:])
        with torch.inference_mode():  # do not compute gradient
            logit_list = model(img_list)
            logit_list = centre_crop(logit_list, hw // 2)
            logit_list = logit_list.permute(0, 2, 3, 1)  # to NHWC
            prob_list = F.relu(logit_list)

        prob_list = prob_list.cpu().numpy()
        return [prob_list]


# -------------------------------------------------------------------------------------
# IOConfig
# -------------------------------------------------------------------------------------


def test_segmentor_ioconfig() -> None:
    """Test for IOConfig."""
    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {"units": "mpp", "resolution": 0.25},
            {"units": "mpp", "resolution": 0.50},
            {"units": "mpp", "resolution": 0.75},
        ],
        output_resolutions=[
            {"units": "mpp", "resolution": 0.25},
            {"units": "mpp", "resolution": 0.50},
        ],
        patch_input_shape=[2048, 2048],
        patch_output_shape=[1024, 1024],
        stride_shape=[512, 512],
    )
    assert ioconfig.highest_input_resolution == {"units": "mpp", "resolution": 0.25}
    ioconfig = ioconfig.to_baseline()
    assert ioconfig.input_resolutions[0]["resolution"] == 1.0
    assert ioconfig.input_resolutions[1]["resolution"] == 0.5
    assert ioconfig.input_resolutions[2]["resolution"] == 1 / 3

    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {"units": "power", "resolution": 20},
            {"units": "power", "resolution": 40},
        ],
        output_resolutions=[
            {"units": "power", "resolution": 20},
            {"units": "power", "resolution": 40},
        ],
        patch_input_shape=[2048, 2048],
        patch_output_shape=[1024, 1024],
        stride_shape=[512, 512],
        save_resolution={"units": "power", "resolution": 8.0},
    )
    assert ioconfig.highest_input_resolution == {"units": "power", "resolution": 40}
    ioconfig = ioconfig.to_baseline()
    assert ioconfig.input_resolutions[0]["resolution"] == 0.5
    assert ioconfig.input_resolutions[1]["resolution"] == 1.0
    assert ioconfig.save_resolution["resolution"] == 8.0 / 40.0

    resolutions = [
        {"units": "mpp", "resolution": 0.25},
        {"units": "mpp", "resolution": 0.50},
        {"units": "mpp", "resolution": 0.75},
    ]
    with pytest.raises(ValueError, match=r".*Unknown units.*"):
        ioconfig.scale_to_highest(resolutions, "axx")


# -------------------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------------------


def test_functional_wsi_stream_dataset(remote_sample: Callable) -> None:
    """Functional test for WSIStreamDataset."""
    gc.collect()  # Force clean up everything on hold
    mini_wsi_svs = Path(remote_sample("wsi4_512_512_svs"))

    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {"units": "mpp", "resolution": 0.25},
            {"units": "mpp", "resolution": 0.50},
            {"units": "mpp", "resolution": 0.75},
        ],
        output_resolutions=[
            {"units": "mpp", "resolution": 0.25},
            {"units": "mpp", "resolution": 0.50},
        ],
        patch_input_shape=[2048, 2048],
        patch_output_shape=[1024, 1024],
        stride_shape=[512, 512],
    )
    mp_manager = torch_mp.Manager()
    mp_shared_space = mp_manager.Namespace()

    sds = WSIStreamDataset(ioconfig, [mini_wsi_svs], mp_shared_space)
    # test for collate
    out = sds.collate_fn([None, 1, 2, 3])
    assert np.sum(out.numpy() != np.array([1, 2, 3])) == 0

    # artificial data injection
    mp_shared_space.wsi_idx = torch.tensor(0)  # a scalar
    mp_shared_space.patch_inputs = torch.from_numpy(
        np.array(
            [
                [0, 0, 256, 256],
                [256, 256, 512, 512],
            ],
        ),
    )
    mp_shared_space.patch_outputs = torch.from_numpy(
        np.array(
            [
                [0, 0, 256, 256],
                [256, 256, 512, 512],
            ],
        ),
    )
    # test read
    for _, sample in enumerate(sds):
        patch_data, _ = sample
        (patch_resolution1, patch_resolution2, patch_resolution3) = patch_data
        assert np.round(patch_resolution1.shape[0] / patch_resolution2.shape[0]) == 2
        assert np.round(patch_resolution1.shape[0] / patch_resolution3.shape[0]) == 3


# -------------------------------------------------------------------------------------
# Engine
# -------------------------------------------------------------------------------------


def test_crash_segmentor(remote_sample: Callable) -> None:
    """Functional crash tests for segmentor."""
    # # convert to pathlib Path to prevent wsireader complaint
    mini_wsi_svs = Path(remote_sample("wsi2_4k_4k_svs"))
    mini_wsi_jpg = Path(remote_sample("wsi2_4k_4k_jpg"))
    mini_wsi_msk = Path(remote_sample("wsi2_4k_4k_msk"))

    model = _CNNTo1()
    semantic_segmentor = SemanticSegmentor(batch_size=BATCH_SIZE, model=model)
    # fake injection to trigger Segmentor to create parallel
    # post-processing workers because baseline Semantic Segmentor does not support
    # post-processing out of the box. It only contains condition to create it
    # for any subclass
    semantic_segmentor.num_postproc_workers = 1

    # * test basic crash
    shutil.rmtree("output", ignore_errors=True)  # default output dir test
    with pytest.raises(TypeError, match=r".*`mask_reader`.*"):
        semantic_segmentor.filter_coordinates(mini_wsi_msk, np.array(["a", "b", "c"]))
    with pytest.raises(ValueError, match=r".*ndarray.*integer.*"):
        semantic_segmentor.filter_coordinates(
            WSIReader.open(mini_wsi_msk),
            np.array([1.0, 2.0]),
        )
    semantic_segmentor.get_reader(mini_wsi_svs, None, "wsi", auto_get_mask=True)
    with pytest.raises(ValueError, match=r".*must be a valid file path.*"):
        semantic_segmentor.get_reader(
            mini_wsi_msk,
            "not_exist",
            "wsi",
            auto_get_mask=True,
        )

    shutil.rmtree("output", ignore_errors=True)  # default output dir test
    with pytest.raises(ValueError, match=r".*provide.*"):
        SemanticSegmentor()
    with pytest.raises(ValueError, match=r".*valid mode.*"):
        semantic_segmentor.predict([], mode="abc")

    # * test not providing any io_config info when not using pretrained model
    with pytest.raises(ValueError, match=r".*provide either `ioconfig`.*"):
        semantic_segmentor.predict(
            [mini_wsi_jpg],
            mode="tile",
            on_gpu=ON_GPU,
            crash_on_exception=True,
        )
    with pytest.raises(ValueError, match=r".*already exists.*"):
        semantic_segmentor.predict([], mode="tile", patch_input_shape=(2048, 2048))
    shutil.rmtree("output", ignore_errors=True)  # default output dir test

    # * test not providing any io_config info when not using pretrained model
    with pytest.raises(ValueError, match=r".*provide either `ioconfig`.*"):
        semantic_segmentor.predict(
            [mini_wsi_jpg],
            mode="tile",
            on_gpu=ON_GPU,
            crash_on_exception=True,
        )
    shutil.rmtree("output", ignore_errors=True)  # default output dir test

    # * Test crash propagation when parallelize post-processing
    semantic_segmentor.num_postproc_workers = 2
    semantic_segmentor.model.forward = _crash_func
    with pytest.raises(ValueError, match=r"Propagation Crash."):
        semantic_segmentor.predict(
            [mini_wsi_svs],
            patch_input_shape=(2048, 2048),
            mode="wsi",
            on_gpu=ON_GPU,
            crash_on_exception=True,
            resolution=1.0,
            units="baseline",
        )

    with pytest.raises(ValueError, match=r"Invalid resolution.*"):
        semantic_segmentor.predict(
            [mini_wsi_svs],
            patch_input_shape=(2048, 2048),
            mode="wsi",
            on_gpu=ON_GPU,
            crash_on_exception=True,
        )
    shutil.rmtree("output", ignore_errors=True)
    # test ignore crash
    semantic_segmentor.predict(
        [mini_wsi_svs],
        patch_input_shape=(2048, 2048),
        mode="wsi",
        on_gpu=ON_GPU,
        crash_on_exception=False,
        resolution=1.0,
        units="baseline",
    )
    shutil.rmtree("output", ignore_errors=True)


def test_functional_segmentor_merging(tmp_path: Path) -> None:
    """Functional test for assmebling output."""
    save_dir = Path(tmp_path)

    model = _CNNTo1()
    semantic_segmentor = SemanticSegmentor(batch_size=BATCH_SIZE, model=model)

    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir()
    # predictions with HW
    _output = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
        ],
    )
    canvas = semantic_segmentor.merge_prediction(
        [4, 4],
        [np.full((2, 2), 1), np.full((2, 2), 2)],
        [[0, 0, 2, 2], [2, 2, 4, 4]],
        save_path=f"{save_dir}/raw.py",
        cache_count_path=f"{save_dir}/count.py",
    )
    assert np.sum(canvas - _output) < 1.0e-8
    # a second rerun to test overlapping count,
    # should still maintain same result
    canvas = semantic_segmentor.merge_prediction(
        [4, 4],
        [np.full((2, 2), 1), np.full((2, 2), 2)],
        [[0, 0, 2, 2], [2, 2, 4, 4]],
        save_path=f"{save_dir}/raw.py",
        cache_count_path=f"{save_dir}/count.py",
    )
    assert np.sum(canvas - _output) < 1.0e-8
    # else will leave hanging file pointer
    # and hence cant remove its folder later
    del canvas  # skipcq

    # * predictions with HWC
    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir()
    _ = semantic_segmentor.merge_prediction(
        [4, 4],
        [np.full((2, 2, 1), 1), np.full((2, 2, 1), 2)],
        [[0, 0, 2, 2], [2, 2, 4, 4]],
        save_path=f"{save_dir}/raw.py",
        cache_count_path=f"{save_dir}/count.py",
    )
    del _  # skipcq

    # * test crashing when switch to image having larger
    # * shape but still provide old links
    semantic_segmentor.merge_prediction(
        [8, 8],
        [np.full((2, 2, 1), 1), np.full((2, 2, 1), 2)],
        [[0, 0, 2, 2], [2, 2, 4, 4]],
        save_path=f"{save_dir}/raw.1.py",
        cache_count_path=f"{save_dir}/count.1.py",
    )
    with pytest.raises(ValueError, match=r".*`save_path` does not match.*"):
        semantic_segmentor.merge_prediction(
            [4, 4],
            [np.full((2, 2, 1), 1), np.full((2, 2, 1), 2)],
            [[0, 0, 2, 2], [2, 2, 4, 4]],
            save_path=f"{save_dir}/raw.1.py",
            cache_count_path=f"{save_dir}/count.py",
        )

    with pytest.raises(ValueError, match=r".*`cache_count_path` does not match.*"):
        semantic_segmentor.merge_prediction(
            [4, 4],
            [np.full((2, 2, 1), 1), np.full((2, 2, 1), 2)],
            [[0, 0, 2, 2], [2, 2, 4, 4]],
            save_path=f"{save_dir}/raw.py",
            cache_count_path=f"{save_dir}/count.1.py",
        )
    # * test non HW predictions
    with pytest.raises(ValueError, match=r".*Prediction is no HW or HWC.*"):
        semantic_segmentor.merge_prediction(
            [4, 4],
            [np.full((2,), 1), np.full((2,), 2)],
            [[0, 0, 2, 2], [2, 2, 4, 4]],
            save_path=f"{save_dir}/raw.py",
            cache_count_path=f"{save_dir}/count.1.py",
        )

    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir()

    # * with an out of bound location
    canvas = semantic_segmentor.merge_prediction(
        [4, 4],
        [
            np.full((2, 2), 1),
            np.full((2, 2), 2),
            np.full((2, 2), 3),
            np.full((2, 2), 4),
        ],
        [[0, 0, 2, 2], [2, 2, 4, 4], [0, 4, 2, 6], [4, 0, 6, 2]],
        save_path=None,
    )
    assert np.sum(canvas - _output) < 1.0e-8
    del canvas  # skipcq


def test_functional_segmentor(remote_sample: Callable, tmp_path: Path) -> None:
    """Functional test for segmentor."""
    save_dir = tmp_path / "dump"
    # # convert to pathlib Path to prevent wsireader complaint
    resolution = 2.0
    mini_wsi_svs = Path(remote_sample("wsi4_1k_1k_svs"))
    reader = WSIReader.open(mini_wsi_svs)
    thumb = reader.slide_thumbnail(resolution=resolution, units="baseline")
    mini_wsi_jpg = f"{tmp_path}/mini_svs.jpg"
    imwrite(mini_wsi_jpg, thumb)
    mini_wsi_msk = f"{tmp_path}/mini_mask.jpg"
    imwrite(mini_wsi_msk, (thumb > 0).astype(np.uint8))

    # preemptive clean up
    shutil.rmtree("output", ignore_errors=True)  # default output dir test
    model = _CNNTo1()
    semantic_segmentor = SemanticSegmentor(batch_size=BATCH_SIZE, model=model)
    # fake injection to trigger Segmentor to create parallel
    # post-processing workers because baseline Semantic Segmentor does not support
    # post-processing out of the box. It only contains condition to create it
    # for any subclass
    semantic_segmentor.num_postproc_workers = 1

    # should still run because we skip exception
    semantic_segmentor.predict(
        [mini_wsi_jpg],
        mode="tile",
        on_gpu=ON_GPU,
        patch_input_shape=(512, 512),
        resolution=resolution,
        units="mpp",
        crash_on_exception=False,
    )

    shutil.rmtree("output", ignore_errors=True)  # default output dir test
    semantic_segmentor.predict(
        [mini_wsi_jpg],
        mode="tile",
        on_gpu=ON_GPU,
        patch_input_shape=(512, 512),
        resolution=1 / resolution,
        units="baseline",
        crash_on_exception=True,
    )
    shutil.rmtree("output", ignore_errors=True)  # default output dir test

    # * check exception bypass in the log
    # there should be no exception, but how to check the log?
    semantic_segmentor.predict(
        [mini_wsi_jpg],
        mode="tile",
        on_gpu=ON_GPU,
        patch_input_shape=(512, 512),
        patch_output_shape=(512, 512),
        stride_shape=(512, 512),
        resolution=1 / resolution,
        units="baseline",
        crash_on_exception=False,
    )
    shutil.rmtree("output", ignore_errors=True)  # default output dir test

    # * test basic running and merging prediction
    # * should dumping all 1 in the output
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        patch_input_shape=[512, 512],
        patch_output_shape=[512, 512],
        stride_shape=[512, 512],
    )

    shutil.rmtree(save_dir, ignore_errors=True)
    file_list = [
        mini_wsi_jpg,
        mini_wsi_jpg,
    ]
    output_list = semantic_segmentor.predict(
        file_list,
        mode="tile",
        on_gpu=ON_GPU,
        ioconfig=ioconfig,
        crash_on_exception=True,
        save_dir=f"{save_dir}/raw/",
    )
    pred_1 = np.load(output_list[0][1] + ".raw.0.npy")
    pred_2 = np.load(output_list[1][1] + ".raw.0.npy")
    assert len(output_list) == 2
    assert np.sum(pred_1 - pred_2) == 0
    # due to overlapping merge and division, will not be
    # exactly 1, but should be approximately so
    assert np.sum((pred_1 - 1) > 1.0e-6) == 0
    shutil.rmtree(save_dir, ignore_errors=True)

    # * test running with mask and svs
    # * also test merging prediction at designated resolution
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": resolution}],
        output_resolutions=[{"units": "mpp", "resolution": resolution}],
        save_resolution={"units": "mpp", "resolution": resolution},
        patch_input_shape=[512, 512],
        patch_output_shape=[256, 256],
        stride_shape=[512, 512],
    )
    shutil.rmtree(save_dir, ignore_errors=True)
    output_list = semantic_segmentor.predict(
        [mini_wsi_svs],
        masks=[mini_wsi_msk],
        mode="wsi",
        on_gpu=ON_GPU,
        ioconfig=ioconfig,
        crash_on_exception=True,
        save_dir=f"{save_dir}/raw/",
    )
    reader = WSIReader.open(mini_wsi_svs)
    expected_shape = reader.slide_dimensions(**ioconfig.save_resolution)
    expected_shape = np.array(expected_shape)[::-1]  # to YX
    pred_1 = np.load(output_list[0][1] + ".raw.0.npy")
    saved_shape = np.array(pred_1.shape[:2])
    assert np.sum(expected_shape - saved_shape) == 0
    assert np.sum((pred_1 - 1) > 1.0e-6) == 0
    shutil.rmtree(save_dir, ignore_errors=True)

    # check normal run with auto get mask
    semantic_segmentor = SemanticSegmentor(
        batch_size=BATCH_SIZE,
        model=model,
        auto_generate_mask=True,
    )
    _ = semantic_segmentor.predict(
        [mini_wsi_svs],
        masks=[mini_wsi_msk],
        mode="wsi",
        on_gpu=ON_GPU,
        ioconfig=ioconfig,
        crash_on_exception=True,
        save_dir=f"{save_dir}/raw/",
    )


def test_subclass(remote_sample: Callable, tmp_path: Path) -> None:
    """Create subclass and test parallel processing setup."""
    save_dir = Path(tmp_path)
    mini_wsi_jpg = Path(remote_sample("wsi2_4k_4k_jpg"))

    model = _CNNTo1()

    class XSegmentor(SemanticSegmentor):
        """Dummy class to test subclassing."""

        def __init__(self) -> None:
            super().__init__(model=model)
            self.num_postproc_worker = 2

    semantic_segmentor = XSegmentor()
    shutil.rmtree(save_dir, ignore_errors=True)  # default output dir test
    semantic_segmentor.predict(
        [mini_wsi_jpg],
        mode="tile",
        on_gpu=ON_GPU,
        patch_input_shape=(1024, 1024),
        patch_output_shape=(512, 512),
        stride_shape=(256, 256),
        resolution=1.0,
        units="baseline",
        crash_on_exception=False,
        save_dir=save_dir / "raw",
    )


# specifically designed for travis
def test_functional_pretrained(remote_sample: Callable, tmp_path: Path) -> None:
    """Test for load up pretrained and over-writing tile mode ioconfig."""
    save_dir = Path(f"{tmp_path}/output")
    mini_wsi_svs = Path(remote_sample("wsi4_512_512_svs"))
    reader = WSIReader.open(mini_wsi_svs)
    thumb = reader.slide_thumbnail(resolution=1.0, units="baseline")
    mini_wsi_jpg = f"{tmp_path}/mini_svs.jpg"
    imwrite(mini_wsi_jpg, thumb)

    semantic_segmentor = SemanticSegmentor(
        batch_size=BATCH_SIZE,
        pretrained_model="fcn-tissue_mask",
    )

    shutil.rmtree(save_dir, ignore_errors=True)
    semantic_segmentor.predict(
        [mini_wsi_svs],
        mode="wsi",
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=f"{save_dir}/raw/",
    )

    shutil.rmtree(save_dir, ignore_errors=True)

    # mainly to test prediction on tile
    semantic_segmentor.predict(
        [mini_wsi_jpg],
        mode="tile",
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=f"{save_dir}/raw/",
    )

    assert save_dir.joinpath("raw/0.raw.0.npy").exists()
    assert save_dir.joinpath("raw/file_map.dat").exists()


@pytest.mark.skipif(
    toolbox_env.running_on_ci() or not ON_GPU,
    reason="Local test on machine with GPU.",
)
def test_behavior_tissue_mask_local(remote_sample: Callable, tmp_path: Path) -> None:
    """Contain test for behavior of the segmentor and pretrained models."""
    save_dir = tmp_path
    wsi_with_artifacts = Path(remote_sample("wsi3_20k_20k_svs"))
    mini_wsi_jpg = Path(remote_sample("wsi2_4k_4k_jpg"))

    semantic_segmentor = SemanticSegmentor(
        batch_size=BATCH_SIZE,
        pretrained_model="fcn-tissue_mask",
    )
    shutil.rmtree(save_dir, ignore_errors=True)
    semantic_segmentor.predict(
        [wsi_with_artifacts],
        mode="wsi",
        on_gpu=True,
        crash_on_exception=True,
        save_dir=save_dir / "raw",
    )
    # load up the raw prediction and perform precision check
    _cache_pred = imread(Path(remote_sample("wsi3_20k_20k_pred")))
    _test_pred = np.load(str(save_dir / "raw" / "0.raw.0.npy"))
    _test_pred = (_test_pred[..., 1] > 0.75) * 255
    # divide 255 to binarize
    assert np.mean(_cache_pred[..., 0] == _test_pred) > 0.99

    shutil.rmtree(save_dir, ignore_errors=True)
    # mainly to test prediction on tile
    semantic_segmentor.predict(
        [mini_wsi_jpg],
        mode="tile",
        on_gpu=True,
        crash_on_exception=True,
        save_dir=f"{save_dir}/raw/",
    )


@pytest.mark.skipif(
    toolbox_env.running_on_ci() or not ON_GPU,
    reason="Local test on machine with GPU.",
)
def test_behavior_bcss_local(remote_sample: Callable, tmp_path: Path) -> None:
    """Contain test for behavior of the segmentor and pretrained models."""
    save_dir = tmp_path

    wsi_breast = Path(remote_sample("wsi4_4k_4k_svs"))
    semantic_segmentor = SemanticSegmentor(
        num_loader_workers=4,
        batch_size=BATCH_SIZE,
        pretrained_model="fcn_resnet50_unet-bcss",
    )
    semantic_segmentor.predict(
        [wsi_breast],
        mode="wsi",
        on_gpu=True,
        crash_on_exception=True,
        save_dir=save_dir / "raw",
    )
    # load up the raw prediction and perform precision check
    _cache_pred = np.load(Path(remote_sample("wsi4_4k_4k_pred")))
    _test_pred = np.load(f"{save_dir}/raw/0.raw.0.npy")
    _test_pred = np.argmax(_test_pred, axis=-1)
    assert np.mean(np.abs(_cache_pred - _test_pred)) < 1.0e-2


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_cli_semantic_segment_out_exists_error(
    remote_sample: Callable,
    tmp_path: Path,
) -> None:
    """Test for semantic segmentation if output path exists."""
    mini_wsi_svs = Path(remote_sample("svs-1-small"))
    sample_wsi_msk = remote_sample("small_svs_tissue_mask")
    sample_wsi_msk = np.load(sample_wsi_msk).astype(np.uint8)
    imwrite(f"{tmp_path}/small_svs_tissue_mask.jpg", sample_wsi_msk)
    sample_wsi_msk = f"{tmp_path}/small_svs_tissue_mask.jpg"
    runner = CliRunner()
    semantic_segment_result = runner.invoke(
        cli.main,
        [
            "semantic-segment",
            "--img-input",
            str(mini_wsi_svs),
            "--mode",
            "wsi",
            "--masks",
            str(sample_wsi_msk),
            "--output-path",
            tmp_path,
        ],
    )

    assert semantic_segment_result.output == ""
    assert semantic_segment_result.exit_code == 1
    assert isinstance(semantic_segment_result.exception, FileExistsError)


def test_cli_semantic_segmentation_ioconfig(
    remote_sample: Callable,
    tmp_path: Path,
) -> None:
    """Test for semantic segmentation single file custom ioconfig."""
    mini_wsi_svs = Path(remote_sample("svs-1-small"))
    sample_wsi_msk = remote_sample("small_svs_tissue_mask")
    sample_wsi_msk = np.load(sample_wsi_msk).astype(np.uint8)
    imwrite(f"{tmp_path}/small_svs_tissue_mask.jpg", sample_wsi_msk)
    sample_wsi_msk = f"{tmp_path}/small_svs_tissue_mask.jpg"

    pretrained_weights = fetch_pretrained_weights("fcn-tissue_mask")

    config = {
        "input_resolutions": [{"units": "mpp", "resolution": 2.0}],
        "output_resolutions": [{"units": "mpp", "resolution": 2.0}],
        "patch_input_shape": [1024, 1024],
        "patch_output_shape": [512, 512],
        "stride_shape": [256, 256],
        "save_resolution": {"units": "mpp", "resolution": 8.0},
    }
    with Path.open(tmp_path.joinpath("config.yaml"), "w") as fptr:
        yaml.dump(config, fptr)

    runner = CliRunner()

    semantic_segment_result = runner.invoke(
        cli.main,
        [
            "semantic-segment",
            "--img-input",
            str(mini_wsi_svs),
            "--pretrained-weights",
            str(pretrained_weights),
            "--mode",
            "wsi",
            "--masks",
            str(sample_wsi_msk),
            "--output-path",
            tmp_path.joinpath("output"),
            "--yaml-config-path",
            tmp_path.joinpath("config.yaml"),
        ],
    )

    assert semantic_segment_result.exit_code == 0
    assert tmp_path.joinpath("output/0.raw.0.npy").exists()
    assert tmp_path.joinpath("output/file_map.dat").exists()
    assert tmp_path.joinpath("output/results.json").exists()


def test_cli_semantic_segmentation_multi_file(
    remote_sample: Callable,
    tmp_path: Path,
) -> None:
    """Test for models CLI multiple file with mask."""
    mini_wsi_svs = Path(remote_sample("svs-1-small"))
    sample_wsi_msk = remote_sample("small_svs_tissue_mask")
    sample_wsi_msk = np.load(sample_wsi_msk).astype(np.uint8)
    imwrite(f"{tmp_path}/small_svs_tissue_mask.jpg", sample_wsi_msk)
    sample_wsi_msk = tmp_path / "small_svs_tissue_mask.jpg"

    # Make multiple copies for test
    dir_path = tmp_path / "new_copies"
    dir_path.mkdir()

    dir_path_masks = tmp_path / "new_copies_masks"
    dir_path_masks.mkdir()

    try:
        dir_path.joinpath("1_" + mini_wsi_svs.name).symlink_to(mini_wsi_svs)
        dir_path.joinpath("2_" + mini_wsi_svs.name).symlink_to(mini_wsi_svs)
    except OSError:
        shutil.copy(mini_wsi_svs, dir_path.joinpath("1_" + mini_wsi_svs.name))
        shutil.copy(mini_wsi_svs, dir_path.joinpath("2_" + mini_wsi_svs.name))

    try:
        dir_path_masks.joinpath("1_" + sample_wsi_msk.name).symlink_to(sample_wsi_msk)
        dir_path_masks.joinpath("2_" + sample_wsi_msk.name).symlink_to(sample_wsi_msk)
    except OSError:
        shutil.copy(sample_wsi_msk, dir_path_masks.joinpath("1_" + sample_wsi_msk.name))
        shutil.copy(sample_wsi_msk, dir_path_masks.joinpath("2_" + sample_wsi_msk.name))

    tmp_path = tmp_path / "output"

    runner = CliRunner()
    semantic_segment_result = runner.invoke(
        cli.main,
        [
            "semantic-segment",
            "--img-input",
            str(dir_path),
            "--mode",
            "wsi",
            "--masks",
            str(dir_path_masks),
            "--output-path",
            str(tmp_path),
        ],
    )

    assert semantic_segment_result.exit_code == 0
    assert tmp_path.joinpath("0.raw.0.npy").exists()
    assert tmp_path.joinpath("1.raw.0.npy").exists()
    assert tmp_path.joinpath("file_map.dat").exists()
    assert tmp_path.joinpath("results.json").exists()

    # load up the raw prediction and perform precision check
    _cache_pred = imread(Path(remote_sample("small_svs_tissue_mask")))
    _test_pred = np.load(str(tmp_path.joinpath("0.raw.0.npy")))
    _test_pred = (_test_pred[..., 1] > 0.50) * 255

    assert np.mean(np.abs(_cache_pred - _test_pred) / 255) < 1e-3
