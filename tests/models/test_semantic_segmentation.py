# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****
"""Tests for Semantic Segmentor."""

import copy
import os
import pathlib
import shutil

import numpy as np
import pytest
import torch
import torch.multiprocessing as torch_mp
import torch.nn as nn
import torch.nn.functional as F

from tiatoolbox.models.abc import ModelABC
from tiatoolbox.models.architecture.utils import crop_op
from tiatoolbox.models.controller.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
    WSIStreamDataset,
)
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore.wsireader import get_wsireader

ON_GPU = False
# ----------------------------------------------------


def _rm_dir(path):
    """Helper func to remove directory."""
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


class _CNNTo1(ModelABC):
    """Contains a convolution.

    Simple model to test functionality, this contains a single
    convolution layer which has weight=0 and bias=1.

    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1)
        self.conv.weight.data.fill_(0)
        self.conv.bias.data.fill_(1)

    def forward(self, img):
        """Define how to use layer."""
        output = self.conv(img)
        return output

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
            logit_list = crop_op(logit_list, hw // 2)
            logit_list = logit_list.permute(0, 2, 3, 1)  # to NHWC
            prob_list = F.relu(logit_list)

        prob_list = prob_list.cpu().numpy()
        return [prob_list]


# -------------------------------------------------------------------------------------
# IOConfig
# -------------------------------------------------------------------------------------


def test_segmentor_ioconfig():
    """Test for IOConfig."""
    default_config = dict(
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
    # error when uniform resolution units are not uniform
    with pytest.raises(ValueError, match=r".*Invalid resolution units.*"):
        xconfig = copy.deepcopy(default_config)
        xconfig["input_resolutions"] = [
            {"units": "mpp", "resolution": 0.25},
            {"units": "power", "resolution": 0.50},
        ]
        ioconfig = IOSegmentorConfig(**xconfig)
    # error when uniform resolution units are not supported
    with pytest.raises(ValueError, match=r".*Invalid resolution units.*"):
        xconfig = copy.deepcopy(default_config)
        xconfig["input_resolutions"] = [
            {"units": "alpha", "resolution": 0.25},
            {"units": "alpha", "resolution": 0.50},
        ]
        ioconfig = IOSegmentorConfig(**xconfig)

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
            {"units": "power", "resolution": 0.25},
            {"units": "power", "resolution": 0.50},
        ],
        output_resolutions=[
            {"units": "power", "resolution": 0.25},
            {"units": "power", "resolution": 0.50},
        ],
        patch_input_shape=[2048, 2048],
        patch_output_shape=[1024, 1024],
        stride_shape=[512, 512],
    )
    assert ioconfig.highest_input_resolution == {"units": "power", "resolution": 0.50}
    ioconfig = ioconfig.to_baseline()
    assert ioconfig.input_resolutions[0]["resolution"] == 0.5
    assert ioconfig.input_resolutions[1]["resolution"] == 1.0


# -------------------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------------------


def test_functional_wsistreamdataset(remote_sample):
    """Functional test for WSIStreamDataset."""
    _mini_wsi_svs = pathlib.Path(remote_sample("wsi2_4k_4k_svs"))

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

    sds = WSIStreamDataset(ioconfig, [_mini_wsi_svs], mp_shared_space)
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
            ]
        )
    )
    mp_shared_space.patch_outputs = torch.from_numpy(
        np.array(
            [
                [0, 0, 256, 256],
                [256, 256, 512, 512],
            ]
        )
    )
    # test read
    for _, sample in enumerate(sds):
        patch_data, _ = sample
        (patch_resolution1, patch_resolution2, patch_resolution3) = patch_data
        assert np.round(patch_resolution1.shape[0] / patch_resolution2.shape[0]) == 2
        assert np.round(patch_resolution1.shape[0] / patch_resolution3.shape[0]) == 3


# -------------------------------------------------------------------------------------
# Controller
# -------------------------------------------------------------------------------------


def test_crash_segmentor(remote_sample):
    """Functional crash tests for segmentor."""
    # # convert to pathlib Path to prevent wsireader complaint
    _mini_wsi_svs = pathlib.Path(remote_sample("wsi2_4k_4k_svs"))
    _mini_wsi_jpg = pathlib.Path(remote_sample("wsi2_4k_4k_jpg"))
    _mini_wsi_msk = pathlib.Path(remote_sample("wsi2_4k_4k_msk"))

    model = _CNNTo1()
    runner = SemanticSegmentor(batch_size=1, model=model)
    # fake injection to trigger Segmentor to create parallel
    # post processing workers because baseline Semantic Segmentor does not support
    # post processing out of the box. It only contains condition to create it
    # for any subclass
    runner.num_postproc_workers = 1

    # * test basic crash
    _rm_dir("output")  # default output dir test
    with pytest.raises(ValueError, match=r".*`mask_reader`.*"):
        runner.filter_coordinates(_mini_wsi_msk, np.array(["a", "b", "c"]))
    with pytest.raises(ValueError, match=r".*ndarray.*integer.*"):
        runner.filter_coordinates(get_wsireader(_mini_wsi_msk), np.array([1.0, 2.0]))
    runner.get_reader(_mini_wsi_svs, None, "wsi", True)
    with pytest.raises(ValueError, match=r".*must be a valid file path.*"):
        runner.get_reader(_mini_wsi_msk, "not_exist", "wsi", True)

    _rm_dir("output")  # default output dir test
    with pytest.raises(ValueError, match=r".*provide.*"):
        SemanticSegmentor()
    with pytest.raises(ValueError, match=r".*valid mode.*"):
        runner.predict([], mode="abc")
    with pytest.raises(ValueError, match=r".*`tile` only use .*baseline.*"):
        runner.predict(
            [_mini_wsi_jpg],
            mode="tile",
            on_gpu=ON_GPU,
            patch_input_shape=[2048, 2048],
            resolution=1.0,
            units="mpp",
            crash_on_exception=True,
        )

    with pytest.raises(ValueError, match=r".*already exists.*"):
        runner.predict([], mode="tile", patch_input_shape=[2048, 2048])
    _rm_dir("output")  # default output dir test

    # * test not providing any ioconfig info when not using pretrained model
    with pytest.raises(ValueError, match=r".*provide either `ioconfig`.*"):
        runner.predict(
            [_mini_wsi_jpg],
            mode="tile",
            on_gpu=ON_GPU,
            crash_on_exception=True,
        )
    _rm_dir("output")  # default output dir test


def test_functional_segmentor_merging(tmp_path):
    """Functional test for assmebling output."""
    save_dir = pathlib.Path(tmp_path)

    model = _CNNTo1()
    runner = SemanticSegmentor(batch_size=1, model=model)

    _rm_dir(save_dir)
    os.mkdir(save_dir)
    # predictions with HW
    _output = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
        ]
    )
    canvas = runner.merge_prediction(
        [4, 4],
        [np.full((2, 2), 1), np.full((2, 2), 2)],
        [[0, 0, 2, 2], [2, 2, 4, 4]],
        save_path=f"{save_dir}/raw.py",
        cache_count_path=f"{save_dir}/count.py",
        free_prediction=False,
    )
    assert np.sum(canvas - _output) < 1.0e-8
    # a second rerun to test overlapping count,
    # should still maintain same result
    canvas = runner.merge_prediction(
        [4, 4],
        [np.full((2, 2), 1), np.full((2, 2), 2)],
        [[0, 0, 2, 2], [2, 2, 4, 4]],
        save_path=f"{save_dir}/raw.py",
        cache_count_path=f"{save_dir}/count.py",
        free_prediction=False,
    )
    assert np.sum(canvas - _output) < 1.0e-8
    # else will leave hanging file pointer
    # and hence cant remove its folder later
    del canvas  # skipcq

    # * predictions with HWC
    _rm_dir(save_dir)
    os.mkdir(save_dir)
    canvas = runner.merge_prediction(
        [4, 4],
        [np.full((2, 2, 1), 1), np.full((2, 2, 1), 2)],
        [[0, 0, 2, 2], [2, 2, 4, 4]],
        save_path=f"{save_dir}/raw.py",
        cache_count_path=f"{save_dir}/count.py",
        free_prediction=False,
    )
    del canvas  # skipcq

    # * test crashing when switch to image having larger
    # * shape but still provide old links
    runner.merge_prediction(
        [8, 8],
        [np.full((2, 2, 1), 1), np.full((2, 2, 1), 2)],
        [[0, 0, 2, 2], [2, 2, 4, 4]],
        save_path=f"{save_dir}/raw.1.py",
        cache_count_path=f"{save_dir}/count.1.py",
        free_prediction=False,
    )
    with pytest.raises(ValueError, match=r".*`cache_count_path` does not match.*"):
        runner.merge_prediction(
            [4, 4],
            [np.full((2, 2, 1), 1), np.full((2, 2, 1), 2)],
            [[0, 0, 2, 2], [2, 2, 4, 4]],
            save_path=f"{save_dir}/raw.py",
            cache_count_path=f"{save_dir}/count.1.py",
            free_prediction=False,
        )
    with pytest.raises(ValueError, match=r".*`save_path` does not match.*"):
        runner.merge_prediction(
            [8, 8],
            [np.full((2, 2, 1), 1), np.full((2, 2, 1), 2)],
            [[0, 0, 2, 2], [2, 2, 4, 4]],
            save_path=f"{save_dir}/raw.1.py",
            cache_count_path=f"{save_dir}/count.py",
            free_prediction=False,
        )

    _rm_dir(save_dir)
    os.mkdir(save_dir)

    # * with out of bound location
    canvas = runner.merge_prediction(
        [4, 4],
        [
            np.full((2, 2), 1),
            np.full((2, 2), 2),
            np.full((2, 2), 3),
            np.full((2, 2), 4),
        ],
        [[0, 0, 2, 2], [2, 2, 4, 4], [0, 4, 2, 6], [4, 0, 6, 2]],
        save_path=None,
        free_prediction=False,
    )
    assert np.sum(canvas - _output) < 1.0e-8
    del canvas  # skipcq
    _rm_dir(save_dir)
    os.mkdir(save_dir)


def test_functional_segmentor(remote_sample, tmp_path):
    """Functional test for segmentor."""
    save_dir = pathlib.Path(tmp_path)
    # # convert to pathlib Path to prevent wsireader complaint
    _mini_wsi_svs = pathlib.Path(remote_sample("wsi2_4k_4k_svs"))
    _mini_wsi_jpg = pathlib.Path(remote_sample("wsi2_4k_4k_jpg"))
    _mini_wsi_msk = pathlib.Path(remote_sample("wsi2_4k_4k_msk"))

    # pre-emptive clean up
    _rm_dir("output")  # default output dir test
    model = _CNNTo1()
    runner = SemanticSegmentor(batch_size=1, model=model)
    # fake injection to trigger Segmentor to create parallel
    # post processing workers because baseline Semantic Segmentor does not support
    # post processing out of the box. It only contains condition to create it
    # for any subclass
    runner.num_postproc_workers = 1

    # should still run because we skip exception
    runner.predict(
        [_mini_wsi_jpg],
        mode="tile",
        on_gpu=ON_GPU,
        patch_input_shape=[2048, 2048],
        resolution=1.0,
        units="mpp",
        crash_on_exception=False,
    )
    _rm_dir("output")  # default output dir test

    # * check exception bypass in the log
    # there should be no exception, but how to check the log?
    runner.predict(
        [_mini_wsi_jpg],
        mode="tile",
        on_gpu=ON_GPU,
        patch_input_shape=[2048, 2048],
        patch_output_shape=[1024, 1024],
        stride_shape=[512, 512],
        resolution=1.0,
        units="baseline",
        crash_on_exception=False,
    )
    _rm_dir("output")  # default output dir test

    # * test basic running and merging prediction
    # * should dumping all 1 in the output
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        patch_input_shape=[2048, 2048],
        patch_output_shape=[1024, 1024],
        stride_shape=[512, 512],
    )

    _rm_dir(save_dir)
    file_list = [
        _mini_wsi_jpg,
        _mini_wsi_jpg,
    ]
    output_list = runner.predict(
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
    _rm_dir(save_dir)

    # * test running with mask and svs
    # * also test merging prediction at designated resolution
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "baseline", "resolution": 1.0}],
        output_resolutions=[{"units": "baseline", "resolution": 1.0}],
        save_resolution={"units": "baseline", "resolution": 0.25},
        patch_input_shape=[2048, 2048],
        patch_output_shape=[1024, 1024],
        stride_shape=[512, 512],
    )
    _rm_dir(save_dir)
    output_list = runner.predict(
        [_mini_wsi_svs],
        masks=[_mini_wsi_msk],
        mode="wsi",
        on_gpu=ON_GPU,
        ioconfig=ioconfig,
        crash_on_exception=True,
        save_dir=f"{save_dir}/raw/",
    )
    reader = get_wsireader(_mini_wsi_svs)
    expected_shape = reader.slide_dimensions(**ioconfig.save_resolution)
    expected_shape = np.array(expected_shape)[::-1]  # to YX
    pred_1 = np.load(output_list[0][1] + ".raw.0.npy")
    saved_shape = np.array(pred_1.shape[:2])
    assert np.sum(expected_shape - saved_shape) == 0
    assert np.sum((pred_1 - 1) > 1.0e-6) == 0
    _rm_dir(save_dir)

    # check normal run with auto get mask
    runner = SemanticSegmentor(batch_size=1, model=model, auto_generate_mask=True)
    output_list = runner.predict(
        [_mini_wsi_svs],
        masks=[_mini_wsi_msk],
        mode="wsi",
        on_gpu=ON_GPU,
        ioconfig=ioconfig,
        crash_on_exception=True,
        save_dir=f"{save_dir}/raw/",
    )
    _rm_dir(save_dir)


def test_subclass(remote_sample, tmp_path):
    """Create subclass and test parallel processing setup."""
    save_dir = pathlib.Path(tmp_path)
    _mini_wsi_jpg = pathlib.Path(remote_sample("wsi2_4k_4k_jpg"))

    model = _CNNTo1()

    class XSegmentor(SemanticSegmentor):
        """Dummy class to test subclassing."""

        def __init__(self):
            super().__init__(model=model)
            self.num_postproc_worker = 2

    runner = XSegmentor()
    _rm_dir(save_dir)  # default output dir test
    runner.predict(
        [_mini_wsi_jpg],
        mode="tile",
        on_gpu=ON_GPU,
        patch_input_shape=[2048, 2048],
        patch_output_shape=[1024, 1024],
        stride_shape=[512, 512],
        resolution=1.0,
        units="baseline",
        crash_on_exception=False,
        save_dir=f"{save_dir}/raw/",
    )


def test_behavior_tissue_mask(remote_sample, tmp_path):
    """Contain test for behavior of the segmentor and pretrained models."""
    save_dir = pathlib.Path(tmp_path)

    wsi_with_artifacts = pathlib.Path(remote_sample("wsi3_20k_20k_svs"))
    runner = SemanticSegmentor(batch_size=1, pretrained_model="fcn-tissue_mask")
    _rm_dir(save_dir)
    runner.predict(
        [wsi_with_artifacts],
        mode="wsi",
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=f"{save_dir}/raw/",
    )
    # load up the raw prediction and perform precision check
    _cache_pred = imread(pathlib.Path(remote_sample("wsi3_20k_20k_pred")))
    _test_pred = np.load(f"{save_dir}/raw/0.raw.0.npy")
    _test_pred = (_test_pred[..., 1] > 0.75) * 255
    # divide 255 to binarize
    assert np.mean(np.abs(_cache_pred[..., 0] - _test_pred) / 255) < 1.0e-3
    _rm_dir(save_dir)


@pytest.mark.skip(reason="Local manual test, not applicable for travis.")
def test_behavior_bcss(remote_sample, tmp_path):
    """Contain test for behavior of the segmentor and pretrained models."""
    save_dir = pathlib.Path(tmp_path)

    _rm_dir(save_dir)
    # wsi_breast = pathlib.Path(remote_sample("wsi4_4k_4k_svs"))
    wsi_breast = pathlib.Path(remote_sample)
    runner = SemanticSegmentor(
        num_loader_workers=4, batch_size=16, pretrained_model="fcn_resnet50_unet-bcss"
    )
    runner.predict(
        [wsi_breast],
        mode="wsi",
        on_gpu=True,
        crash_on_exception=True,
        save_dir=f"{save_dir}/raw/",
    )
    # load up the raw prediction and perform precision check
    _cache_pred = np.load(pathlib.Path(remote_sample("wsi4_4k_4k_pred")))
    _test_pred = np.load(f"{save_dir}/raw/0.raw.0.npy")
    _test_pred = np.argmax(_test_pred, axis=-1)
    assert np.mean(np.abs(_cache_pred - _test_pred)) < 1.0e-6
    _rm_dir(save_dir)
