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

from tiatoolbox import rcParam
from tiatoolbox.models.abc import ModelABC
from tiatoolbox.models.segmentation import (
    IOConfigSegmentor,
    SemanticSegmentor,
    WSIStreamDataset,
)
from tiatoolbox.wsicore.wsireader import get_wsireader

ON_GPU = False
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


def _crop_op(x, cropping, data_format="NCHW"):
    """Center crop image.

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`

    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


class _CNNTo1(ModelABC):
    """Contain a convolution.

    Simple model to test functionality, this contains a single
    convolution layer which has weight=0 and bias=1.
    """

    def __init__(self):
        super(_CNNTo1, self).__init__()
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
        device = "cuda" if ON_GPU else "cpu"
        ####
        model.eval()  # infer mode

        ####
        img_list = batch_data

        img_list = img_list.to(device).type(torch.float32)
        img_list = img_list.permute(0, 3, 1, 2)  # to NCHW

        hw = np.array(img_list.shape[2:])
        with torch.no_grad():  # dont compute gradient
            logit_list = model(img_list)
            logit_list = _crop_op(logit_list, hw // 2)
            logit_list = logit_list.permute(0, 2, 3, 1)  # to NHWC
            prob_list = F.relu(logit_list)

        prob_list = prob_list.cpu().numpy()
        return [prob_list]


# ----------------------------------------------------


def test_segmentor_ioconfig():
    """Test for IOConfig"""
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
        ioconfig = IOConfigSegmentor(**xconfig)
    # error when uniform resolution units are not supported
    with pytest.raises(ValueError, match=r".*Invalid resolution units.*"):
        xconfig = copy.deepcopy(default_config)
        xconfig["input_resolutions"] = [
            {"units": "alpha", "resolution": 0.25},
            {"units": "alpha", "resolution": 0.50},
        ]
        ioconfig = IOConfigSegmentor(**xconfig)

    ioconfig = IOConfigSegmentor(
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
    ioconfig.to_baseline()
    assert ioconfig.input_resolutions[0]["resolution"] == 1.0
    assert ioconfig.input_resolutions[1]["resolution"] == 0.5
    assert ioconfig.input_resolutions[2]["resolution"] == 1 / 3

    ioconfig = IOConfigSegmentor(
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
    ioconfig.to_baseline()
    assert ioconfig.input_resolutions[0]["resolution"] == 0.5
    assert ioconfig.input_resolutions[1]["resolution"] == 1.0


def test_functional_WSIStreamDataset(_sample_wsi_dict):
    """Functional test for WSIStreamDataset."""
    _mini_wsi_svs = pathlib.Path(_sample_wsi_dict["wsi2_4k_4k_svs"])

    ioconfig = IOConfigSegmentor(
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
    assert out == [1, 2, 3]

    # faking data injecttion
    mp_shared_space.wsi_idx = torch.tensor(0)  # a scalar
    mp_shared_space.patch_inputs = torch.from_numpy(
        np.array(
            [
                # skipcq
                [0, 0, 256, 256],
                [256, 256, 512, 512],
            ]
        )
    )
    mp_shared_space.patch_outputs = torch.from_numpy(
        np.array(
            [
                # skipcq
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


def test_functional_segmentor(_sample_wsi_dict):
    """Functional test for segmentor."""
    save_dir = _get_temp_folder_path()
    save_dir = pathlib.Path(save_dir)
    # # convert to pathlib Path to prevent wsireader complaint
    _mini_wsi_svs = pathlib.Path(_sample_wsi_dict["wsi2_4k_4k_svs"])
    _mini_wsi_jpg = pathlib.Path(_sample_wsi_dict["wsi2_4k_4k_jpg"])
    _mini_wsi_msk = pathlib.Path(_sample_wsi_dict["wsi2_4k_4k_msk"])

    model = _CNNTo1()
    runner = SemanticSegmentor(batch_size=4, model=model)

    # * test basic crash
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
        runner.predict([], mode="tile")
    _rm_dir("output")  # default output dir test
    # * check exception bypass in the log
    # there should be no exception, but how to check the log?
    runner.predict(
        [_mini_wsi_jpg],
        mode="tile",
        on_gpu=ON_GPU,
        patch_input_shape=[2048, 2048],
        resolution=1.0,
        units="baseline",
        crash_on_exception=False,
    )
    _rm_dir("output")  # default output dir test

    # * test basic running and merging prediction
    # * should dumping all 1 in the output
    ioconfig = IOConfigSegmentor(
        input_resolutions=[{"units": "baseline", "resolution": 2.0}],
        output_resolutions=[{"units": "baseline", "resolution": 2.0}],
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
    # * also test mergin prediction at designated resolution
    ioconfig = IOConfigSegmentor(
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
