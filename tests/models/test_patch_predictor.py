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
"""Tests for Patch Predictor."""

import os
import pathlib
import shutil

import cv2
import numpy as np
import pytest
import torch
from click.testing import CliRunner

from tiatoolbox import cli, rcParam
from tiatoolbox.models.architecture.vanilla import CNNModel
from tiatoolbox.models.controller.patch_predictor import (
    CNNPatchPredictor,
    IOPatchPredictorConfig,
)
from tiatoolbox.models.dataset import (
    PatchDataset,
    PatchDatasetABC,
    WSIPatchDataset,
    predefined_preproc_func,
)
from tiatoolbox.utils.misc import download_data, imread, imwrite
from tiatoolbox.wsicore.wsireader import get_wsireader

ON_GPU = True


def _rm_dir(path):
    """Helper func to remove directory."""
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


# -------------------------------------------------------------------------------------
# Dataloader
# -------------------------------------------------------------------------------------


def test_patchdatasetpath_imgs(sample_patch1, sample_patch2):
    """Test for patch dataset with a list of file paths as input."""
    size = (224, 224, 3)

    dataset = PatchDataset([pathlib.Path(sample_patch1), pathlib.Path(sample_patch2)])

    for _, sample_data in enumerate(dataset):
        sampled_img_shape = sample_data["image"].shape
        assert sampled_img_shape[0] == size[0]
        assert sampled_img_shape[1] == size[1]
        assert sampled_img_shape[2] == size[2]


def test_PatchDatasetlist_imgs(tmp_path):
    """Test for patch dataset with a list of images as input."""
    save_dir_path = tmp_path

    size = (5, 5, 3)
    img = np.random.randint(0, 255, size=size)
    list_imgs = [img, img, img]
    dataset = PatchDataset(list_imgs)

    dataset.preproc_func = lambda x: x

    for _, sample_data in enumerate(dataset):
        sampled_img_shape = sample_data["image"].shape
        assert (
            sampled_img_shape[0] == size[0]
            and sampled_img_shape[1] == size[1]
            and sampled_img_shape[2] == size[2]
        )

    # test for changing to another preproc
    dataset.preproc_func = lambda x: x - 10
    item = dataset[0]
    assert np.sum(item["image"] - (list_imgs[0] - 10)) == 0

    # * test for loading npy
    # remove previously generated data
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    os.makedirs(save_dir_path)
    np.save(
        os.path.join(save_dir_path, "sample2.npy"), np.random.randint(0, 255, (4, 4, 3))
    )
    imgs = [
        os.path.join(save_dir_path, "sample2.npy"),
    ]
    _ = PatchDataset(imgs)
    assert imgs[0] is not None
    # test for path object
    imgs = [
        pathlib.Path(os.path.join(save_dir_path, "sample2.npy")),
    ]
    _ = PatchDataset(imgs)
    _rm_dir(save_dir_path)


def test_PatchDatasetarray_imgs():
    """Test for patch dataset with a numpy array of a list of images."""
    size = (5, 5, 3)
    img = np.random.randint(0, 255, size=size)
    list_imgs = [img, img, img]
    labels = [1, 2, 3]
    array_imgs = np.array(list_imgs)

    # test different setter for label
    dataset = PatchDataset(array_imgs, labels=labels)
    an_item = dataset[2]
    assert an_item["label"] == 3
    dataset = PatchDataset(array_imgs, labels=None)
    an_item = dataset[2]
    assert "label" not in an_item

    dataset = PatchDataset(array_imgs)
    for _, sample_data in enumerate(dataset):
        sampled_img_shape = sample_data["image"].shape
        assert (
            sampled_img_shape[0] == size[0]
            and sampled_img_shape[1] == size[1]
            and sampled_img_shape[2] == size[2]
        )


def test_PatchDataset_crash(tmp_path):
    """Test to make sure patch dataset crashes with incorrect input."""
    # all below examples below should fail when input to PatchDataset
    save_dir_path = tmp_path

    # not supported input type
    imgs = {"a": np.random.randint(0, 255, (4, 4, 4))}
    with pytest.raises(
        ValueError, match=r".*Input must be either a list/array of images.*"
    ):
        _ = PatchDataset(imgs)

    # ndarray of mixed dtype
    imgs = np.array([np.random.randint(0, 255, (4, 5, 3)), "Should crash"])
    with pytest.raises(ValueError, match="Provided input array is non-numerical."):
        _ = PatchDataset(imgs)

    # ndarray(s) of NHW images
    imgs = np.random.randint(0, 255, (4, 4, 4))
    with pytest.raises(ValueError, match=r".*array of images of the form NHWC.*"):
        _ = PatchDataset(imgs)

    # list of ndarray(s) with different sizes
    imgs = [
        np.random.randint(0, 255, (4, 4, 3)),
        np.random.randint(0, 255, (4, 5, 3)),
    ]
    with pytest.raises(ValueError, match="Images must have the same dimensions."):
        _ = PatchDataset(imgs)

    # list of ndarray(s) with HW and HWC mixed up
    imgs = [
        np.random.randint(0, 255, (4, 4, 3)),
        np.random.randint(0, 255, (4, 4)),
    ]
    with pytest.raises(
        ValueError, match="Each sample must be an array of the form HWC."
    ):
        _ = PatchDataset(imgs)

    # list of mixed dtype
    imgs = [np.random.randint(0, 255, (4, 4, 3)), "you_should_crash_here", 123, 456]
    with pytest.raises(
        ValueError,
        match="Input must be either a list/array of images or a list of "
        "valid image paths.",
    ):
        _ = PatchDataset(imgs)

    # list of mixed dtype
    imgs = ["you_should_crash_here", 123, 456]
    with pytest.raises(
        ValueError,
        match="Input must be either a list/array of images or a list of "
        "valid image paths.",
    ):
        _ = PatchDataset(imgs)

    # list not exist paths
    with pytest.raises(
        ValueError,
        match=r".*valid image paths.*",
    ):
        _ = PatchDataset(["img.npy"])

    # ** test different extension parser
    # save dummy data to temporary location
    # remove prev generated data
    _rm_dir(save_dir_path)
    os.makedirs(save_dir_path)
    torch.save({"a": "a"}, os.path.join(save_dir_path, "sample1.tar"))
    np.save(
        os.path.join(save_dir_path, "sample2.npy"), np.random.randint(0, 255, (4, 4, 3))
    )

    imgs = [
        os.path.join(save_dir_path, "sample1.tar"),
        os.path.join(save_dir_path, "sample2.npy"),
    ]
    with pytest.raises(
        ValueError,
        match=r"Can not load data of .*",
    ):
        _ = PatchDataset(imgs)
    _rm_dir(rcParam["TIATOOLBOX_HOME"])

    # preproc func for not defined dataset
    with pytest.raises(
        ValueError,
        match=r".* preprocessing .* does not exist.",
    ):
        predefined_preproc_func("secret-dataset")


def test_WSIPatchDataset(sample_wsi_dict):
    """A test for creation and bare output."""
    # convert to pathlib Path to prevent wsireader complaint
    _mini_wsi_svs = pathlib.Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    _mini_wsi_jpg = pathlib.Path(sample_wsi_dict["wsi2_4k_4k_jpg"])
    _mini_wsi_msk = pathlib.Path(sample_wsi_dict["wsi2_4k_4k_msk"])

    def reuse_init(img_path=_mini_wsi_svs, **kwargs):
        """Testing function."""
        return WSIPatchDataset(img_path=_mini_wsi_svs, **kwargs)

    def reuse_init_wsi(**kwargs):
        """Testing function."""
        return reuse_init(mode="wsi", **kwargs)

    # test for ABC validate
    with pytest.raises(
        ValueError, match=r".*inputs should be a list of patch coordinates.*"
    ):
        # intentionally create to check error
        # skipcq
        class Proto(PatchDatasetABC):
            def __init__(self):
                super().__init__()
                self.inputs = "CRASH"
                self._check_input_integrity("wsi")

            # skipcq
            def __getitem__(self, idx):
                pass

        # intentionally create to check error
        Proto()  # skipcq

    # invalid path input
    with pytest.raises(ValueError, match=r".*`img_path` must be a valid file path.*"):
        WSIPatchDataset(
            img_path="aaaa",
            mode="wsi",
            patch_size=[512, 512],
            stride_size=[256, 256],
            auto_get_mask=False,
        )

    # invalid mask path input
    with pytest.raises(ValueError, match=r".*`mask_path` must be a valid file path.*"):
        WSIPatchDataset(
            img_path=_mini_wsi_svs,
            mask_path="aaaa",
            mode="wsi",
            patch_size=[512, 512],
            stride_size=[256, 256],
            resolution=1.0,
            units="mpp",
            auto_get_mask=False,
        )

    # invalid mode
    with pytest.raises(ValueError):
        reuse_init(mode="X")

    # invalid patch
    with pytest.raises(ValueError):
        reuse_init()
    with pytest.raises(ValueError):
        reuse_init_wsi(patch_size=[512, 512, 512])
    with pytest.raises(ValueError):
        reuse_init_wsi(patch_size=[512, "a"])
    with pytest.raises(ValueError):
        reuse_init_wsi(patch_size=512)
    # invalid stride
    with pytest.raises(ValueError):
        reuse_init_wsi(patch_size=[512, 512], stride_size=[512, "a"])
    with pytest.raises(ValueError):
        reuse_init_wsi(patch_size=[512, 512], stride_size=[512, 512, 512])
    # negative
    with pytest.raises(ValueError):
        reuse_init_wsi(patch_size=[512, -512], stride_size=[512, 512])
    with pytest.raises(ValueError):
        reuse_init_wsi(patch_size=[512, 512], stride_size=[512, -512])

    # * for wsi
    # dummy test for analysing the output
    # stride and patch size should be as expected
    patch_size = [512, 512]
    stride_size = [256, 256]
    ds = reuse_init_wsi(
        patch_size=patch_size,
        stride_size=stride_size,
        resolution=1.0,
        units="mpp",
        auto_get_mask=False,
    )
    reader = get_wsireader(_mini_wsi_svs)
    # tiling top to bottom, left to right
    ds_roi = ds[2]["image"]
    step_idx = 2  # manually calibrate
    start = (step_idx * stride_size[1], 0)
    end = (start[0] + patch_size[0], start[1] + patch_size[1])
    rd_roi = reader.read_bounds(
        start + end, resolution=1.0, units="mpp", coord_space="resolution"
    )
    correlation = np.corrcoef(
        cv2.cvtColor(ds_roi, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(rd_roi, cv2.COLOR_RGB2GRAY).flatten(),
    )
    assert ds_roi.shape[0] == rd_roi.shape[0]
    assert ds_roi.shape[1] == rd_roi.shape[1]
    assert np.min(correlation) > 0.9, correlation

    # test creation with auto mask gen and input mask
    ds = reuse_init_wsi(
        patch_size=patch_size,
        stride_size=stride_size,
        resolution=1.0,
        units="mpp",
        auto_get_mask=True,
    )
    assert len(ds) > 0
    ds = WSIPatchDataset(
        img_path=_mini_wsi_svs,
        mask_path=_mini_wsi_msk,
        mode="wsi",
        patch_size=[512, 512],
        stride_size=[256, 256],
        auto_get_mask=False,
        resolution=1.0,
        units="mpp",
    )
    negative_mask = imread(_mini_wsi_msk)
    negative_mask = np.zeros_like(negative_mask)
    imwrite("negative_mask.png", negative_mask)
    with pytest.raises(ValueError, match=r".*No coordinate remain after tiling.*"):
        ds = WSIPatchDataset(
            img_path=_mini_wsi_svs,
            mask_path="negative_mask.png",
            mode="wsi",
            patch_size=[512, 512],
            stride_size=[256, 256],
            auto_get_mask=False,
            resolution=1.0,
            units="mpp",
        )
    shutil.rmtree("negative_mask.png", ignore_errors=True)

    # * for tile
    reader = get_wsireader(_mini_wsi_jpg)
    tile_ds = WSIPatchDataset(
        img_path=_mini_wsi_jpg,
        mode="tile",
        patch_size=patch_size,
        stride_size=stride_size,
        auto_get_mask=False,
    )
    step_idx = 3  # manually calibrate
    start = (step_idx * stride_size[1], 0)
    end = (start[0] + patch_size[0], start[1] + patch_size[1])
    roi2 = reader.read_bounds(
        start + end, resolution=1.0, units="baseline", coord_space="resolution"
    )
    roi1 = tile_ds[3]["image"]  # match with step_idx
    correlation = np.corrcoef(
        cv2.cvtColor(roi1, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(roi2, cv2.COLOR_RGB2GRAY).flatten(),
    )
    assert roi1.shape[0] == roi2.shape[0]
    assert roi1.shape[1] == roi2.shape[1]
    assert np.min(correlation) > 0.9, correlation


def test_PatchDataset_abc():
    """Test for ABC methods."""
    # test missing definition for abstract
    with pytest.raises(TypeError):

        # intentionally create to check error
        # skipcq
        class Proto(PatchDatasetABC):
            # skipcq
            def __init__(self):
                super().__init__()

        # crash due to not define __getitem__
        Proto()  # skipcq

    # skipcq
    class Proto(PatchDatasetABC):
        # skipcq
        def __init__(self):
            super().__init__()

        # skipcq
        def __getitem__(self, idx):
            pass

    ds = Proto()  # skipcq

    # test setter and getter
    assert ds.preproc_func(1) == 1
    ds.preproc_func = lambda x: x - 1
    assert ds.preproc_func(1) == 0
    assert ds.preproc(1) == 1, "Must be unchanged!"
    ds.preproc_func = None
    assert ds.preproc_func(2) == 2

    # test assign uncallable to preproc_func/postproc_func
    with pytest.raises(ValueError, match=r".*callable*"):
        ds.preproc_func = 1


# -------------------------------------------------------------------------------------
# Dataloader
# -------------------------------------------------------------------------------------


def test_IOPatchPredictorConfig():
    """Test for IOConfig."""
    # test for creating
    cfg = IOPatchPredictorConfig(
        patch_size=[224, 224],
        stride_size=[224, 224],
        input_resolutions=[{"resolution": 0.5, "units": "mpp"}],
        output_resolutions=[{"resolution": 0.5, "units": "mpp"}],
        # test adding random kwarg and they should be accessible as kwargs
        crop_from_source=True,
    )
    assert cfg.crop_from_source


# -------------------------------------------------------------------------------------
# Controller
# -------------------------------------------------------------------------------------


def test_predictor_crash():
    """Test for crash when making predictor."""
    # without providing any model
    with pytest.raises(ValueError, match=r"Must provide.*"):
        CNNPatchPredictor()

    # provide wrong unknown pretrained model
    with pytest.raises(ValueError, match=r"Pretrained .* does not exist"):
        CNNPatchPredictor(pretrained_model="secret_model-kather100k")

    # provide wrong model of unknown type, deprecated later with type hint
    with pytest.raises(ValueError, match=r".*must be a string.*"):
        CNNPatchPredictor(pretrained_model=123)

    # test predict crash
    predictor = CNNPatchPredictor(pretrained_model="resnet18-kather100k", batch_size=32)

    with pytest.raises(ValueError, match=r".*not a valid mode.*"):
        predictor.predict("aaa", mode="random")
    # remove previously generated data
    if os.path.exists("output"):
        _rm_dir("output")
    with pytest.raises(ValueError, match=r".*must be a list of file paths.*"):
        predictor.predict("aaa", mode="wsi")
    # remove previously generated data
    _rm_dir("output")
    with pytest.raises(ValueError, match=r".*masks.*!=.*imgs.*"):
        predictor.predict([1, 2, 3], masks=[1, 2], mode="wsi")
    with pytest.raises(ValueError, match=r".*labels.*!=.*imgs.*"):
        predictor.predict([1, 2, 3], labels=[1, 2], mode="patch")
    # remove previously generated data
    _rm_dir("output")


def test_patch_predictor_api(sample_patch1, sample_patch2, tmp_path):
    """Helper function to get the model output using API 1."""
    save_dir_path = tmp_path

    # convert to pathlib Path to prevent reader complaint
    inputs = [pathlib.Path(sample_patch1), pathlib.Path(sample_patch2)]
    predictor = CNNPatchPredictor(pretrained_model="resnet18-kather100k", batch_size=1)
    # don't run test on GPU
    output = predictor.predict(
        inputs,
        on_gpu=ON_GPU,
    )
    assert sorted(output.keys()) == ["predictions"]
    assert len(output["predictions"]) == 2

    output = predictor.predict(
        inputs,
        labels=[1, "a"],
        return_labels=True,
        on_gpu=ON_GPU,
    )
    assert sorted(output.keys()) == sorted(["labels", "predictions"])
    assert len(output["predictions"]) == len(output["labels"])
    assert output["labels"] == [1, "a"]

    output = predictor.predict(
        inputs,
        return_probabilities=True,
        on_gpu=ON_GPU,
    )
    assert sorted(output.keys()) == sorted(["predictions", "probabilities"])
    assert len(output["predictions"]) == len(output["probabilities"])

    output = predictor.predict(
        inputs,
        return_probabilities=True,
        labels=[1, "a"],
        return_labels=True,
        on_gpu=ON_GPU,
    )
    assert sorted(output.keys()) == sorted(["labels", "predictions", "probabilities"])
    assert len(output["predictions"]) == len(output["labels"])
    assert len(output["predictions"]) == len(output["probabilities"])

    # test saving output, should have no effect
    output = predictor.predict(
        inputs,
        on_gpu=ON_GPU,
        save_dir="special_dir_not_exist",
    )
    assert not os.path.isdir("special_dir_not_exist")

    # test loading user weight
    pretrained_weights_url = (
        "https://tiatoolbox.dcs.warwick.ac.uk/models/pc/resnet18-kather100k.pth"
    )

    # remove prev generated data
    _rm_dir(save_dir_path)
    os.makedirs(save_dir_path)
    pretrained_weights = os.path.join(
        rcParam["TIATOOLBOX_HOME"],
        "tmp_pretrained_weigths",
        "resnet18-kather100k.pth",
    )
    download_data(pretrained_weights_url, pretrained_weights)

    predictor = CNNPatchPredictor(
        pretrained_model="resnet18-kather100k",
        pretrained_weights=pretrained_weights,
        batch_size=1,
    )

    # --- test different using user model
    model = CNNModel(backbone="resnet18", num_classes=9)
    # test prediction
    predictor = CNNPatchPredictor(model=model, batch_size=1, verbose=False)
    output = predictor.predict(
        inputs,
        return_probabilities=True,
        labels=[1, "a"],
        return_labels=True,
        on_gpu=ON_GPU,
    )
    assert sorted(output.keys()) == sorted(["labels", "predictions", "probabilities"])
    assert len(output["predictions"]) == len(output["labels"])
    assert len(output["predictions"]) == len(output["probabilities"])


def test_wsi_predictor_api(sample_wsi_dict, tmp_path):
    """Test normal run of wsi predictor."""
    save_dir_path = tmp_path

    # convert to pathlib Path to prevent wsireader complaint
    _mini_wsi_svs = pathlib.Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    _mini_wsi_jpg = pathlib.Path(sample_wsi_dict["wsi2_4k_4k_jpg"])
    _mini_wsi_msk = pathlib.Path(sample_wsi_dict["wsi2_4k_4k_msk"])

    patch_size = np.array([224, 224])
    predictor = CNNPatchPredictor(pretrained_model="resnet18-kather100k", batch_size=32)

    # wrapper to make this more clean
    kwargs = dict(
        return_probabilities=True,
        return_labels=True,
        on_gpu=ON_GPU,
        patch_size=patch_size,
        stride_size=patch_size,
        resolution=1.0,
        units="baseline",
    )
    # ! add this test back once the read at `baseline` is fixed
    # sanity check, both output should be the same with same resolution read args
    wsi_output = predictor.predict(
        [_mini_wsi_svs],
        masks=[_mini_wsi_msk],
        mode="wsi",
        **kwargs,
    )

    tile_output = predictor.predict(
        [_mini_wsi_jpg],
        masks=[_mini_wsi_msk],
        mode="tile",
        **kwargs,
    )

    wpred = np.array(wsi_output[0]["predictions"])
    tpred = np.array(tile_output[0]["predictions"])
    diff = tpred == wpred
    accuracy = np.sum(diff) / np.size(wpred)
    assert accuracy > 0.9, np.nonzero(~diff)

    # remove previously generated data
    save_dir = f"{save_dir_path}/model_wsi_output"
    _rm_dir(save_dir)

    kwargs = dict(
        return_probabilities=True,
        return_labels=True,
        on_gpu=ON_GPU,
        patch_size=patch_size,
        stride_size=patch_size,
        resolution=0.5,
        save_dir=save_dir,
        merge_predictions=True,  # to test the api coverage
        units="mpp",
    )

    import copy

    _kwargs = copy.deepcopy(kwargs)
    _kwargs["merge_predictions"] = False
    # test reading of multiple whole-slide images
    output = predictor.predict(
        [_mini_wsi_svs, _mini_wsi_svs],
        masks=[_mini_wsi_msk, _mini_wsi_msk],
        mode="wsi",
        **_kwargs,
    )
    for output_info in output.values():
        assert os.path.exists(output_info["raw"])
        assert "merged" not in output_info
    _rm_dir(_kwargs["save_dir"])

    # coverage test
    _kwargs = copy.deepcopy(kwargs)
    _kwargs["merge_predictions"] = True
    # test reading of multiple whole-slide images
    predictor.predict(
        [_mini_wsi_svs, _mini_wsi_svs],
        masks=[_mini_wsi_msk, _mini_wsi_msk],
        mode="wsi",
        **_kwargs,
    )
    with pytest.raises(FileExistsError):
        _kwargs = copy.deepcopy(kwargs)
        predictor.predict(
            [_mini_wsi_svs, _mini_wsi_svs],
            masks=[_mini_wsi_msk, _mini_wsi_msk],
            mode="wsi",
            **_kwargs,
        )
    # remove previously generated data
    _rm_dir(_kwargs["save_dir"])
    _rm_dir("output")

    # test reading of multiple whole-slide images
    _kwargs = copy.deepcopy(kwargs)
    _kwargs["save_dir"] = None  # default coverage
    _kwargs["return_probabilities"] = False
    output = predictor.predict(
        [_mini_wsi_svs, _mini_wsi_svs],
        masks=[_mini_wsi_msk, _mini_wsi_msk],
        mode="wsi",
        **_kwargs,
    )
    assert os.path.exists("output")
    for output_info in output.values():
        assert os.path.exists(output_info["raw"])
        assert "merged" in output_info and os.path.exists(output_info["merged"])

    # remove previously generated data
    _rm_dir("output")


def test_wsi_predictor_merge_predictions(sample_wsi_dict):
    """Test normal run of wsi predictor with merge predictions option."""
    # convert to pathlib Path to prevent reader complaint
    _mini_wsi_svs = pathlib.Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    _mini_wsi_jpg = pathlib.Path(sample_wsi_dict["wsi2_4k_4k_jpg"])
    _mini_wsi_msk = pathlib.Path(sample_wsi_dict["wsi2_4k_4k_msk"])

    # blind test
    # pseudo output dict from model with 2 patches
    output = {
        "resolution": 1.0,
        "units": "baseline",
        "probabilities": [[0.45, 0.55], [0.90, 0.10]],
        "predictions": [1, 0],
        "coordinates": [[0, 0, 2, 2], [2, 2, 4, 4]],
    }
    merged = CNNPatchPredictor.merge_predictions(
        np.zeros([4, 4]), output, resolution=1.0, units="baseline"
    )
    _merged = np.array([[2, 2, 0, 0], [2, 2, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
    assert np.sum(merged - _merged) == 0

    # integration test
    predictor = CNNPatchPredictor(pretrained_model="resnet18-kather100k", batch_size=1)

    kwargs = dict(
        return_probabilities=True,
        return_labels=True,
        on_gpu=ON_GPU,
        patch_size=np.array([224, 224]),
        stride_size=np.array([224, 224]),
        resolution=1.0,
        units="baseline",
        merge_predictions=True,
    )
    # sanity check, both output should be the same with same resolution read args
    wsi_output = predictor.predict(
        [_mini_wsi_svs],
        masks=[_mini_wsi_msk],
        mode="wsi",
        **kwargs,
    )

    # mockup to change the preproc func and
    # force to use the default in merge function
    # still should have the same results
    kwargs["merge_predictions"] = False
    tile_output = predictor.predict(
        [_mini_wsi_jpg],
        masks=[_mini_wsi_msk],
        mode="tile",
        **kwargs,
    )
    merged_tile_output = predictor.merge_predictions(
        _mini_wsi_jpg,
        tile_output[0],
        resolution=kwargs["resolution"],
        units=kwargs["units"],
    )
    tile_output.append(merged_tile_output)

    # first make sure nothing breaks with predictions
    wpred = np.array(wsi_output[0]["predictions"])
    tpred = np.array(tile_output[0]["predictions"])
    diff = tpred == wpred
    accuracy = np.sum(diff) / np.size(wpred)
    assert accuracy > 0.9, np.nonzero(~diff)

    merged_wsi = wsi_output[1]
    merged_tile = tile_output[1]
    # ensure shape of merged predictions of tile and wsi input are the same
    assert merged_wsi.shape == merged_tile.shape
    # ensure consistent predictions between tile and wsi mode
    diff = merged_tile == merged_wsi
    accuracy = np.sum(diff) / np.size(merged_wsi)
    assert accuracy > 0.9, np.nonzero(~diff)


def _test_predictor_output(
    inputs,
    pretrained_model,
    probabilities_check=None,
    predictions_check=None,
    on_gpu=ON_GPU,
):
    """Test the predictions of multiple models included in tiatoolbox."""
    predictor = CNNPatchPredictor(
        pretrained_model=pretrained_model, batch_size=32, verbose=False
    )
    # don't run test on GPU
    output = predictor.predict(
        inputs,
        return_probabilities=True,
        return_labels=False,
        on_gpu=on_gpu,
    )
    predictions = output["predictions"]
    probabilities = output["probabilities"]
    for idx, probabilities_ in enumerate(probabilities):
        probabilities_max = max(probabilities_)
        assert (
            np.abs(probabilities_max - probabilities_check[idx]) <= 1e-6
            and predictions[idx] == predictions_check[idx]
        ), (
            pretrained_model,
            probabilities_max,
            probabilities_check[idx],
            predictions[idx],
            predictions_check[idx],
        )


def test_patch_predictor_output(sample_patch1, sample_patch2):
    """Test the output of patch prediction models."""
    inputs = [pathlib.Path(sample_patch1), pathlib.Path(sample_patch2)]
    pretrained_info = {
        "alexnet-kather100k": [1.0, 0.9999735355377197],
        "resnet18-kather100k": [1.0, 0.9999911785125732],
        "resnet34-kather100k": [1.0, 0.9979840517044067],
        "resnet50-kather100k": [1.0, 0.9999986886978149],
        "resnet101-kather100k": [1.0, 0.9999932050704956],
        "resnext50_32x4d-kather100k": [1.0, 0.9910059571266174],
        "resnext101_32x8d-kather100k": [1.0, 0.9999971389770508],
        "wide_resnet50_2-kather100k": [1.0, 0.9953408241271973],
        "wide_resnet101_2-kather100k": [1.0, 0.9999831914901733],
        "densenet121-kather100k": [1.0, 1.0],
        "densenet161-kather100k": [1.0, 0.9999959468841553],
        "densenet169-kather100k": [1.0, 0.9999934434890747],
        "densenet201-kather100k": [1.0, 0.9999983310699463],
        "mobilenet_v2-kather100k": [0.9999998807907104, 0.9999126195907593],
        "mobilenet_v3_large-kather100k": [0.9999996423721313, 0.9999878406524658],
        "mobilenet_v3_small-kather100k": [0.9999998807907104, 0.9999997615814209],
        "googlenet-kather100k": [1.0, 0.9999639987945557],
    }
    for pretrained_model, expected_prob in pretrained_info.items():
        _test_predictor_output(
            inputs,
            pretrained_model,
            probabilities_check=expected_prob,
            predictions_check=[6, 3],
            on_gpu=ON_GPU,
        )


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_models_file_not_found(sample_svs, tmp_path):
    """Test for models CLI file not found error."""
    runner = CliRunner()
    model_file_not_found_result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--img_input",
            str(sample_svs)[:-1],
            "--file_types",
            '"*.ndpi, *.svs"',
            "--output_path",
            tmp_path,
        ],
    )

    assert model_file_not_found_result.output == ""
    assert model_file_not_found_result.exit_code == 1
    assert isinstance(model_file_not_found_result.exception, FileNotFoundError)


def test_command_line_models_incorrect_mode(sample_svs, tmp_path):
    """Test for models CLI mode not in wsi, tile."""
    runner = CliRunner()
    mode_not_in_wsi_tile_result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--img_input",
            str(sample_svs),
            "--file_types",
            '"*.ndpi, *.svs"',
            "--mode",
            '"patch"',
            "--output_path",
            tmp_path,
        ],
    )

    assert mode_not_in_wsi_tile_result.output == ""
    assert mode_not_in_wsi_tile_result.exit_code == 1
    assert isinstance(mode_not_in_wsi_tile_result.exception, ValueError)


def test_cli_model_single_file(sample_svs, tmp_path):
    """Test for models CLI single file."""
    runner = CliRunner()
    models_wsi_result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--img_input",
            str(sample_svs),
            "--mode",
            "wsi",
            "--output_path",
            tmp_path,
        ],
    )

    assert models_wsi_result.exit_code == 0
    assert tmp_path.joinpath("0.merged.npy").exists()
    assert tmp_path.joinpath("0.raw.json").exists()
    assert tmp_path.joinpath("results.json").exists()


def test_cli_model_single_file_mask(sample_wsi_dict, tmp_path):
    """Test for models CLI single file with mask."""
    _mini_wsi_svs = pathlib.Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    _mini_wsi_msk = pathlib.Path(sample_wsi_dict["wsi2_4k_4k_msk"])
    runner = CliRunner()
    models_tiles_result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--img_input",
            str(_mini_wsi_svs),
            "--mode",
            "wsi",
            "--masks",
            str(_mini_wsi_msk),
            "--output_path",
            tmp_path,
        ],
    )

    assert models_tiles_result.exit_code == 0
    assert tmp_path.joinpath("0.merged.npy").exists()
    assert tmp_path.joinpath("0.raw.json").exists()
    assert tmp_path.joinpath("results.json").exists()


def test_cli_model_multiple_file_mask(sample_wsi_dict, tmp_path):
    """Test for models CLI multiple file with mask."""
    _mini_wsi_svs = sample_wsi_dict["wsi2_4k_4k_svs"]
    _mini_wsi_msk = sample_wsi_dict["wsi2_4k_4k_msk"]

    # Make multiple copies for test
    dir_path = tmp_path.joinpath("new_copies")
    dir_path.mkdir()

    dir_path_masks = tmp_path.joinpath("new_copies_masks")
    dir_path_masks.mkdir()

    try:
        dir_path.joinpath("1_" + _mini_wsi_svs.name).symlink_to(_mini_wsi_svs)
        dir_path.joinpath("2_" + _mini_wsi_svs.name).symlink_to(_mini_wsi_svs)
        dir_path.joinpath("3_" + _mini_wsi_svs.name).symlink_to(_mini_wsi_svs)
    except OSError:
        shutil.copy(_mini_wsi_svs, dir_path.joinpath("1_" + _mini_wsi_svs.name))
        shutil.copy(_mini_wsi_svs, dir_path.joinpath("2_" + _mini_wsi_svs.name))
        shutil.copy(_mini_wsi_svs, dir_path.joinpath("3_" + _mini_wsi_svs.name))

    try:
        dir_path_masks.joinpath("1_" + _mini_wsi_msk.name).symlink_to(_mini_wsi_msk)
        dir_path_masks.joinpath("2_" + _mini_wsi_msk.name).symlink_to(_mini_wsi_msk)
        dir_path_masks.joinpath("3_" + _mini_wsi_msk.name).symlink_to(_mini_wsi_msk)
    except OSError:
        shutil.copy(_mini_wsi_msk, dir_path_masks.joinpath("1_" + _mini_wsi_msk.name))
        shutil.copy(_mini_wsi_msk, dir_path_masks.joinpath("2_" + _mini_wsi_msk.name))
        shutil.copy(_mini_wsi_msk, dir_path_masks.joinpath("3_" + _mini_wsi_msk.name))

    tmp_path = tmp_path.joinpath("output")

    runner = CliRunner()
    models_tiles_result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--img_input",
            str(dir_path),
            "--mode",
            "wsi",
            "--masks",
            str(dir_path_masks),
            "--output_path",
            str(tmp_path),
        ],
    )

    assert models_tiles_result.exit_code == 0
    assert tmp_path.joinpath("0.merged.npy").exists()
    assert tmp_path.joinpath("0.raw.json").exists()
    assert tmp_path.joinpath("1.merged.npy").exists()
    assert tmp_path.joinpath("1.raw.json").exists()
    assert tmp_path.joinpath("2.merged.npy").exists()
    assert tmp_path.joinpath("2.raw.json").exists()
    assert tmp_path.joinpath("results.json").exists()
