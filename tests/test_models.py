"""Tests for code related to model usage."""

import os
import pathlib
import shutil
import cv2
import numpy as np
import pytest
import torch

# import sys
# sys.path.append('..')
# os.environ['CUDA_VISIBLE_DEVICE'] = '3'

from click.testing import CliRunner

from tiatoolbox import cli
from tiatoolbox import rcParam
from tiatoolbox.models.backbone import get_model
from tiatoolbox.models.classification import CNNPatchModel, CNNPatchPredictor
from tiatoolbox.models.classification.abc import ModelBase
from tiatoolbox.models.dataset import (
    ABCDatasetInfo,
    KatherPatchDataset,
    PatchDataset,
    WSIPatchDataset,
    predefined_preproc_func,
)
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils.misc import download_data, unzip_data
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, get_wsireader


@pytest.mark.skip(reason="working, skip to run other test")
def test_get_coordinates():
    """Test get tile cooordinates functionality."""
    expected_output = np.array(
        [
            [0, 0, 4, 4],
            [4, 0, 8, 4],
        ]
    )
    output = PatchExtractor.get_coordinates([9, 6], [4, 4], [4, 4], within_bound=True)
    assert np.sum(expected_output - output) == 0

    expected_output = np.array(
        [
            [0, 0, 4, 4],
            [0, 4, 4, 8],
            [4, 0, 8, 4],
            [4, 4, 8, 8],
            [8, 0, 12, 4],
            [8, 4, 12, 8],
        ]
    )
    output = PatchExtractor.get_coordinates([9, 6], [4, 4], [4, 4], within_bound=False)
    assert np.sum(expected_output - output) == 0
    # test when patch shape is larger than image
    output = PatchExtractor.get_coordinates([9, 6], [9, 9], [9, 9], within_bound=False)
    assert len(output) == 1
    # test when patch shape is larger than image
    output = PatchExtractor.get_coordinates([9, 6], [9, 9], [9, 9], within_bound=True)
    assert len(output) == 0

    # test error input form
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates([9j, 6], [4, 4], [4, 4], within_bound=False)
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates([9, 6], [4, 4], [4, 4j], within_bound=False)
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates([9, 6], [4j, 4], [4, 4], within_bound=False)
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates([9, 6], [4, -1], [4, 4], within_bound=False)
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates([9, -6], [4, -1], [4, 4], within_bound=False)
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates([9, 6, 3], [4, 4], [4, 4], within_bound=False)
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates([9, 6], [4, 4, 3], [4, 4], within_bound=False)
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates([9, 6], [4, 4], [4, 4, 3], within_bound=False)
    with pytest.raises(ValueError, match=r"stride.*> 1.*"):
        PatchExtractor.get_coordinates([9, 6], [4, 4], [0, 0], within_bound=False)

    # test filtering
    bbox_list = np.array(
        [
            [0, 0, 4, 4],
            [0, 4, 4, 8],
            [4, 0, 8, 4],
            [4, 4, 8, 8],
            [8, 0, 12, 4],
            [8, 4, 12, 8],
        ]
    )
    mask = np.zeros([9, 6])
    mask[0:4, 3:8] = 1  # will flag first 2
    mask_reader = VirtualWSIReader(mask)
    flag_list = PatchExtractor.filter_coordinates(
        mask_reader, bbox_list, resolution=1.0, units="baseline"
    )
    assert np.sum(flag_list - np.array([1, 1, 0, 0, 0, 0])) == 0


@pytest.mark.skip(reason="working, skip to run other test")
def test_create_backbone():
    """Test for creating backbone."""
    backbone_list = [
        "alexnet",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        "googlenet",
        "mobilenet_v2",
        "mobilenet_v3_large",
        "mobilenet_v3_small",
    ]
    for backbone in backbone_list:
        try:
            get_model(backbone, pretrained=False)
        except ValueError:
            assert False, "Model %s failed." % backbone

    # test for model not defined
    with pytest.raises(ValueError, match=r".*not supported.*"):
        get_model("secret_model-kather100k", pretrained=False)


@pytest.mark.skip(reason="working, skip to run other test")
def test_set_root_dir():
    """Test for setting new root dir."""
    # skipcq
    from tiatoolbox import rcParam

    old_root_dir = rcParam["TIATOOLBOX_HOME"]
    test_dir_path = os.path.join(os.getcwd(), "tmp_check/")
    # clean up previous test
    if os.path.exists(test_dir_path):
        os.rmdir(test_dir_path)
    rcParam["TIATOOLBOX_HOME"] = test_dir_path
    # reimport to see if it overwrites
    # silence Deep Source because this is an intentional check
    # skipcq
    from tiatoolbox import rcParam

    os.makedirs(rcParam["TIATOOLBOX_HOME"])
    if not os.path.exists(test_dir_path):
        assert False, "`%s` != `%s`" % (rcParam["TIATOOLBOX_HOME"], test_dir_path)
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"], ignore_errors=True)
    rcParam["TIATOOLBOX_HOME"] = old_root_dir  # reassign for subsequent test


@pytest.mark.skip(reason="working, skip to run other test")
def test_DatasetInfo():
    """Test for kather patch dataset."""
    # test defining a subclass of dataset info but not defining
    # enforcing attributes - should crash
    with pytest.raises(TypeError):

        # intentionally create to check error
        # skipcq
        class Proto(ABCDatasetInfo):
            def __init__(self):
                self.a = "a"

        # intentionally create to check error
        # skipcq
        Proto()
    with pytest.raises(TypeError):

        # intentionally create to check error
        # skipcq
        class Proto(ABCDatasetInfo):
            def __init__(self):
                self.input_list = "a"

        # intentionally create to check error
        # skipcq
        Proto()
    with pytest.raises(TypeError):
        # intentionally create to check error
        # skipcq
        class Proto(ABCDatasetInfo):
            def __init__(self):
                self.input_list = "a"
                self.label_list = "a"

        # intentionally create to check error
        # skipcq
        Proto()
    with pytest.raises(TypeError):

        # intentionally create to check error
        # skipcq
        class Proto(ABCDatasetInfo):
            def __init__(self):
                self.input_list = "a"
                self.label_name = "a"

        # intentionally create to check error
        # skipcq
        Proto()
    # test kather with default init
    dataset = KatherPatchDataset()
    # kather with default data path skip download
    dataset = KatherPatchDataset()
    # pytest for not exist dir
    with pytest.raises(
        ValueError,
        match=r".*not exist.*",
    ):
        _ = KatherPatchDataset(save_dir_path="unknown_place")

    # save to temporary location
    save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "tmp_check/")
    # remove previously generated data
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    url = (
        "https://zenodo.org/record/53169/files/"
        "Kather_texture_2016_image_tiles_5000.zip"
    )
    save_zip_path = os.path.join(save_dir_path, "Kather.zip")
    download_data(url, save_zip_path)
    unzip_data(save_zip_path, save_dir_path)
    extracted_dir = os.path.join(save_dir_path, "Kather_texture_2016_image_tiles_5000/")
    dataset = KatherPatchDataset(save_dir_path=extracted_dir)
    assert dataset.input_list is not None
    assert dataset.label_list is not None
    assert dataset.label_name is not None
    assert len(dataset.input_list) == len(dataset.label_list)

    # to actually get the image, we feed it to PatchDataset
    actual_ds = PatchDataset(dataset.input_list, dataset.label_list)
    sample_patch = actual_ds[100]
    assert isinstance(sample_patch["image"], np.ndarray)
    assert sample_patch["label"] is not None

    # remove generated data
    shutil.rmtree(save_dir_path, ignore_errors=True)
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"])


@pytest.mark.skip(reason="working, skip to run other test")
def test_PatchDatasetpath_imgs(_sample_patch1, _sample_patch2):
    """Test for patch dataset with a list of file paths as input."""
    size = (224, 224, 3)

    dataset = PatchDataset([pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)])

    dataset.preproc_func = lambda x: x

    for _, sample_data in enumerate(dataset):
        sampled_img_shape = sample_data["image"].shape
        assert (
            sampled_img_shape[0] == size[0]
            and sampled_img_shape[1] == size[1]
            and sampled_img_shape[2] == size[2]
        )


@pytest.mark.skip(reason="working, skip to run other test")
def test_PatchDatasetlist_imgs():
    """Test for patch dataset with a list of images as input."""
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

    # test for loading npy
    save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "tmp_check/")
    # remove previously generated data
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    os.makedirs(save_dir_path)
    np.save(
        os.path.join(save_dir_path, "sample2.npy"), np.random.randint(0, 255, (4, 4, 3))
    )
    img_list = [
        os.path.join(save_dir_path, "sample2.npy"),
    ]
    _ = PatchDataset(img_list)
    assert img_list[0] is not None
    # test for path object
    img_list = [
        pathlib.Path(os.path.join(save_dir_path, "sample2.npy")),
    ]
    _ = PatchDataset(img_list)
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"])


@pytest.mark.skip(reason="working, skip to run other test")
def test_PatchDatasetarray_imgs():
    """Test for patch dataset with a numpy array of a list of images."""
    size = (5, 5, 3)
    img = np.random.randint(0, 255, size=size)
    list_imgs = [img, img, img]
    label_list = [1, 2, 3]
    array_imgs = np.array(list_imgs)

    # test different setter for label
    dataset = PatchDataset(array_imgs, label_list=label_list)
    an_item = dataset[2]
    assert an_item["label"] == 3
    dataset = PatchDataset(array_imgs, label_list=None)
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


@pytest.mark.skip(reason="working, skip to run other test")
def test_PatchDatasetcrash():
    """Test to make sure patch dataset crashes with incorrect input."""
    # all below examples below should fail when input to PatchDataset

    # not supported input type
    img_list = {"a": np.random.randint(0, 255, (4, 4, 4))}
    with pytest.raises(
        ValueError, match=r".*Input must be either a list/array of images.*"
    ):
        _ = PatchDataset(img_list)

    # ndarray of mixed dtype
    img_list = np.array([np.random.randint(0, 255, (4, 5, 3)), "Should crash"])
    with pytest.raises(ValueError, match="Provided input array is non-numerical."):
        _ = PatchDataset(img_list)

    # ndarrays of NHW images
    img_list = np.random.randint(0, 255, (4, 4, 4))
    with pytest.raises(ValueError, match=r".*array of images of the form NHWC.*"):
        _ = PatchDataset(img_list)

    # list of ndarrays with different sizes
    img_list = [
        np.random.randint(0, 255, (4, 4, 3)),
        np.random.randint(0, 255, (4, 5, 3)),
    ]
    with pytest.raises(ValueError, match="Images must have the same dimensions."):
        _ = PatchDataset(img_list)

    # list of ndarrays with HW and HWC mixed up
    img_list = [
        np.random.randint(0, 255, (4, 4, 3)),
        np.random.randint(0, 255, (4, 4)),
    ]
    with pytest.raises(
        ValueError, match="Each sample must be an array of the form HWC."
    ):
        _ = PatchDataset(img_list)

    # list of mixed dtype
    img_list = [np.random.randint(0, 255, (4, 4, 3)), "you_should_crash_here", 123, 456]
    with pytest.raises(
        ValueError,
        match="Input must be either a list/array of images or a list of "
        "valid image paths.",
    ):
        _ = PatchDataset(img_list)

    # list of mixed dtype
    img_list = ["you_should_crash_here", 123, 456]
    with pytest.raises(
        ValueError,
        match="Input must be either a list/array of images or a list of "
        "valid image paths.",
    ):
        _ = PatchDataset(img_list)

    # list not exist paths
    with pytest.raises(
        ValueError,
        match=r".*valid image paths.*",
    ):
        _ = PatchDataset(["img.npy"])

    # ** test different extenstion parser
    # save dummy data to temporary location
    save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "tmp_check/")
    # remove prev generated data
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    os.makedirs(save_dir_path)
    torch.save({"a": "a"}, os.path.join(save_dir_path, "sample1.tar"))
    np.save(
        os.path.join(save_dir_path, "sample2.npy"), np.random.randint(0, 255, (4, 4, 3))
    )

    img_list = [
        os.path.join(save_dir_path, "sample1.tar"),
        os.path.join(save_dir_path, "sample2.npy"),
    ]
    with pytest.raises(
        ValueError,
        match=r"Can not load data of .*",
    ):
        _ = PatchDataset(img_list)
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"])

    # preproc func for not defined dataset
    with pytest.raises(
        ValueError,
        match=r".* preprocessing .* does not exist.",
    ):
        predefined_preproc_func("secret-dataset")


@pytest.mark.skip(reason="working, skip to run other test")
def test_WSIPatchDataset(_mini_wsi1_svs, _mini_wsi1_jpg):
    """A test for creation and bare output."""
    # convert to pathlib Path to prevent wsireader complaint
    _mini_wsi1_svs = pathlib.Path(_mini_wsi1_svs)
    _mini_wsi1_jpg = pathlib.Path(_mini_wsi1_jpg)

    def reuse_init(**kwargs):
        """Testing function."""
        return WSIPatchDataset(img_path=_mini_wsi1_svs, **kwargs)

    def reuse_init_wsi(**kwargs):
        """Testing function."""
        return reuse_init(mode="wsi", **kwargs)

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

    # dummy test for analysing the output
    # stride and patch size should be as expected
    patch_size = [2048, 2048]
    stride_size = [1024, 1024]
    ds = reuse_init_wsi(
        patch_size=patch_size,
        stride_size=stride_size,
        resolution=1.0,
        units="mpp",
        auto_get_mask=False
    )
    reader = get_wsireader(_mini_wsi1_svs)
    # tiling top to bottom, left to right
    ds_roi = ds[2]["image"]
    step_idx = 2  # manually calibrate
    start = (0, step_idx * stride_size[1])
    end = (start[0] + patch_size[0], start[1] + patch_size[1])
    rd_roi = reader.read_bounds(
                        start + end,
                        resolution=1.0, units="mpp",
                        location_is_at_requested=True)
    correlation = np.corrcoef(
        cv2.cvtColor(ds_roi, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(rd_roi, cv2.COLOR_RGB2GRAY).flatten(),
    )

    assert ds_roi.shape[0] == rd_roi.shape[0]
    assert ds_roi.shape[1] == rd_roi.shape[1]
    assert np.min(correlation) > 0.9, correlation

    reader = get_wsireader(_mini_wsi1_jpg)
    tile_ds = WSIPatchDataset(
        img_path=_mini_wsi1_jpg,
        mode="tile",
        patch_size=patch_size,
        stride_size=stride_size,
        auto_get_mask=False
    )
    step_idx = 3  # manually calibrate
    start = (0, step_idx * stride_size[1])
    end = (start[0] + patch_size[0], start[1] + patch_size[1])
    roi2 = reader.read_bounds(
                start + end,
                resolution=1.0, units="baseline",
                location_is_at_requested=True)
    roi1 = tile_ds[3]["image"]  # match with step_idx
    correlation = np.corrcoef(
        cv2.cvtColor(roi1, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(roi2, cv2.COLOR_RGB2GRAY).flatten(),
    )
    assert roi1.shape[0] == roi2.shape[0]
    assert roi1.shape[1] == roi2.shape[1]
    assert np.min(correlation) > 0.9, correlation


@pytest.mark.skip(reason="working, skip to run other test")
def test_WSIPatchDataset_varying_resolution_read(_mini_wsi1_svs, _mini_wsi1_jpg):
    """Test if different resolution read is as expected."""
    _mini_wsi1_svs = pathlib.Path(_mini_wsi1_svs)
    idx = 0  # due to the auto mask generation, it may pick dumb place
    patch_size = np.array([1024, 1024])
    mpp_10 = WSIPatchDataset(
        img_path=_mini_wsi1_svs,
        mode="wsi",
        patch_size=patch_size,
        stride_size=patch_size,
        resolution=1.0,
        units="mpp",
        auto_get_mask=False
    )[idx]["image"]
    mpp_20 = WSIPatchDataset(
        img_path=_mini_wsi1_svs,
        mode="wsi",
        patch_size=(patch_size / 2).astype(np.int32),
        stride_size=(patch_size / 2).astype(np.int32),
        resolution=2.0,
        units="mpp",
        auto_get_mask=False
    )[idx]["image"]
    mpp_05 = WSIPatchDataset(
        img_path=_mini_wsi1_svs,
        mode="wsi",
        patch_size=(patch_size * 2).astype(np.int32),
        stride_size=(patch_size * 2).astype(np.int32),
        resolution=0.5,
        units="mpp",
        auto_get_mask=False
    )[idx]["image"]
    # resizing then do correlation check
    mpp_20 = cv2.resize(mpp_20, (1024, 1024))
    mpp_05 = cv2.resize(mpp_05, (1024, 1024))
    cc = np.corrcoef(
        cv2.cvtColor(mpp_05, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(mpp_10, cv2.COLOR_RGB2GRAY).flatten(),
    )
    assert np.min(cc) > 0.9, cc
    cc = np.corrcoef(
        cv2.cvtColor(mpp_20, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(mpp_10, cv2.COLOR_RGB2GRAY).flatten(),
    )
    assert np.min(cc) > 0.9, cc
    cc = np.corrcoef(
        cv2.cvtColor(mpp_20, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(mpp_05, cv2.COLOR_RGB2GRAY).flatten(),
    )
    assert np.min(cc) > 0.9, cc

    # test run time only for different resolution units
    ds = WSIPatchDataset(
        img_path=_mini_wsi1_svs,
        mode="wsi",
        patch_size=patch_size,
        stride_size=patch_size,
        resolution=10.0,
        units="power",
        auto_get_mask=False
    )[idx]["image"]
    assert ds is not None

    # unsupport resolution/unit
    with pytest.raises(ValueError):
        ds = WSIPatchDataset(
            img_path=_mini_wsi1_svs,
            mode="wsi",
            patch_size=patch_size,
            stride_size=patch_size,
            resolution=4.0,
            units="baseline",
            auto_get_mask=True
        )[idx]["image"]
        assert ds is not None
    with pytest.raises(ValueError):
        ds = WSIPatchDataset(
            img_path=_mini_wsi1_svs,
            mode="wsi",
            patch_size=patch_size,
            stride_size=patch_size,
            resolution=1,
            units="level",
            auto_get_mask=True
        )[idx]["image"]
        assert ds is not None

    # test tile metadata enforcement
    # only read at 1 resolution for tile, so resolution
    # and units should have no effect
    roi1 = WSIPatchDataset(
        img_path=_mini_wsi1_jpg,
        mode="tile",
        patch_size=patch_size,
        stride_size=patch_size,
        resolution=1,
        units="mpp",
        auto_get_mask=False
    )[idx]["image"]
    roi2 = WSIPatchDataset(
        img_path=_mini_wsi1_jpg,
        mode="tile",
        patch_size=patch_size,
        stride_size=patch_size,
        resolution=4.0,
        units="power",
        auto_get_mask=False
    )[idx]["image"]
    assert (roi1 - roi2).sum() == 0


@pytest.mark.skip(reason="working, skip to run other test")
def test_predictor_crash():
    """Test for crash when making predictor."""
    # test abc
    with pytest.raises(NotImplementedError):
        ModelBase()
    with pytest.raises(NotImplementedError):
        ModelBase.infer_batch(1, 2, 3)

    # without providing any model
    with pytest.raises(ValueError, match=r"Must provide.*"):
        CNNPatchPredictor()

    # provide wrong unknown pretrained model
    with pytest.raises(ValueError, match=r"Pretrained .* does not exist"):
        CNNPatchPredictor(pretrained_model="secret_model-kather100k")

    # provide wrong model of unknown type, deprecated later with type hint
    with pytest.raises(ValueError, match=r".*must be a string.*"):
        CNNPatchPredictor(pretrained_model=123)


@pytest.mark.skip(reason="working, skip to run other test")
def test_patch_predictor_api(_sample_patch1, _sample_patch2):
    """Helper function to get the model output using API 1."""
    # must wrap or sthg stupid happens
    input_list = [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)]
    predictor = CNNPatchPredictor(pretrained_model="resnet18-kather100k", batch_size=1)
    # don't run test on GPU
    output = predictor.predict(
        input_list,
        on_gpu=False,
    )
    assert sorted(list(output.keys())) == ["predictions"]
    assert len(output["predictions"]) == 2

    output = predictor.predict(
        input_list,
        label_list=[1, "a"],
        return_labels=True,
        on_gpu=False,
    )
    assert sorted(list(output.keys())) == sorted(["labels", "predictions"])
    assert len(output["predictions"]) == len(output["labels"])
    assert output["labels"] == [1, "a"]

    output = predictor.predict(
        input_list,
        return_probabilities=True,
        on_gpu=False,
    )
    assert sorted(list(output.keys())) == sorted(["predictions", "probabilities"])
    assert len(output["predictions"]) == len(output["probabilities"])

    output = predictor.predict(
        input_list,
        return_probabilities=True,
        label_list=[1, "a"],
        return_labels=True,
        on_gpu=False,
    )
    assert sorted(list(output.keys())) == sorted(
        ["labels", "predictions", "probabilities"]
    )
    assert len(output["predictions"]) == len(output["labels"])
    assert len(output["predictions"]) == len(output["probabilities"])

    # test saving output, should have no effect
    output = predictor.predict(
        input_list,
        on_gpu=False,
        save_dir="special_dir_not_exist",
    )
    assert not os.path.isdir("special_dir_not_exist")

    # test loading user weight
    pretrained_weight_url = (
        "https://tiatoolbox.dcs.warwick.ac.uk/models/pc/resnet18-kather100k.pth"
    )

    save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "tmp_pretrained_weigths")
    # remove prev generated data
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    os.makedirs(save_dir_path)
    pretrained_weight = os.path.join(
        rcParam["TIATOOLBOX_HOME"],
        "tmp_pretrained_weigths",
        "resnet18-kather100k.pth",
    )
    download_data(pretrained_weight_url, pretrained_weight)

    predictor = CNNPatchPredictor(
        pretrained_model="resnet18-kather100k",
        pretrained_weight=pretrained_weight,
        batch_size=1,
    )

    # test different using user model
    model = CNNPatchModel(backbone="resnet18", num_classes=9)
    # coverage setter check
    model.set_preproc_func(lambda x: x - 1)  # do this for coverage
    assert model.get_preproc_func()(1) == 0
    # coverage setter check
    model.set_preproc_func(None)  # do this for coverage
    assert model.get_preproc_func()(1) == 1
    predictor = CNNPatchPredictor(model=model, batch_size=1, verbose=False)
    output = predictor.predict(
        input_list,
        return_probabilities=True,
        label_list=[1, "a"],
        return_labels=True,
        on_gpu=False,
    )
    assert sorted(list(output.keys())) == sorted(
        ["labels", "predictions", "probabilities"]
    )
    assert len(output["predictions"]) == len(output["labels"])
    assert len(output["predictions"]) == len(output["probabilities"])


@pytest.mark.skip(reason="working, skip to run other test")
def test_wsi_predictor_api(_mini_wsi1_svs, _mini_wsi1_jpg, _mini_wsi1_msk):
    """Test normal run of wsi predictor."""
    on_gpu = False
    # convert to pathlib Path to prevent wsireader complaint
    _mini_wsi1_svs = pathlib.Path(_mini_wsi1_svs)
    _mini_wsi1_jpg = pathlib.Path(_mini_wsi1_jpg)
    _mini_wsi1_msk = pathlib.Path(_mini_wsi1_msk)

    patch_size = np.array([224, 224])
    predictor = CNNPatchPredictor(pretrained_model="resnet18-kather100k", batch_size=1)

    # ! add this test back once the read at `baseline` is fixed
    # sanity check, both output should be the same with same resolution read args
    # wsi_output = predictor.predict(
    #     [_mini_wsi1_svs],
    #     mask_list=[_mini_wsi1_msk],
    #     mode="wsi",
    #     return_probabilities=True,
    #     return_labels=True,
    #     on_gpu=False,
    #     patch_size=patch_size,
    #     stride_size=patch_size,
    #     resolution=1.0,
    #     units="baseline",
    # )[0][0]
    # tile_output = predictor.predict(
    #     [_mini_wsi1_jpg],
    #     mask_list=[_mini_wsi1_msk],
    #     mode="tile",
    #     return_probabilities=True,
    #     return_labels=True,
    #     on_gpu=False,
    #     patch_size=patch_size,
    #     stride_size=patch_size,
    #     resolution=1.0,
    #     units="baseline",
    # )[0][0]
    # wpred = np.array(wsi_output["predictions"])
    # tpred = np.array(tile_output["predictions"])
    # diff = tpred == wpred
    # accuracy = np.sum(diff) / np.size(wpred)
    # assert accuracy > 0.9, np.nonzero(~diff)

    # remove previously generated data
    save_dir = "model_wsi_output"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True)

    # test reading of multiple whole-slide images
    predictor.predict(
        [_mini_wsi1_svs, _mini_wsi1_svs, _mini_wsi1_svs],
        mask_list=[_mini_wsi1_msk, _mini_wsi1_msk, _mini_wsi1_msk],
        mode="wsi",
        return_probabilities=True,
        return_labels=True,
        on_gpu=on_gpu,
        patch_size=patch_size,
        stride_size=patch_size,
        resolution=1.0,
        units="mpp",
        save_dir=save_dir,
    )
    with pytest.raises(ValueError, match=r".*save_dir.*exist.*"):
        predictor.predict(
            [_mini_wsi1_svs, _mini_wsi1_svs, _mini_wsi1_svs],
            mask_list=[_mini_wsi1_msk, _mini_wsi1_msk, _mini_wsi1_msk],
            mode="wsi",
            return_probabilities=True,
            return_labels=True,
            on_gpu=on_gpu,
            patch_size=patch_size,
            stride_size=patch_size,
            resolution=1.0,
            units="mpp",
            save_dir=save_dir,
        )

    # remove previously generated data
    save_dir = "model_wsi_output"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True)


@pytest.mark.skip(reason="working, skip to run other test")
def test_wsi_predictor_merge_predictions(
    _mini_wsi1_svs,
    _mini_wsi1_jpg,
    _mini_wsi1_msk,
    # as replacement for multiple tile tests,
    # else will be very mem expensive and crash travis
    _patch_extr_vf_image,
):
    """Test normal run of wsi predictor with merge predictions option."""
    # convert to pathlib Path to prevent wsireader complaint
    _mini_wsi1_svs = pathlib.Path(_mini_wsi1_svs)
    _mini_wsi1_jpg = pathlib.Path(_mini_wsi1_jpg)
    _mini_wsi1_msk = pathlib.Path(_mini_wsi1_msk)
    _tile_2k_x_2k = pathlib.Path(_patch_extr_vf_image)

    # _tile_2k_x_2k = '/home/tialab-dang/workstation_storage_1/workspace/tiatoolbox/
    # tests/local_samples/TCGA-HE-7130-01Z-00-DX1.png'
    # _tile_2k_x_2k = pathlib.Path(_tile_2k_x_2k)
    on_gpu = False
    patch_size = np.array([224, 224])
    predictor = CNNPatchPredictor(pretrained_model="resnet18-kather100k", batch_size=1)

    # ! add this test back once the read at `baseline` is fixed
    # ~~~
    # sanity check, both output should be the same with same resolution read args
    # wsi_output = predictor.predict(
    #     [_mini_wsi1_svs],
    #     mask_list=[_mini_wsi1_msk],
    #     mode="wsi",
    #     return_probabilities=True,
    #     return_labels=True,
    #     on_gpu=False,
    #     patch_size=patch_size,
    #     stride_size=patch_size,
    #     resolution=1.0,
    #     units="baseline",
    #     merge_predictions=True,
    #     return_overlay=True,
    # )[0]
    # tile_output = predictor.predict(
    #     [_mini_wsi1_jpg],
    #     mask_list=[_mini_wsi1_msk],
    #     mode="tile",
    #     return_probabilities=True,
    #     return_labels=True,
    #     on_gpu=False,
    #     patch_size=patch_size,
    #     stride_size=patch_size,
    #     resolution=1.0,
    #     units="baseline",
    #     merge_predictions=True,
    #     return_overlay=True,
    # )[0]
    # # first make sure nothing breaks with predictions
    # wpred = np.array(wsi_output[0]["predictions"])
    # tpred = np.array(tile_output[0]["predictions"])
    # diff = tpred == wpred
    # accuracy = np.sum(diff) / np.size(wpred)
    # assert accuracy > 0.9, np.nonzero(~diff)

    # merged_wsi = wsi_output[1]
    # merged_tile = tile_output[1]
    # # enure shape of merged predictions of tile and wsi input are the same
    # assert merged_wsi.shape == merged_tile.shape
    # # ensure conistent predictions between tile and wsi mode
    # diff = merged_tile == merged_wsi
    # accuracy = np.sum(diff) / np.size(merged_wsi)
    # assert accuracy > 0.9, np.nonzero(~diff)

    # overlay_wsi = wsi_output[2]
    # overlay_tile = tile_output[2]
    # # ensure returned overlay can be saved as matplotlib figure
    # save_dir_path = "tmp_figure/"
    # if os.path.exists(save_dir_path):
    #     shutil.rmtree(save_dir_path, ignore_errors=True)
    # os.makedirs(save_dir_path)
    # overlay_wsi.savefig(save_dir_path + "overlay1.png")
    # overlay_tile.savefig(save_dir_path + "overlay2.png")

    # assert os.path.isfile(save_dir_path + "overlay1.png")
    # assert os.path.isfile(save_dir_path + "overlay2.png")

    # shutil.rmtree(save_dir_path, ignore_errors=True)
    # ~~~

    # remove previously generated data
    # save_dir = "model_wsi_output"
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir, ignore_errors=True)

    # # test read multiple whole-slide images
    # predictor.predict(
    #     [_mini_wsi1_svs, _mini_wsi1_svs, _mini_wsi1_svs],
    #     mask_list=[_mini_wsi1_msk, _mini_wsi1_msk, _mini_wsi1_msk],
    #     mode="wsi",
    #     return_probabilities=True,
    #     return_labels=True,
    #     on_gpu=on_gpu,
    #     patch_size=patch_size,
    #     stride_size=patch_size,
    #     resolution=1.0,
    #     units="mpp",
    #     merge_predictions=True,
    #     return_overlay=True,
    #     save_dir=save_dir,
    # )

    # print(_mini_wsi1_svs)
    # print(_mini_wsi1_svs.stem)

    # assert os.path.isfile(save_dir + "/%s/results.json" % _mini_wsi1_svs.stem)
    # assert os.path.isfile(save_dir + "/%s/prediction_map.npy" % _mini_wsi1_svs.stem)
    # assert os.path.isfile(save_dir + "/%s/overlay.png" % _mini_wsi1_svs.stem)
    # shutil.rmtree(save_dir, ignore_errors=True)

    # remove previously generated data
    save_dir = "model_tile_output"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True)

    # this is very expensive
    predictor.predict(
        [_tile_2k_x_2k, _tile_2k_x_2k, _tile_2k_x_2k],
        mode="tile",
        return_probabilities=True,
        return_labels=True,
        on_gpu=on_gpu,
        patch_size=patch_size,
        stride_size=patch_size,
        resolution=1.0,
        units="baseline",
        merge_predictions=True,
        return_overlay=True,
        save_dir=save_dir,
    )
    assert os.path.isfile(save_dir + "/%s/results.json" % _tile_2k_x_2k.stem)
    assert os.path.isfile(save_dir + "/%s/prediction_map.npy" % _tile_2k_x_2k.stem)
    assert os.path.isfile(save_dir + "/%s/overlay.png" % _tile_2k_x_2k.stem)
    shutil.rmtree(save_dir, ignore_errors=True)


def _test_predictor_output(
    input_list,
    pretrained_model,
    probabilities_check=None,
    predictions_check=None,
    on_gpu=False,
):
    """Test the predictions of multiple models included in tiatoolbox."""
    predictor = CNNPatchPredictor(
        pretrained_model=pretrained_model, batch_size=1, verbose=False
    )
    # don't run test on GPU
    output = predictor.predict(
        input_list,
        return_probabilities=True,
        return_labels=False,
        on_gpu=on_gpu,
    )
    predictions = output["predictions"]
    probabilities = output["probabilities"]
    for idx, probabilities_ in enumerate(probabilities):
        probabilities_max = max(probabilities_)
        print(predictions[idx], predictions_check[idx])
        assert (
            np.abs(probabilities_max - probabilities_check[idx]) <= 1e-8
            and predictions[idx] == predictions_check[idx]
        ), pretrained_model


@pytest.mark.skip(reason="working, skip to run other test")
def test_patch_predictor_output(_sample_patch1, _sample_patch2):
    """Test the output of patch prediction models."""
    input_list = [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)]
    pretrained_info = {
        "alexnet-kather100k": [1.0, 0.9999735355377197],
        "resnet18-Kather100k": [1.0, 0.9999911785125732],
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
            input_list,
            pretrained_model,
            probabilities_check=expected_prob,
            predictions_check=[6, 3],
        )


# ----------------------------------------------------------------------------------
# Command Line Interface
# ----------------------------------------------------------------------------------


# @pytest.mark.skip(reason="working, skip to run other test")
def test_command_line_patch_predictor_patches(_dir_sample_patches, _sample_patch1):
    """Test for the patch predictor CLI using patches as input."""
    runner = CliRunner()

    result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--pretrained_model",
            "resnet18-kather100k",
            "--img_input",
            str(pathlib.Path(_dir_sample_patches)),
            "--output_path",
            "tmp_output",
            "--batch_size",
            2,
            "--mode",
            "patch",
            "--return_probabilities",
            False,
            "--on_gpu",
            False,
        ],
        catch_exceptions=True,
    )

    shutil.rmtree("tmp_output", ignore_errors=True)
    if result.exit_code != 0:
        error_mssg = result.stdout_bytes.decode(encoding='UTF-8')
        assert False, error_mssg

    result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--pretrained_model",
            "resnet18-kather100k",
            "--img_input",
            pathlib.Path(_sample_patch1),
            "--output_path",
            "tmp_output",
            "--batch_size",
            1,
            "--mode",
            "patch",
            "--return_probabilities",
            False,
            "--on_gpu",
            False,
        ],
        catch_exceptions=True,
    )

    shutil.rmtree("tmp_output", ignore_errors=True)
    if result.exit_code != 0:
        error_mssg = result.stdout_bytes.decode(encoding='UTF-8')
        assert False, error_mssg


# @pytest.mark.skip(reason="working, skip to run other test")
def test_command_line_patch_predictor_wsi(
    _dir_sample_tile, _dir_sample_msk,
    # _mini_wsi1_jpg, _mini_wsi1_msk,
    _patch_extr_vf_image  # need to change this
):
    """Test for the patch predictor CLI using tiles/wsi as input."""
    runner = CliRunner()

    on_gpu = False  # for local debug mode, not travis
    result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--pretrained_model",
            "resnet18-kather100k",
            "--img_input",
            pathlib.Path(_dir_sample_tile),
            "--mask_input",
            pathlib.Path(_dir_sample_msk),
            "--output_path",
            "tmp_output",
            "--batch_size",
            2,
            "--mode",
            "tile",
            "--return_probabilities",
            False,
            "--on_gpu",
            on_gpu,
        ],
        catch_exceptions=True,
    )

    shutil.rmtree("tmp_output", ignore_errors=True)
    if result.exit_code != 0:
        error_mssg = result.stdout_bytes.decode(encoding='UTF-8')
        assert False, error_mssg

    result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--pretrained_model",
            "resnet18-kather100k",
            "--img_input",
            pathlib.Path(_patch_extr_vf_image),
            # "--mask_input",
            # pathlib.Path(_mini_wsi1_msk),
            "--output_path",
            "tmp_output",
            "--batch_size",
            2,
            "--mode",
            "tile",
            "--return_probabilities",
            False,
            "--on_gpu",
            on_gpu,
        ],
        catch_exceptions=True,
    )

    shutil.rmtree("tmp_output", ignore_errors=True)
    if result.exit_code != 0:
        error_mssg = result.stdout_bytes.decode(encoding='UTF-8')
        assert False, error_mssg


# @pytest.mark.skip(reason="working, skip to run other test")
def test_command_line_patch_predictor_crash(_sample_patch1):
    """Test for the patch predictor CLI."""
    # test single image not exist
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--pretrained_model",
            "resnet18-kather100k",
            "--img_input",
            "imaginary_img.tif",
            "--mode",
            "patch",
            "--on_gpu",
            False,
        ],
    )
    assert result.exit_code != 0

    # test not pretrained model
    result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--pretrained_model",
            "secret_model-kather100k",
            "--img_input",
            pathlib.Path(_sample_patch1),
            "--mode",
            "patch",
            "--on_gpu",
            False,
        ],
    )
    assert result.exit_code != 0
