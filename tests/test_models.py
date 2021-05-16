"""Tests for code related to model usage."""

# %%
import os
import pathlib
import shutil
import cv2
from cv2 import data

import numpy as np
import pytest
import torch
from click.testing import CliRunner

import sys
# sys.path.append('.')
# sys.path.append('..')

from sklearn import metrics
from tiatoolbox import rcParam
from tiatoolbox.models.backbone import get_model
from tiatoolbox.models.classification.abc import ModelBase
from tiatoolbox.models.classification import CNNPatchModel, CNNPatchPredictor
from tiatoolbox.models.dataset import (
    ABCDatasetInfo,
    KatherPatchDataset,
    PatchDataset,
    WSIPatchDataset,
    predefined_preproc_func,
)

from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils.misc import download_data, unzip_data, imread
from tiatoolbox import cli
from tiatoolbox.wsicore import wsireader

from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIMeta, get_wsireader


# @pytest.mark.skip(reason="working, skip to run other test")
def test_get_coordinates():
    """Test tiling coordinate getter."""
    expected_output = np.array([
        [0, 0, 4, 4],
        [4, 0, 8, 4],
    ])
    output = PatchExtractor.get_coordinates(
                [9, 6], [4, 4], [4, 4],
                within_bound=True)
    assert np.sum(expected_output - output) == 0

    expected_output = np.array([
        [0, 0, 4, 4],
        [0, 4, 4, 8],
        [4, 0, 8, 4],
        [4, 4, 8, 8],
        [8, 0, 12, 4],
        [8, 4, 12, 8],
    ])
    output = PatchExtractor.get_coordinates(
                [9, 6], [4, 4], [4, 4],
                within_bound=False)
    assert np.sum(expected_output - output) == 0
    # test patch shape larger than image
    output = PatchExtractor.get_coordinates(
                [9, 6], [9, 9], [9, 9],
                within_bound=False)
    assert len(output) == 1
    # test patch shape larger than image
    output = PatchExtractor.get_coordinates(
                [9, 6], [9, 9], [9, 9],
                within_bound=True)
    assert len(output) == 0

    # test error input form
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            [9j, 6], [4, 4], [4, 4],
            within_bound=False)
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            [9, 6], [4, 4], [4, 4j],
            within_bound=False)
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            [9, 6], [4j, 4], [4, 4],
            within_bound=False)
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            [9, 6], [4, -1], [4, 4],
            within_bound=False)
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            [9, -6], [4, -1], [4, 4],
            within_bound=False)
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            [9, 6, 3], [4, 4], [4, 4],
            within_bound=False)
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            [9, 6], [4, 4, 3], [4, 4],
            within_bound=False)
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            [9, 6], [4, 4], [4, 4, 3],
            within_bound=False)
    with pytest.raises(ValueError, match=r"stride.*> 1.*"):
        PatchExtractor.get_coordinates(
            [9, 6], [4, 4], [0, 0],
            within_bound=False)

    # * test filtering
    bbox_list = np.array([
        [0, 0, 4, 4],
        [0, 4, 4, 8],
        [4, 0, 8, 4],
        [4, 4, 8, 8],
        [8, 0, 12, 4],
        [8, 4, 12, 8],
    ])
    mask = np.zeros([9, 6])
    mask[0:4, 3:8] = 1  # will flag first 2
    mask_reader = VirtualWSIReader(mask)
    flag_list = PatchExtractor.filter_coordinates(
                    mask_reader, bbox_list,
                    resolution=1.0, units='baseline')
    assert np.sum(flag_list - np.array([1, 1, 0, 0, 0, 0])) == 0


# @pytest.mark.skip(reason="working, skip to run other test")
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
        # "googlenet",
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
        get_model("secret_model", pretrained=False)


# @pytest.mark.skip(reason="working, skip to run other test")
def test_set_root_dir():
    """Test for setting new root dir."""
    # skipcq
    from tiatoolbox import rcParam

    old_root_dir = rcParam["TIATOOLBOX_HOME"]
    test_dir_path = os.path.join(os.getcwd(), "tmp_check/")
    # clean up prev test
    if os.path.exists(test_dir_path):
        os.rmdir(test_dir_path)
    rcParam["TIATOOLBOX_HOME"] = test_dir_path
    # reimport to see if it overwrites
    # silence Deep Source because this is intentional check
    # skipcq
    from tiatoolbox import rcParam

    os.makedirs(rcParam["TIATOOLBOX_HOME"])
    if not os.path.exists(test_dir_path):
        assert False, "`%s` != `%s`" % (rcParam["TIATOOLBOX_HOME"], test_dir_path)
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"], ignore_errors=True)
    rcParam["TIATOOLBOX_HOME"] = old_root_dir  # reassign for subsequent test


# @pytest.mark.skip(reason="working, skip to run other test")
def test_DatasetInfo():  # Working
    """Test for kather patch dataset."""
    # test defining a subclas of dataset info but not defining
    # enforcing attributes, should crash
    with pytest.raises(TypeError):

        class Proto(ABCDatasetInfo):
            def __init__(self):
                self.a = "a"

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
    # remove prev generated data - just a test!
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

    # to actually get the image, we feed it to a PatchDataset
    actual_ds = PatchDataset(dataset.input_list, dataset.label_list)
    sample_patch = actual_ds[100]
    assert isinstance(sample_patch["image"], np.ndarray)
    assert sample_patch["label"] is not None

    # remove generated data - just a test!
    shutil.rmtree(save_dir_path, ignore_errors=True)
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"])


# @pytest.mark.skip(reason="working, skip to run other test")
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


# @pytest.mark.skip(reason="working, skip to run other test")
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
    # remove prev generated data - just a test!
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


# @pytest.mark.skip(reason="working, skip to run other test")
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


# @pytest.mark.skip(reason="working, skip to run other test")
def test_PatchDatasetcrash():
    """Test to make sure patch dataset crashes with incorrect input."""
    # all examples below should fail when input to PatchDataset

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
    # remove prev generated data - just a test!
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
        predefined_preproc_func("secret_dataset")


# @pytest.mark.skip(reason="working, skip to run other test")
def test_WSIPatchDataset(_mini_wsi1_svs, _mini_wsi1_jpg):
    """A test for creation and bare output."""
    # to prevent wsireader complaint
    _mini_wsi1_svs = pathlib.Path(_mini_wsi1_svs)
    _mini_wsi1_jpg = pathlib.Path(_mini_wsi1_jpg)

    def reuse_init(**kwargs):
        return WSIPatchDataset(wsi_path=_mini_wsi1_svs, **kwargs)

    def reuse_init_wsi(**kwargs):
        return reuse_init(mode="wsi", **kwargs)

    # invalid mode
    with pytest.raises(ValueError):
        reuse_init(mode="X")

    # invalid patch
    with pytest.raises(ValueError):
        reuse_init()
    with pytest.raises(ValueError):
        reuse_init_wsi(patch_shape=[512, 512, 512])
    with pytest.raises(ValueError):
        reuse_init_wsi(patch_shape=[512, "a"])
    with pytest.raises(ValueError):
        reuse_init_wsi(patch_shape=512)
    # invalid stride
    with pytest.raises(ValueError):
        reuse_init_wsi(patch_shape=[512, 512], stride_shape=[512, "a"])
    with pytest.raises(ValueError):
        reuse_init_wsi(
            patch_shape=[512, 512],
            stride_shape=[512, 512, 512])
    # negative
    with pytest.raises(ValueError):
        reuse_init_wsi(
            patch_shape=[512, -512],
            stride_shape=[512, 512])
    with pytest.raises(ValueError):
        reuse_init_wsi(
            patch_shape=[512, 512],
            stride_shape=[512, -512])

    # * dummy test for output correctness
    # * striding and patch should be as expected
    # * so we just need to do a manual retrieval and do sum check (hopefully)
    # * correct tiling or will be test in another way
    patch_shape = [4096, 4096]
    stride_shape = [2048, 2048]
    ds = reuse_init_wsi(
        patch_shape=patch_shape,
        stride_shape=stride_shape,
        resolution=1.0,
        units="baseline",
    )
    # tiling top to bottom, left to right
    ds_roi = ds[2]["image"]
    step_idx = 2  # manual calibrate
    reader = get_wsireader(_mini_wsi1_svs)
    start = (0, step_idx * stride_shape[1])
    end = (start[0] + patch_shape[0], start[1] + patch_shape[1])
    rd_roi = reader.read_bounds(start + end, resolution=1.0, units="baseline")
    correlation = np.corrcoef(
        cv2.cvtColor(ds_roi, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(rd_roi, cv2.COLOR_RGB2GRAY).flatten(),
    )

    assert ds_roi.shape[0] == rd_roi.shape[0]
    assert ds_roi.shape[1] == rd_roi.shape[1]
    assert np.min(correlation) > 0.9, correlation
    # uncomment these for internal viz check
    # import matplotlib.pyplot as plt
    # plt.subplot(1,2,1)
    # plt.imshow(ds_roi)
    # plt.subplot(1,2,2)
    # plt.imshow(rd_roi)
    # plt.savefig('dump.png')

    # ** repeated above test for tile at the same resolution as baseline
    # ** but is not pyramidal
    wsi_ds = WSIPatchDataset(
        wsi_path=_mini_wsi1_svs,
        mode="wsi",
        patch_shape=patch_shape,
        stride_shape=stride_shape,
        resolution=1.0,
        units="baseline",
    )
    tile_ds = WSIPatchDataset(
        wsi_path=_mini_wsi1_jpg,
        mode="tile",
        patch_shape=patch_shape,
        stride_shape=stride_shape,
        resolution=1.0,
        units="baseline",
    )
    assert len(tile_ds) == len(wsi_ds), "%s vs %s" % (len(tile_ds), len(wsi_ds))
    roi1 = wsi_ds[3]["image"]
    roi2 = tile_ds[3]["image"]
    correlation = np.corrcoef(
        cv2.cvtColor(roi1, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(roi2, cv2.COLOR_RGB2GRAY).flatten(),
    )
    assert roi1.shape[0] == roi2.shape[0]
    assert roi1.shape[1] == roi2.shape[1]
    assert np.min(correlation) > 0.9, correlation

    # import matplotlib.pyplot as plt
    # plt.subplot(1,2,1)
    # plt.imshow(roi1)
    # plt.subplot(1,2,2)
    # plt.imshow(roi2)
    # plt.savefig('dump.png')


# @pytest.mark.skip(reason="working, skip to run other test")
def test_WSIPatchDataset_varying_resolution_read(_mini_wsi1_svs, _mini_wsi1_jpg):
    """Test if different resolution read is as expected."""
    _mini_wsi1_svs = pathlib.Path(_mini_wsi1_svs)
    idx = 3
    patch_shape = np.array([1024, 1024])
    mpp_10 = WSIPatchDataset(
            wsi_path=_mini_wsi1_svs,
            mode='wsi',
            patch_shape=patch_shape,
            stride_shape=patch_shape,
            resolution=1.0,
            units='mpp')[idx]['image']
    mpp_20 = WSIPatchDataset(
            wsi_path=_mini_wsi1_svs,
            mode='wsi',
            patch_shape=(patch_shape / 2).astype(np.int32),
            stride_shape=(patch_shape / 2).astype(np.int32),
            resolution=2.0,
            units='mpp')[idx]['image']
    mpp_05 = WSIPatchDataset(
            wsi_path=_mini_wsi1_svs,
            mode='wsi',
            patch_shape=(patch_shape * 2).astype(np.int32),
            stride_shape=(patch_shape * 2).astype(np.int32),
            resolution=0.5,
            units='mpp')[idx]['image']
    # resizing then do correlation check
    mpp_20 = cv2.resize(mpp_20, (1024, 1024))
    mpp_05 = cv2.resize(mpp_05, (1024, 1024))
    cc = np.corrcoef(
        cv2.cvtColor(mpp_05, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(mpp_10, cv2.COLOR_RGB2GRAY).flatten()
    )
    assert np.min(cc) > 0.9, cc
    cc = np.corrcoef(
        cv2.cvtColor(mpp_20, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(mpp_10, cv2.COLOR_RGB2GRAY).flatten()
    )
    assert np.min(cc) > 0.9, cc
    cc = np.corrcoef(
        cv2.cvtColor(mpp_20, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(mpp_05, cv2.COLOR_RGB2GRAY).flatten()
    )
    assert np.min(cc) > 0.9, cc

    # test run time only for different resolution units
    WSIPatchDataset(
        wsi_path=_mini_wsi1_svs,
        mode='wsi',
        patch_shape=patch_shape,
        stride_shape=patch_shape,
        resolution=10.0,
        units='power')[idx]['image']

    WSIPatchDataset(
        wsi_path=_mini_wsi1_svs,
        mode='wsi',
        patch_shape=patch_shape,
        stride_shape=patch_shape,
        resolution=4.0,
        units='baseline')[idx]['image']

    WSIPatchDataset(
        wsi_path=_mini_wsi1_svs,
        mode='wsi',
        patch_shape=patch_shape,
        stride_shape=patch_shape,
        resolution=1,
        units='level')[idx]['image']

    # test tile metadata enforcement
    # * only read at 1 resolution for tile, so resolution
    # * and units should have no effect
    roi1 = WSIPatchDataset(
        wsi_path=_mini_wsi1_jpg,
        mode='tile',
        patch_shape=patch_shape,
        stride_shape=patch_shape,
        resolution=1,
        units='mpp')[idx]['image']
    roi2 = WSIPatchDataset(
        wsi_path=_mini_wsi1_jpg,
        mode='tile',
        patch_shape=patch_shape,
        stride_shape=patch_shape,
        resolution=4.0,
        units='power')[idx]['image']
    assert (roi1 - roi2).sum() == 0


# @pytest.mark.skip(reason="working, skip to run other test")
def test_sync_VirtualReader_read(_mini_wsi1_svs, _mini_wsi1_jpg, _mini_wsi1_msk):
    """Test synchronize read for VirtualReader"""
    # _mini_wsi1_svs = '/home/tialab-dang/local/project/tiatoolbox/tests/data/CMU-mini.svs'
    # _mini_wsi1_jpg = '/home/tialab-dang/local/project/tiatoolbox/tests/data/CMU-mini.jpg'
    # _mini_wsi1_msk = '/home/tialab-dang/local/project/tiatoolbox/tests/data/CMU-mask.png'

    _mini_wsi1_svs = pathlib.Path(_mini_wsi1_svs)
    _mini_wsi1_msk = pathlib.Path(_mini_wsi1_msk)
    _mini_wsi1_jpg = pathlib.Path(_mini_wsi1_jpg)

    wsi_reader = get_wsireader(_mini_wsi1_svs)

    msk = imread(_mini_wsi1_msk)
    msk_reader = VirtualWSIReader(msk)
    old_metadata = msk_reader.info
    msk_reader.attach_to_reader(wsi_reader.info)
    # check that attach altered vreader metadata
    assert np.any(old_metadata.mpp != msk_reader.info.mpp)

    # now check sync read by comparing the RoI with different base
    # the output should be at same resolution even if source is of different base
    bigger_msk = cv2.resize(msk, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
    bigger_msk_reader = VirtualWSIReader(bigger_msk)
    # * must set mpp metadata to not None else wont work
    # error checking first
    ref_metadata = bigger_msk_reader.info
    ref_metadata.mpp = 1.0
    ref_metadata.objective_power = None
    with pytest.raises(ValueError, match=r".*objective.*None.*"):
        msk_reader.attach_to_reader(ref_metadata)
    ref_metadata.mpp = None
    ref_metadata.objective_power = 1.0
    with pytest.raises(ValueError, match=r".*mpp.*None.*"):
        msk_reader.attach_to_reader(ref_metadata)

    # must set mpp metadata to not None else wont
    # !?! why do this doesn modify ?, but modify
    # !!! reference above seem to work? @John
    ref_metadata.mpp = 1.0
    ref_metadata.objective_power = 1.0
    msk_reader.attach_to_reader(ref_metadata)

    # ! box should be within image
    lv0_coords = np.array([0, 1000, 2000, 3000])
    # with mpp
    roi1 = bigger_msk_reader.read_bounds(
                lv0_coords,
                resolution=0.25,
                units='mpp'
            )
    scale_wrt_ref = msk_reader.info.level_downsamples[0]
    roi2 = msk_reader.read_bounds(
                lv0_coords / scale_wrt_ref,
                resolution=0.25,
                units='mpp'
            )
    cc = np.corrcoef(roi1[..., 0].flatten(), roi2[..., 0].flatten())
    assert np.min(cc) > 0.95, cc
    # with objective
    roi1 = bigger_msk_reader.read_bounds(
                lv0_coords,
                resolution=0.25,
                units='power'
            )
    scale_wrt_ref = msk_reader.info.level_downsamples[0]
    roi2 = msk_reader.read_bounds(
                lv0_coords / scale_wrt_ref,
                resolution=0.25,
                units='power'
            )
    cc = np.corrcoef(roi1[..., 0].flatten(), roi2[..., 0].flatten())
    assert np.min(cc) > 0.95, cc
    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.imshow(roi1)
    # plt.subplot(1, 2, 2)
    # plt.imshow(roi2)
    # plt.show()
    # plt.savefig('dump.png')

    # * now check attaching and read to WSIReader and varying resolution
    # need to think how to check correctness
    lv0_coords = np.array([4500, 9500, 6500, 11500])
    msk_reader.attach_to_reader(wsi_reader.info)
    msk_reader.read_bounds(
            lv0_coords / scale_wrt_ref,
            resolution=15.0,
            units='power'
        )
    msk_reader.read_bounds(
            lv0_coords / scale_wrt_ref,
            resolution=1.0,
            units='mpp'
        )
    msk_reader.read_bounds(
            lv0_coords / scale_wrt_ref,
            resolution=1.0,
            units='baseline'
        )

    patch_shape = [512, 512]
    # now check normal reading for dataset with mask
    ds = WSIPatchDataset(
        _mini_wsi1_svs,
        mode='wsi',
        mask_path=_mini_wsi1_msk,
        patch_shape=patch_shape,
        stride_shape=patch_shape,
        resolution=1.0,
        units='mpp')
    ds[10]
    ds = WSIPatchDataset(
        _mini_wsi1_svs,
        mode='wsi',
        mask_path=_mini_wsi1_msk,
        patch_shape=patch_shape,
        stride_shape=patch_shape,
        resolution=1.0,
        units='baseline')
    ds[10]
    ds = WSIPatchDataset(
        _mini_wsi1_svs,
        mode='wsi',
        mask_path=_mini_wsi1_msk,
        patch_shape=patch_shape,
        stride_shape=patch_shape,
        resolution=15.0,
        units='power')
    ds[10]

    # * now check sync read for tile ans wsi
    patch_shape = np.array([2048, 2048])
    wds = WSIPatchDataset(
        _mini_wsi1_svs,
        mask_path=_mini_wsi1_msk,
        mode="wsi",
        patch_shape=patch_shape,
        stride_shape=patch_shape,
        resolution=1.0,
        units="baseline",
    )
    tds = WSIPatchDataset(
        _mini_wsi1_jpg,
        mask_path=_mini_wsi1_msk,
        mode="tile",
        patch_shape=patch_shape,
        stride_shape=patch_shape,
        resolution=1.0,
        units="baseline",
    )
    assert len(wds) == len(tds)
    predictor = CNNPatchPredictor(
        pretrained_model='resnet18-kather100K',
        batch_size=1,
        verbose=False
    )
    # now loop over each read and ensure they look similar
    for idx in range(len(wds)):
        cc = np.corrcoef(
                cv2.cvtColor(wds[idx]['image'], cv2.COLOR_RGB2GRAY).flatten(),
                cv2.cvtColor(tds[idx]['image'], cv2.COLOR_RGB2GRAY).flatten())
        assert np.min(cc) > 0.95, (cc, idx)


# @pytest.mark.skip(reason="working, skip to run other test")
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
        CNNPatchPredictor(pretrained_model="secret_model")

    # provide wrong model of unknown type, deprecated later with type hint
    with pytest.raises(ValueError, match=r".*must be a string.*"):
        CNNPatchPredictor(pretrained_model=123)


# @pytest.mark.skip(reason="working, skip to run other test")
def test_patch_predictor_api(_sample_patch1, _sample_patch2):
    """Helper function to get the model output using API 1."""
    # must wrap or sthg stupid happens
    input_list = [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)]
    predictor = CNNPatchPredictor(pretrained_model='resnet18-kather100K', batch_size=1)
    # don't run test on GPU
    output = predictor.predict(
        input_list,
        on_gpu=False,
    )
    assert sorted(list(output.keys())) == ['predictions']
    assert len(output['predictions']) == 2

    output = predictor.predict(
        input_list,
        label_list=[1, 'a'],
        return_labels=True,
        on_gpu=False,
    )
    assert sorted(list(output.keys())) == sorted(['labels', 'predictions'])
    assert len(output['predictions']) == len(output['labels'])
    assert output['labels'] == [1, 'a']

    output = predictor.predict(
        input_list,
        return_probabilities=True,
        on_gpu=False,
    )
    assert sorted(list(output.keys())) == sorted(['predictions', 'probabilities'])
    assert len(output['predictions']) == len(output['probabilities'])

    output = predictor.predict(
        input_list,
        return_probabilities=True,
        label_list=[1, 'a'],
        return_labels=True,
        on_gpu=False,
    )
    assert sorted(list(output.keys())) == \
        sorted(['labels', 'predictions', 'probabilities'])
    assert len(output['predictions']) == len(output['labels'])
    assert len(output['predictions']) == len(output['probabilities'])

    # test saving output, should have no effect
    output = predictor.predict(
        input_list,
        on_gpu=False,
        save_dir='special_dir_not_exist',
    )
    assert not os.path.isdir('special_dir_not_exist')

    # test loading user weight
    pretrained_weight_url = (
        "https://tiatoolbox.dcs.warwick.ac.uk/models/resnet18-kather100K-pc.pth"
    )

    save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "tmp_pretrained_weigths")
    # remove prev generated data - just a test!
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    os.makedirs(save_dir_path)
    pretrained_weight = os.path.join(
        rcParam["TIATOOLBOX_HOME"], "tmp_pretrained_weigths",
        "resnet18-kather100K-pc.pth"
    )
    download_data(pretrained_weight_url, pretrained_weight)

    predictor = CNNPatchPredictor(
        pretrained_model='resnet18-kather100K',
        pretrained_weight=pretrained_weight,
        batch_size=1,
    )

    # test different using user model
    model = CNNPatchModel(backbone='resnet18', num_classes=9)
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
        label_list=[1, 'a'],
        return_labels=True,
        on_gpu=False,
    )
    assert sorted(list(output.keys())) == \
        sorted(['labels', 'predictions', 'probabilities'])
    assert len(output['predictions']) == len(output['labels'])
    assert len(output['predictions']) == len(output['probabilities'])


# @pytest.mark.skip(reason="working, skip to run other test")
def test_wsi_predictor_api(_mini_wsi1_svs, _mini_wsi1_jpg, _mini_wsi1_msk):
    """Test normal run of wsi predictor.

    This is not prediction correctness test. Correctness test need to check
    - correct patch read at varying resolution args (more about dataset test,
    such as the sync test and varying resolution tiling test).
    - expected prediction at simple patch.
    """
    # _mini_wsi1_svs = '/home/tialab-dang/local/project/tiatoolbox/tests/data/CMU-mini.svs'
    # _mini_wsi1_jpg = '/home/tialab-dang/local/project/tiatoolbox/tests/data/CMU-mini.jpg'
    # _mini_wsi1_msk = '/home/tialab-dang/local/project/tiatoolbox/tests/data/CMU-mask.png'
    # to prevent wsireader complaint
    _mini_wsi1_svs = pathlib.Path(_mini_wsi1_svs)
    _mini_wsi1_jpg = pathlib.Path(_mini_wsi1_jpg)
    _mini_wsi1_msk = pathlib.Path(_mini_wsi1_msk)

    predictor = CNNPatchPredictor(pretrained_model="resnet18-kather100K", batch_size=1)

    # * sanity check, both output should be the same with same resolution read args
    patch_shape = np.array([224, 224])
    wsi_output = predictor.predict(
        [_mini_wsi1_svs],
        mask_list=[_mini_wsi1_msk],
        mode="wsi",
        return_probabilities=True,
        return_labels=True,
        on_gpu=False,
        patch_shape=patch_shape,
        stride_shape=patch_shape,
        resolution=1.0,
        units="baseline",
    )[0]
    tile_output = predictor.predict(
        [_mini_wsi1_jpg],
        mask_list=[_mini_wsi1_msk],
        mode="tile",
        return_probabilities=True,
        return_labels=True,
        on_gpu=False,
        patch_shape=patch_shape,
        stride_shape=patch_shape,
        resolution=1.0,
        units="baseline",
    )[0]
    wpred = np.array(wsi_output["predictions"])
    tpred = np.array(tile_output["predictions"])
    diff = tpred == wpred
    accuracy = np.sum(diff) / np.size(wpred)
    # ! cant do exact test because different base seem to
    # ! mess up some patch
    assert accuracy > 0.9, np.nonzero(~diff)


def _test_predictor_correctness(
    input_list,
    pretrained_model,
    probabilities_check=None,
    predictions_check=None,
    on_gpu=False,
):
    predictor = CNNPatchPredictor(
        pretrained_model=pretrained_model,
        batch_size=1,
        verbose=False
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
        assert (
            np.abs(probabilities_max - probabilities_check[idx]) <= 1e-8
            and predictions[idx] == predictions_check[idx]
        ), pretrained_model


# @pytest.mark.skip(reason="working, skip to run other test")
def test_patch_predictor_correctness(_sample_patch1, _sample_patch2):
    input_list = [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)]
    pretrained_info = {
        'resnet18-kather100K'           : [1.0, 0.9999717473983765],
        'alexnet-kather100K'            : [1.0, 0.9998185038566589],
        'resnet50-kather100K'           : [1.0, 0.9969022870063782],
        'resnet34-kather100K'           : [1.0, 0.9991286396980286],
        'resnet101-kather100K'          : [1.0, 0.9999957084655762],
        'resnext50_32x4d-kather100K'    : [1.0, 0.9999779462814331],
        'resnext101_32x8d-kather100K'   : [1.0, 0.9999345541000366],
        'wide_resnet50_2-kather100K'    : [1.0, 0.9999997615814209],
        'wide_resnet101_2-kather100K'   : [1.0, 0.999420166015625],
        'densenet121-kather100K'        : [1.0, 0.9998136162757874],
        'densenet161-kather100K'        : [1.0, 0.9999997615814209],
        'densenet169-kather100K'        : [1.0, 0.9999773502349854],
        'densenet201-kather100K'        : [1.0, 0.9999812841415405],
        'mobilenet_v2-kather100K'       : [1.0, 0.9998366832733154],
        'mobilenet_v3_large-kather100K' : [1.0, 0.9999945163726807],
        'mobilenet_v3_small-kather100K' : [1.0, 0.9999963045120239],
        'googlenet-kather100K'          : [1.0, 0.998254120349884],

    }
    for pretrained_model, expected_prob in pretrained_info.items():
        _test_predictor_correctness(
            input_list, pretrained_model,
            probabilities_check=expected_prob,
            predictions_check=[5, 8])


# # -------------------------------------------------------------------------------------
# # Command Line Interface
# # -------------------------------------------------------------------------------------


# def test_command_line_patch_predictor(_dir_sample_patches, _sample_patch1):
#     """Test for the patch predictor CLI."""
#     runner = CliRunner()
#     patch_predictor_dir = runner.invoke(
#         cli.main,
#         [
#             "patch-predictor",
#             "--pretrained_model",
#             "resnet18-kather100K",
#             "--img_input",
#             str(pathlib.Path(_dir_sample_patches)),
#             "--output_path",
#             "tmp_output",
#             "--batch_size",
#             2,
#             "--mode",
#             "patch",
#             "--return_probabilities",
#             False,
#         ],
#     )

#     assert patch_predictor_dir.exit_code == 0
#     shutil.rmtree("tmp_output", ignore_errors=True)

#     patch_predictor_single_path = runner.invoke(
#         cli.main,
#         [
#             "patch-predictor",
#             "--pretrained_model",
#             "resnet18-kather100K",
#             "--img_input",
#             pathlib.Path(_sample_patch1),
#             "--output_path",
#             "tmp_output",
#             "--batch_size",
#             2,
#             "--mode",
#             "patch",
#             "--return_probabilities",
#             False,
#         ],
#     )

#     assert patch_predictor_single_path.exit_code == 0
#     shutil.rmtree("tmp_output", ignore_errors=True)


# def test_command_line_patch_predictor_crash(_sample_patch1):
#     """Test for the patch predictor CLI."""
#     # test single image not exist
#     runner = CliRunner()
#     result = runner.invoke(
#         cli.main,
#         [
#             "patch-predictor",
#             "--pretrained_model",
#             "resnet18-kather100K",
#             "--img_input",
#             "imaginary_img.tif",
#             "--mode",
#             "patch",
#         ],
#     )
#     assert result.exit_code != 0

#     # test not pretrained model
#     result = runner.invoke(
#         cli.main,
#         [
#             "patch-predictor",
#             "--pretrained_model",
#             "secret_model",
#             "--img_input",
#             pathlib.Path(_sample_patch1),
#             "--mode",
#             "patch",
#         ],
#     )
#     assert result.exit_code != 0
