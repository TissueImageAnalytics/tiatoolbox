from tiatoolbox.models.classification import CNN_Patch_Model, CNN_Patch_Predictor
from tiatoolbox.models.dataset import Patch_Dataset, Kather_Patch_Dataset
from tiatoolbox.utils.misc import download_data
from tiatoolbox import rcParam
from tiatoolbox.utils.misc import grab_files_from_dir

import pytest
import torch
import pathlib
import os
import numpy as np
import shutil


def test_set_root_dir():
    """Test for setting new root dir."""
    old_root_dir = rcParam["TIATOOLBOX_HOME"]
    print(os.getcwd())
    test_dir_path = os.path.join(os.getcwd(), "tmp_check/")
    # clean up prev test
    if os.path.exists(test_dir_path):
        os.rmdir(test_dir_path)
    rcParam["TIATOOLBOX_HOME"] = test_dir_path
    # reimport to see if its overwrite, it should be changed
    # silence deep source becaus this is intentional check
    # skipcq
    from tiatoolbox import rcParam

    os.makedirs(rcParam["TIATOOLBOX_HOME"])
    if not os.path.exists(test_dir_path):
        assert False, "`%s` != `%s`" % (rcParam["TIATOOLBOX_HOME"], test_dir_path)
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"], ignore_errors=True)
    rcParam["TIATOOLBOX_HOME"] = old_root_dir  # reassign for subsequent test


def test_patch_dataset_path_imgs():
    """Test for patch dataset with a list of file paths as input."""
    size = (224, 224, 3)
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    dataset = Patch_Dataset(list_paths)

    dataset.preproc_func = lambda x: x

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        drop_last=False,
    )

    for _, sampled_img in enumerate(dataloader):
        sampled_img_shape = sampled_img.shape
        assert (
            sampled_img_shape[1] == size[0]
            and sampled_img_shape[2] == size[1]
            and sampled_img_shape[3] == size[2]
        )


def test_patch_dataset_list_imgs():
    """Test for patch dataset with a list of images as input."""
    size = (5, 5, 3)
    img = np.random.randint(0, 255, size=size)
    list_imgs = [img, img, img]
    dataset = Patch_Dataset(list_imgs)

    dataset.preproc_func = lambda x: x

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        drop_last=False,
    )

    for _, sampled_img in enumerate(dataloader):
        sampled_img_shape = sampled_img.shape
        assert (
            sampled_img_shape[1] == size[0]
            and sampled_img_shape[2] == size[1]
            and sampled_img_shape[3] == size[2]
        )


def test_patch_dataset_array_imgs():
    """Test for patch dataset with a numpy array of a list of images."""
    size = (5, 5, 3)
    img = np.random.randint(0, 255, size=size)
    list_imgs = [img, img, img]
    array_imgs = np.array(list_imgs)
    dataset = Patch_Dataset(array_imgs)

    dataset.preproc_func = lambda x: x

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        drop_last=False,
    )

    for _, sampled_img in enumerate(dataloader):
        sampled_img_shape = sampled_img.shape
        assert (
            sampled_img_shape[1] == size[0]
            and sampled_img_shape[2] == size[1]
            and sampled_img_shape[3] == size[2]
        )


def test_patch_dataset_crash():
    """Test to make sure patch dataset crashes with incorrect input."""
    # all examples below should fail when input to Patch_Dataset

    # ndarray of mixed dtype
    img_list = np.array([np.random.randint(0, 255, (4, 5, 3)), "Should crash"])
    with pytest.raises(ValueError, match="Provided input array is non-numerical."):
        _ = Patch_Dataset(img_list)

    # list of ndarrays with different sizes
    img_list = [
        np.random.randint(0, 255, (4, 4, 3)),
        np.random.randint(0, 255, (4, 5, 3)),
    ]
    with pytest.raises(ValueError, match="Images must have the same dimensions."):
        _ = Patch_Dataset(img_list)

    # list of mixed dtype
    img_list = [np.random.randint(0, 255, (4, 4, 3)), "you_should_crash_here", 123, 456]
    with pytest.raises(
        ValueError,
        match="Input must be either a list/array of images or a list of valid image paths.",
    ):
        _ = Patch_Dataset(img_list)

    # list of mixed dtype
    img_list = ["you_should_crash_here", 123, 456]
    with pytest.raises(
        ValueError,
        match="Input must be either a list/array of images or a list of valid image paths.",
    ):
        _ = Patch_Dataset(img_list)


def test_kather_patch_dataset():
    """Test for kather patch dataset."""
    size = (224, 224, 3)
    # save to temporary location
    save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "tmp_check/")
    # remove prev generated data - just a test!
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    print(save_dir_path)
    dataset = Kather_Patch_Dataset(save_dir_path=save_dir_path, return_labels=True)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    for _, sampled_data in enumerate(dataloader):
        sampled_img, sampled_labels = sampled_data
        assert np.sum(sampled_img.shape == size) == 0
        assert len(sampled_labels) == 1

    # remove generated data - just a test!
    shutil.rmtree(save_dir_path, ignore_errors=True)


def test_patch_predictor_api1():
    """Test for patch predictor API 1. Test with resnet18 on Kather 100K dataset."""
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    print(list_paths)
    dataset = Patch_Dataset(list_paths)

    # API 1, also test with return_labels
    predictor = CNN_Patch_Predictor(
        predefined_model="resnet18-kather100K", batch_size=1
    )
    # don't run test on GPU
    output = predictor.predict(
        dataset, return_probs=True, return_labels=True, on_gpu=False
    )
    probs = output["probs"]
    preds = output["preds"]
    labels = output["labels"]

    # ensure that the raw output is correct
    # ! @SIMON how to assert check ^^^^ this ?

    # placeholder
    assert len(probs) == len(preds)
    assert len(probs) == len(labels)


def test_patch_predictor_api2():
    """Test for patch predictor API 2. Test with resnet18 on Kather 100K dataset."""
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    dataset = Patch_Dataset(list_paths)

    # API 2
    pretrained_weight_url = (
        "https://tiatoolbox.dcs.warwick.ac.uk/models/resnet18-kather100K-pc.pth"
    )

    save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "tmp_api2")
    # remove prev generated data - just a test!
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    os.makedirs(save_dir_path)

    pretrained_weight = os.path.join(
        rcParam["TIATOOLBOX_HOME"], "tmp_api2", "resnet18-kather100K-pc.pth"
    )
    download_data(pretrained_weight_url, pretrained_weight)

    predictor = CNN_Patch_Predictor(
        predefined_model="resnet18-kather100K",
        pretrained_weight=pretrained_weight,
        batch_size=1,
    )
    # don't run test on GPU
    output = predictor.predict(dataset, return_probs=True, on_gpu=False)
    probs = output["probs"]
    preds = output["preds"]

    # ensure that the raw output is correct
    # ! @SIMON how to assert check ^^^^ this ?

    # placeholder
    assert len(probs) == len(preds)

    # remove generated data - just a test!
    shutil.rmtree(save_dir_path, ignore_errors=True)


def test_patch_predictor_api3():
    """Test for patch predictor API 3. Test with resnet18 on Kather 100K dataset."""
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    dataset = Patch_Dataset(list_paths)

    # API 3
    model = CNN_Patch_Model(backbone="resnet18", nr_classes=9)
    predictor = CNN_Patch_Predictor(model=model, batch_size=1)
    # don't run test on GPU
    output = predictor.predict(dataset, return_probs=True, on_gpu=False)

    probs = output["probs"]
    preds = output["preds"]

    # ensure that the raw output is correct
    # ! @SIMON how to assert check ^^^^ this ?

    # placeholder
    assert len(probs) == len(preds)


def test_patch_predictor_alexnet_kather100K():
    """Test for patch predictor with alexnet on Kather 100K dataset."""
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    print(list_paths)
    dataset = Patch_Dataset(list_paths)

    # API 1, also test with return_labels
    predictor = CNN_Patch_Predictor(predefined_model="alexnet-kather100K", batch_size=1)
    # don't run test on GPU
    output = predictor.predict(
        dataset, return_probs=True, return_labels=True, on_gpu=False
    )
    probs = output["probs"]
    preds = output["preds"]
    labels = output["labels"]

    # ensure that the raw output is correct
    # ! @SIMON how to assert check ^^^^ this ?

    # placeholder
    assert len(probs) == len(preds)
    assert len(probs) == len(labels)


def test_patch_predictor_resnet34_kather100K():
    """Test for patch predictor with resnet34 on Kather 100K dataset."""
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    print(list_paths)
    dataset = Patch_Dataset(list_paths)

    # API 1, also test with return_labels
    predictor = CNN_Patch_Predictor(
        predefined_model="resnet34-kather100K", batch_size=1
    )
    # don't run test on GPU
    output = predictor.predict(
        dataset, return_probs=True, return_labels=True, on_gpu=False
    )
    probs = output["probs"]
    preds = output["preds"]
    labels = output["labels"]

    # ensure that the raw output is correct
    # ! @SIMON how to assert check ^^^^ this ?

    # placeholder
    assert len(probs) == len(preds)
    assert len(probs) == len(labels)


def test_patch_predictor_resnet50_kather100K():
    """Test for patch predictor with resnet50 on Kather 100K dataset."""
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    print(list_paths)
    dataset = Patch_Dataset(list_paths)

    # API 1, also test with return_labels
    predictor = CNN_Patch_Predictor(
        predefined_model="resnet50-kather100K", batch_size=1
    )
    # don't run test on GPU
    output = predictor.predict(
        dataset, return_probs=True, return_labels=True, on_gpu=False
    )
    probs = output["probs"]
    preds = output["preds"]
    labels = output["labels"]

    # ensure that the raw output is correct
    # ! @SIMON how to assert check ^^^^ this ?

    # placeholder
    assert len(probs) == len(preds)
    assert len(probs) == len(labels)


def test_patch_predictor_resnet101_kather100K():
    """Test for patch predictor with resnet101 on Kather 100K dataset."""
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    print(list_paths)
    dataset = Patch_Dataset(list_paths)

    # API 1, also test with return_labels
    predictor = CNN_Patch_Predictor(
        predefined_model="resnet101-kather100K", batch_size=1
    )
    # don't run test on GPU
    output = predictor.predict(
        dataset, return_probs=True, return_labels=True, on_gpu=False
    )
    probs = output["probs"]
    preds = output["preds"]
    labels = output["labels"]

    # ensure that the raw output is correct
    # ! @SIMON how to assert check ^^^^ this ?

    # placeholder
    assert len(probs) == len(preds)
    assert len(probs) == len(labels)


def test_patch_predictor_resnext50_32x4d_kather100K():
    """Test for patch predictor with resnext50_32x4d on Kather 100K dataset."""
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    print(list_paths)
    dataset = Patch_Dataset(list_paths)

    # API 1, also test with return_labels
    predictor = CNN_Patch_Predictor(
        predefined_model="resnext50_32x4d-kather100K", batch_size=1
    )
    # don't run test on GPU
    output = predictor.predict(
        dataset, return_probs=True, return_labels=True, on_gpu=False
    )
    probs = output["probs"]
    preds = output["preds"]
    labels = output["labels"]

    # ensure that the raw output is correct
    # ! @SIMON how to assert check ^^^^ this ?
    # placeholder
    assert len(probs) == len(preds)
    assert len(preds) == len(labels)


def test_patch_predictor_resnext101_32x8d_kather100K():
    """Test for patch predictor with resnext101_32x8d on Kather 100K dataset."""
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    print(list_paths)
    dataset = Patch_Dataset(list_paths)

    # API 1, also test with return_labels
    predictor = CNN_Patch_Predictor(
        predefined_model="resnext101_32x8d-kather100K", batch_size=1
    )
    # don't run test on GPU
    output = predictor.predict(
        dataset, return_probs=True, return_labels=True, on_gpu=False
    )
    probs = output["probs"]
    preds = output["preds"]
    labels = output["labels"]

    # ensure that the raw output is correct
    # ! @SIMON how to assert check ^^^^ this ?

    # placeholder
    assert len(probs) == len(preds)
    assert len(probs) == len(labels)


def test_patch_predictor_wide_resnet50_2_kather100K():
    """Test for patch predictor with wide_resnet50_2 on Kather 100K dataset."""
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    print(list_paths)
    dataset = Patch_Dataset(list_paths)

    # API 1, also test with return_labels
    predictor = CNN_Patch_Predictor(
        predefined_model="wide_resnet50_2-kather100K", batch_size=1
    )
    # don't run test on GPU
    output = predictor.predict(
        dataset, return_probs=True, return_labels=True, on_gpu=False
    )
    probs = output["probs"]
    preds = output["preds"]
    labels = output["labels"]

    # ensure that the raw output is correct
    # ! @SIMON how to assert check ^^^^ this ?

    # placeholder
    assert len(probs) == len(preds)
    assert len(probs) == len(labels)
