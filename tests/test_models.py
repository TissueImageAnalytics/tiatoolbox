from tiatoolbox.models.classification import CNN_Patch_Model, CNN_Patch_Predictor
from tiatoolbox.models.dataset import Patch_Dataset, Kather_Patch_Dataset
from tiatoolbox.utils.misc import download_data
from tiatoolbox import TIATOOLBOX_HOME
from tiatoolbox.utils.misc import grab_files_from_dir

import pytest
import torch
import pathlib
import os
import numpy as np
import shutil


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
    save_dir_path = os.path.join(TIATOOLBOX_HOME, "tmp")
    os.mkdir(save_dir_path)
    dataset = Kather_Patch_Dataset(save_dir_path=save_dir_path)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    for _, sampled_img in enumerate(dataloader):
        assert (sampled_img.shape.shape == size).all()

    # remove generated data - just a test!
    shutil.rmtree(save_dir_path, ignore_errors=True)


def test_patch_predictor_kather_resnet18_api1():
    """Test for patch predictor on kather dataset with resnet18 using method 1."""
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    dataset = Patch_Dataset(list_paths)

    # API 1
    predictor = CNN_Patch_Predictor(predefined_model="resnet18_kather", batch_size=1)
    # don't run test on GPU
    output = predictor.predict(dataset, return_probs=True, on_gpu=True)
    probs = output["probs"]
    preds = output["preds"]

    # ensure that the output is correct
    len(probs[0]) == 9


def test_patch_predictor_kather_resnet18_api2():
    """Test for patch predictor on kather dataset with resnet18 using method 2."""
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    dataset = Patch_Dataset(list_paths)

    # API 2
    pretrained_weight_url = (
        "https://tiatoolbox.dcs.warwick.ac.uk/models/resnet18_kather_pc.pth"
    )

    os.mkdir = os.path.join(TIATOOLBOX_HOME, "tmp")
    pretrained_weight = os.path.join(TIATOOLBOX_HOME, "tmp", "resnet18_kather_pc.pth")
    download_data(pretrained_weight_url, pretrained_weight)

    predictor = CNN_Patch_Predictor(
        predefined_model="resnet18_kather",
        pretrained_weight=pretrained_weight,
        batch_size=1,
    )
    # don't run test on GPU
    output = predictor.predict(dataset, return_probs=True, on_gpu=True)
    probs = output["probs"]
    preds = output["preds"]

    # ensure that the output is correct
    assert len(probs[0]) == 9

    # remove generated data - just a test!
    shutil.rmtree(os.path.join(TIATOOLBOX_HOME, "tmp"), ignore_errors=True)


def test_patch_predictor_kather_resnet18_api3():
    """Test for patch predictor on kather dataset with resnet18 using method 3."""
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    dataset = Patch_Dataset(list_paths)

    # API 3
    model = CNN_Patch_Model(backbone="resnet18", nr_classes=9)
    predictor = CNN_Patch_Predictor(model=model, batch_size=1)
    # don't run test on GPU
    output = predictor.predict(dataset, return_probs=True, on_gpu=True)

    probs = output["probs"]
    preds = output["preds"]

    # ensure that the raw output is correct
    len(probs[0]) == 9
