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


def test_patch_dataset_path_imgs():
    """Test for patch dataset with a list of file paths as input."""
    size = (224, 224, 3)
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    dataset = Patch_Dataset(list_paths)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    for idx, sampled_img in enumerate(dataloader):
        assert (sampled_img.shape.shape == size).all()

    return


def test_patch_dataset_list_imgs():
    """Test for patch dataset with a list of images as input."""
    size = (5, 5, 3)
    img = np.random.randint(0, 255, size=size)
    list_imgs = [img, img, img]
    dataset = Patch_Dataset(list_imgs)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    for idx, sampled_img in enumerate(dataloader):
        assert (sampled_img.shape.shape == size).all()


def test_patch_dataset_array_imgs():
    """Test for patch dataset with a numpy array of a list of images."""
    size = (5, 5, 3)
    img = np.random.randint(0, 255, size=size)
    list_imgs = [img, img, img]
    array_imgs = np.array(list_imgs)
    dataset = Patch_Dataset(array_imgs)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    for idx, sampled_img in enumerate(dataloader):
        assert (sampled_img.shape.shape == size).all()


def test_kather_patch_dataset():
    """Test for kather patch dataset."""
    size = (224, 224, 3)
    # save to temporary location
    save_dir_path = (os.path.join(TIATOOLBOX_HOME, "tmp/"),)
    os.mkdir(save_dir_path)
    dataset = Kather_Patch_Dataset(save_dir_path=save_dir_path)
    # remove generated data - just a test!
    os.rmdir(save_dir_path)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    for idx, sampled_img in enumerate(dataloader):
        assert (sampled_img.shape.shape == size).all()


def test_patch_predictor_kather_resnet18_api1():
    """Test for patch predictor on kather dataset with resnet18 using method 1."""

    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    dataset = Patch_Dataset(list_paths)

    # API 1
    predictor = CNN_Patch_Predictor(predefined_model="resnet18_kather", batch_size=1)
    # don't run test on GPU
    output = predictor.predict(dataset, on_gpu=False)
    # ensure that the raw output is correct
    assert np.sum(output) == 1.5


def test_patch_predictor_kather_resnet18_api1():
    """Test for patch predictor on kather dataset with resnet18 using method 2."""

    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    dataset = Patch_Dataset(list_paths)

    # API 2
    pretrained_weight_url = (
        "https://tiatoolbox.dcs.warwick.ac.uk/models/resnet18_kather_pc.pth",
    )
    os.mkdir = os.path.join(TIATOOLBOX_HOME, "tmp/")
    pretrained_weight = os.path.join(TIATOOLBOX_HOME, "tmp/", "resnet18_kather_pc.pth")
    download_data(pretrained_weight_url, pretrained_weight)

    predictor = CNN_Patch_Predictor(
        predefined_model="resnet18_kather",
        pretrained_weight=pretrained_weight,
        batch_size=1,
    )
    # don't run test on GPU
    output = predictor.predict(dataset, on_gpu=False)
    # ensure that the raw output is correct
    assert np.sum(output) == 1.5

    os.rmdir(os.path.join(TIATOOLBOX_HOME, "tmp/"))


def test_patch_predictor_kather_resnet18_api1():
    """Test for patch predictor on kather dataset with resnet18 using method 3."""

    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")
    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    dataset = Patch_Dataset(list_paths)

    # API 3
    model = CNN_Patch_Model(backbone="resnet18", nr_classes=9)
    predictor = CNN_Patch_Predictor(model=model, batch_size=1)
    # don't run test on GPU
    output = predictor.predict(dataset, on_gpu=False)
    # ensure that the raw output is correct
    assert np.sum(output) == 1.5
