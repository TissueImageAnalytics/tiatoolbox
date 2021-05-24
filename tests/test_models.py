"""Tests for code related to model usage."""

import os
import pathlib
import shutil

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from tiatoolbox import rcParam
from tiatoolbox.models.backbone import get_model
from tiatoolbox.models.classification.abc import ModelBase
from tiatoolbox.models.classification import CNNPatchModel, CNNPatchPredictor
from tiatoolbox.models.dataset import (
    KatherPatchDataset,
    PatchDataset,
    predefined_preproc_func,
)
from tiatoolbox.utils.misc import download_data, unzip_data
from tiatoolbox import cli


def _test_outputs_api1(
    dataset,
    predefined_model,
    return_probabilities=True,
    return_labels=True,
    probabilities_check=None,
    predictions_check=None,
    on_gpu=False,
):
    """Helper function to get the model output using API 1."""
    # API 1, also test with return_labels
    predictor = CNNPatchPredictor(predefined_model=predefined_model, batch_size=1)
    # don't run test on GPU
    output = predictor.predict(
        dataset,
        return_probabilities=return_probabilities,
        return_labels=return_labels,
        on_gpu=on_gpu,
    )
    predictions = output["predictions"]
    if return_probabilities:
        probabilities = output["probabilities"]
        assert len(probabilities) == len(predictions)
    if return_labels:
        labels = output["labels"]
        assert len(labels) == len(predictions)

    if return_probabilities:
        for idx, probabilities_ in enumerate(probabilities):
            probabilities_max = max(probabilities_)
            print(probabilities_max, predictions[idx])
            assert (
                np.abs(probabilities_max - probabilities_check[idx]) <= 1e-8
                and predictions[idx] == predictions_check[idx]
            )


def _test_outputs_api2(
    dataset,
    predefined_model,
    return_probabilities=True,
    return_labels=True,
    probabilities_check=None,
    predictions_check=None,
    on_gpu=False,
):
    """Helper function to get the model output using API 2."""
    # API 2
    pretrained_weight_url = (
        "https://tiatoolbox.dcs.warwick.ac.uk/models/pc/resnet18-kather100k.pth"
    )

    save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "tmp_api2")
    # remove prev generated data - just a test!
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    os.makedirs(save_dir_path)

    pretrained_weight = os.path.join(
        rcParam["TIATOOLBOX_HOME"], "tmp_api2", "resnet18-kather100k.pth"
    )
    download_data(pretrained_weight_url, pretrained_weight)

    predictor = CNNPatchPredictor(
        predefined_model=predefined_model,
        pretrained_weight=pretrained_weight,
        batch_size=1,
    )
    # don't run test on GPU
    output = predictor.predict(
        dataset,
        return_probabilities=return_probabilities,
        return_labels=return_labels,
        on_gpu=on_gpu,
    )
    predictions = output["predictions"]
    if return_probabilities:
        probabilities = output["probabilities"]
        assert len(probabilities) == len(predictions)
    if return_labels:
        labels = output["labels"]
        assert len(labels) == len(predictions)

    if return_probabilities:
        for idx, probabilities_ in enumerate(probabilities):
            probabilities_max = max(probabilities_)
            assert (
                np.abs(probabilities_max - probabilities_check[idx]) <= 1e-8
                and predictions[idx] == predictions_check[idx]
            )

    return save_dir_path


def _test_outputs_api3(
    dataset,
    backbone,
    return_probabilities=True,
    return_labels=True,
    probabilities_check=None,
    predictions_check=None,
    num_classes=9,
    on_gpu=False,
):
    """Helper function to get the model output using API 3."""
    # API 3
    model = CNNPatchModel(backbone=backbone, num_classes=num_classes)

    # coverage setter check
    model.set_preproc_func(lambda x: x - 1)  # do this for coverage
    assert model.get_preproc_func()(1) == 0
    # coverage setter check
    model.set_preproc_func(None)  # do this for coverage
    assert model.get_preproc_func()(1) == 1

    predictor = CNNPatchPredictor(model=model, batch_size=1, verbose=False)
    # don't run test on GPU
    output = predictor.predict(
        dataset,
        return_probabilities=return_probabilities,
        return_labels=return_labels,
        on_gpu=on_gpu,
    )
    predictions = output["predictions"]
    if return_probabilities:
        probabilities = output["probabilities"]
        assert len(probabilities) == len(predictions)
    if return_labels:
        labels = output["labels"]
        assert len(labels) == len(predictions)


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
        get_model("secret_model", pretrained=False)


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

    # provide wrong unknown predefined model
    with pytest.raises(ValueError, match=r"Predefined .* does not exist"):
        CNNPatchPredictor(predefined_model="secret_model")

    # provide wrong model of unknown type, deprecated later with type hint
    with pytest.raises(ValueError, match=r".*must be a string.*"):
        CNNPatchPredictor(predefined_model=123)

    # model and dummy input
    model = CNNPatchPredictor(predefined_model="resnet34-kather100k")
    img_list = [
        np.random.randint(0, 255, (4, 4, 3)),
        np.random.randint(0, 255, (4, 4, 3)),
    ]
    # only receive a dataset object
    with pytest.raises(ValueError, match=r".*torch.utils.data.Dataset.*"):
        model.predict(img_list)


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


def test_PatchDataset_path_imgs(_sample_patch1, _sample_patch2):
    """Test for patch dataset with a list of file paths as input."""
    size = (224, 224, 3)

    dataset = PatchDataset([pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)])

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


def test_PatchDataset_list_imgs():
    """Test for patch dataset with a list of images as input."""
    size = (5, 5, 3)
    img = np.random.randint(0, 255, size=size)
    list_imgs = [img, img, img]
    dataset = PatchDataset(list_imgs)

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


def test_PatchDataset_array_imgs():
    """Test for patch dataset with a numpy array of a list of images."""
    size = (5, 5, 3)
    img = np.random.randint(0, 255, size=size)
    list_imgs = [img, img, img]
    label_list = [1, 2, 3]
    array_imgs = np.array(list_imgs)

    # test different setter for label
    dataset = PatchDataset(array_imgs, label_list=label_list, return_labels=True)
    assert dataset[2][1] == 3
    dataset = PatchDataset(array_imgs, label_list=None, return_labels=True)
    assert np.isnan(dataset[2][1]), dataset[2][1]

    dataset = PatchDataset(array_imgs)
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


def test_PatchDataset_crash():
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


def test_KatherPatchDataset():
    """Test for kather patch dataset."""
    size = (224, 224, 3)
    # test kather with default param
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
    dataset = KatherPatchDataset(save_dir_path=extracted_dir, return_labels=True)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    for _, sampled_data in enumerate(dataloader):
        sampled_img, sampled_labels = sampled_data
        assert np.sum(sampled_img.shape == size) == 0
        assert len(sampled_labels) == 1

    # remove generated data - just a test!
    shutil.rmtree(save_dir_path, ignore_errors=True)
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"])


def test_patch_predictor_api1(_sample_patch1, _sample_patch2):
    """Test for patch predictor API 1. Test with resnet18 on Kather 100K dataset."""
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9999911785125732]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "resnet18-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_api2(_sample_patch1, _sample_patch2):
    """Test for patch predictor API 2. Test with resnet18 on Kather 100K dataset."""
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9999911785125732]
    predictions_check = [6, 3]
    save_dir_path = _test_outputs_api2(
        dataset,
        "resnet18-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )
    # remove generated data - just a test!
    shutil.rmtree(save_dir_path, ignore_errors=True)


def test_patch_predictor_api3(_sample_patch1, _sample_patch2):
    """Test for patch predictor API 3. Test with resnet18 on Kather 100K dataset."""
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    _test_outputs_api3(
        dataset,
        "resnet18",
        return_probabilities=False,
        return_labels=True,
    )


def test_patch_predictor_api1_no_probs_and_labels(_sample_patch1, _sample_patch2):
    """Test for patch predictor API 1 that doesn't return probabilities or labels.
    Test with resnet18 on Kather 100K dataset.

    """
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9999911785125732]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "resnet18-kather100k",
        return_probabilities=False,
        return_labels=False,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_alexnet_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with alexnet on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9999735355377197]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "alexnet-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_resnet34_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with resnet34 on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9979840517044067]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "resnet34-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_resnet50_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with resnet50 on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9999986886978149]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "resnet50-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_resnet101_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with resnet101 on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9999932050704956]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "resnet101-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_resnext50_32x4d_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with resnext50_32x4d on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9910059571266174]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "resnext50_32x4d-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_resnext101_32x8d_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with resnext101_32x8d on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9999971389770508]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "resnext101_32x8d-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_wide_resnet50_2_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with wide_resnet50_2 on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9953408241271973]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "wide_resnet50_2-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_wide_resnet101_2_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with wide_resnet101_2 on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9999831914901733]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "wide_resnet101_2-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_densenet121_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with densenet121 on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 1.0]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "densenet121-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_densenet161_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with densenet161 on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9999959468841553]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "densenet161-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_densenet169_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with densenet169 on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9999934434890747]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "densenet169-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_densenet201_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with densenet201 on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9999983310699463]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "densenet201-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_mobilenet_v2_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with mobilenet_v2 on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [0.9999998807907104, 0.9999126195907593]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "mobilenet_v2-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_mobilenet_v3_large_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with mobilenet_v3_large on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [0.9999996423721313, 0.9999878406524658]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "mobilenet_v3_large-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_mobilenet_v3_small_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with mobilenet_v3_small on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [0.9999998807907104, 0.9999997615814209]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "mobilenet_v3_small-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


def test_patch_predictor_googlenet(_sample_patch1, _sample_patch2):
    """Test for patch predictor with googlenet on Kather 100K dataset."""
    # API 1, also test with return_labels
    dataset = PatchDataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probabilities_check = [1.0, 0.9999639987945557]
    predictions_check = [6, 3]
    _test_outputs_api1(
        dataset,
        "googlenet-kather100k",
        return_probabilities=True,
        return_labels=True,
        probabilities_check=probabilities_check,
        predictions_check=predictions_check,
    )


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_patch_predictor(_dir_sample_patches, _sample_patch1):
    """Test for the patch predictor CLI."""
    runner = CliRunner()
    patch_predictor_dir = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--predefined_model",
            "resnet18-kather100k",
            "--img_input",
            str(pathlib.Path(_dir_sample_patches)),
            "--output_path",
            "tmp_output",
            "--batch_size",
            2,
            "--return_probabilities",
            False,
        ],
    )

    assert patch_predictor_dir.exit_code == 0
    shutil.rmtree("tmp_output", ignore_errors=True)

    patch_predictor_single_path = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--predefined_model",
            "resnet18-kather100k",
            "--img_input",
            pathlib.Path(_sample_patch1),
            "--output_path",
            "tmp_output",
            "--batch_size",
            2,
            "--return_probabilities",
            False,
        ],
    )

    # remove dir and re-create to test coverage
    shutil.rmtree("tmp_output", ignore_errors=True)
    os.makedirs("tmp_output")
    patch_predictor_single_path = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--predefined_model",
            "resnet18-kather100k",
            "--img_input",
            pathlib.Path(_sample_patch1),
            "--output_path",
            "tmp_output",
            "--batch_size",
            2,
            "--return_probabilities",
            False,
        ],
    )
    output_list = os.listdir("tmp_output")
    assert len(output_list) > 0

    assert patch_predictor_single_path.exit_code == 0
    shutil.rmtree("tmp_output", ignore_errors=True)


def test_command_line_patch_predictor_crash(_sample_patch1):
    """Test for the patch predictor CLI."""
    # test single image not exist
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--predefined_model",
            "resnet18-kather100k",
            "--img_input",
            "imaginary_img.tif",
        ],
    )
    assert result.exit_code != 0

    # test not predefined model
    result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--predefined_model",
            "secret_model",
            "--img_input",
            pathlib.Path(_sample_patch1),
        ],
    )
    assert result.exit_code != 0
