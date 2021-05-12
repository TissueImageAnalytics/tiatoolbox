import os
import pathlib
import shutil

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from tiatoolbox import rcParam
from tiatoolbox.models.backbone import get_model
from tiatoolbox.models.classification.abc import Model_Base
from tiatoolbox.models.classification import CNN_Patch_Model, CNN_Patch_Predictor
from tiatoolbox.models.dataset import (
    Kather_Patch_Dataset,
    Patch_Dataset,
    predefined_preproc_func,
)
from tiatoolbox.utils.misc import download_data, grab_files_from_dir, unzip_data
from tiatoolbox import cli


def _get_outputs_api1(dataset, predefined_model):
    """Helper function to get the model output using API 1."""

    # API 1, also test with return_labels
    predictor = CNN_Patch_Predictor(predefined_model=predefined_model, batch_size=1)
    # don't run test on GPU
    output = predictor.predict(
        dataset, return_probs=True, return_labels=True, on_gpu=False
    )
    probs = output["probs"]
    preds = output["preds"]
    labels = output["labels"]

    return probs, preds, labels


def _get_outputs_api2(dataset, predefined_model):
    """Helper function to get the model output using API 2."""

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
        predefined_model=predefined_model,
        pretrained_weight=pretrained_weight,
        batch_size=1,
    )
    # don't run test on GPU
    output = predictor.predict(
        dataset, return_probs=True, return_labels=True, on_gpu=False
    )
    probs = output["probs"]
    preds = output["preds"]
    labels = output["labels"]

    return probs, preds, labels, save_dir_path


def _get_outputs_api3(dataset, backbone, nr_classes=9):
    """Helper function to get the model output using API 3."""

    # API 3
    model = CNN_Patch_Model(backbone=backbone, nr_classes=nr_classes)

    # coverage setter check
    model.set_preproc_func(lambda x: x - 1)  # do this for coverage
    assert model.get_preproc_func()(1) == 0
    # coverage setter check
    model.set_preproc_func(None)  # do this for coverage
    assert model.get_preproc_func()(1) == 1

    predictor = CNN_Patch_Predictor(model=model, batch_size=1, verbose=False)
    # don't run test on GPU
    output = predictor.predict(
        dataset, return_probs=True, return_labels=True, on_gpu=False
    )

    probs = output["probs"]
    preds = output["preds"]
    labels = output["labels"]

    return probs, preds, labels


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
        Model_Base()
    with pytest.raises(NotImplementedError):
        Model_Base.infer_batch(1, 2, 3)

    # without providing any model
    with pytest.raises(ValueError, match=r"Must provide.*"):
        CNN_Patch_Predictor()

    # provide wrong unknown predefined model
    with pytest.raises(ValueError, match=r"Predefined .* does not exist"):
        CNN_Patch_Predictor(predefined_model="secret_model")

    # provide wrong model of unknown type, deprecated later with type hint
    with pytest.raises(ValueError, match=r".*must be a string.*"):
        CNN_Patch_Predictor(predefined_model=123)

    # model and dummy input
    model = CNN_Patch_Predictor(predefined_model="resnet34-kather100K")
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


def test_patch_dataset_path_imgs(_sample_patch1, _sample_patch2):
    """Test for patch dataset with a list of file paths as input."""
    size = (224, 224, 3)

    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)]
    )

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
    _ = Patch_Dataset(img_list)
    assert img_list[0] is not None
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"])


def test_patch_dataset_array_imgs():
    """Test for patch dataset with a numpy array of a list of images."""
    size = (5, 5, 3)
    img = np.random.randint(0, 255, size=size)
    list_imgs = [img, img, img]
    label_list = [1, 2, 3]
    array_imgs = np.array(list_imgs)

    # test different setter for label
    dataset = Patch_Dataset(array_imgs, label_list=label_list, return_labels=True)
    assert dataset[2][1] == 3
    dataset = Patch_Dataset(array_imgs, label_list=None, return_labels=True)
    assert np.isnan(dataset[2][1]), dataset[2][1]

    dataset = Patch_Dataset(array_imgs)
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

    # not supported input type
    img_list = {"a": np.random.randint(0, 255, (4, 4, 4))}
    with pytest.raises(
        ValueError, match=r".*Input must be either a list/array of images.*"
    ):
        _ = Patch_Dataset(img_list)

    # ndarray of mixed dtype
    img_list = np.array([np.random.randint(0, 255, (4, 5, 3)), "Should crash"])
    with pytest.raises(ValueError, match="Provided input array is non-numerical."):
        _ = Patch_Dataset(img_list)

    # ndarrays of NHW images
    img_list = np.random.randint(0, 255, (4, 4, 4))
    with pytest.raises(ValueError, match=r".*array of images of the form NHWC.*"):
        _ = Patch_Dataset(img_list)

    # list of ndarrays with different sizes
    img_list = [
        np.random.randint(0, 255, (4, 4, 3)),
        np.random.randint(0, 255, (4, 5, 3)),
    ]
    with pytest.raises(ValueError, match="Images must have the same dimensions."):
        _ = Patch_Dataset(img_list)

    # list of ndarrays with HW and HWC mixed up
    img_list = [
        np.random.randint(0, 255, (4, 4, 3)),
        np.random.randint(0, 255, (4, 4)),
    ]
    with pytest.raises(
        ValueError, match="Each sample must be an array of the form HWC."
    ):
        _ = Patch_Dataset(img_list)

    # list of mixed dtype
    img_list = [np.random.randint(0, 255, (4, 4, 3)), "you_should_crash_here", 123, 456]
    with pytest.raises(
        ValueError,
        match="Input must be either a list/array of images or a list of "
        "valid image paths.",
    ):
        _ = Patch_Dataset(img_list)

    # list of mixed dtype
    img_list = ["you_should_crash_here", 123, 456]
    with pytest.raises(
        ValueError,
        match="Input must be either a list/array of images or a list of "
        "valid image paths.",
    ):
        _ = Patch_Dataset(img_list)

    # list not exist paths
    with pytest.raises(
        ValueError,
        match=r".*valid image paths.*",
    ):
        _ = Patch_Dataset(["img.npy"])

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
        _ = Patch_Dataset(img_list)
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"])

    # preproc func for not defined dataset
    with pytest.raises(
        ValueError,
        match=r".* preprocessing .* does not exist.",
    ):
        predefined_preproc_func("secret_dataset")


def test_kather_patch_dataset():
    """Test for kather patch dataset."""
    size = (224, 224, 3)
    # test kather with default param
    dataset = Kather_Patch_Dataset()
    # kather with default data path skip download
    dataset = Kather_Patch_Dataset()
    # pytest for not exist dir
    with pytest.raises(
        ValueError,
        match=r".*not exist.*",
    ):
        _ = Kather_Patch_Dataset(save_dir_path="unknown_place")
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
    dataset = Kather_Patch_Dataset(save_dir_path=extracted_dir, return_labels=True)

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

    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "resnet18-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9999717473983765]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_api2(_sample_patch1, _sample_patch2):
    """Test for patch predictor API 2. Test with resnet18 on Kather 100K dataset."""

    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels, save_dir_path = _get_outputs_api2(
        dataset, "resnet18-kather100K"
    )

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9999717473983765]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )

    # remove generated data - just a test!
    shutil.rmtree(save_dir_path, ignore_errors=True)


def test_patch_predictor_api3(_sample_patch1, _sample_patch2):
    """Test for patch predictor API 3. Test with resnet18 on Kather 100K dataset."""

    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api3(dataset, "resnet18")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)


def test_patch_predictor_alexnet_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with alexnet on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset([_sample_patch1, _sample_patch2], return_labels=True)
    probs, preds, labels = _get_outputs_api1(dataset, "alexnet-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9998185038566589]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_resnet34_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with resnet34 on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "resnet34-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9991286396980286]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_resnet50_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with resnet50 on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "resnet50-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9969022870063782]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_resnet101_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with resnet101 on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "resnet101-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9999957084655762]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_resnext50_32x4d_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with resnext50_32x4d on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "resnext50_32x4d-kather100K")

    assert len(probs) == len(preds)
    assert len(preds) == len(labels)

    prob_check = [1.0, 0.9999779462814331]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_resnext101_32x8d_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with resnext101_32x8d on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "resnext101_32x8d-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9999345541000366]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_wide_resnet50_2_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with wide_resnet50_2 on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "wide_resnet50_2-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9999997615814209]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_wide_resnet101_2_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with wide_resnet101_2 on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "wide_resnet101_2-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.999420166015625]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_densenet121_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with densenet121 on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "densenet121-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9998136162757874]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_densenet161_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with densenet161 on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "densenet161-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9999997615814209]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_densenet169_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with densenet169 on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "densenet169-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9999773502349854]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_densenet201_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with densenet201 on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "densenet201-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9999812841415405]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_mobilenet_v2_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with mobilenet_v2 on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "mobilenet_v2-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9998366832733154]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_mobilenet_v3_large_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with mobilenet_v3_large on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "mobilenet_v3_large-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9999945163726807]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_mobilenet_v3_small_kather100K(_sample_patch1, _sample_patch2):
    """Test for patch predictor with mobilenet_v3_small on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "mobilenet_v3_small-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9999963045120239]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


def test_patch_predictor_googlenet(_sample_patch1, _sample_patch2):
    """Test for patch predictor with googlenet on Kather 100K dataset."""

    # API 1, also test with return_labels
    dataset = Patch_Dataset(
        [pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)], return_labels=True
    )
    probs, preds, labels = _get_outputs_api1(dataset, "googlenet-kather100K")

    assert len(probs) == len(preds)
    assert len(probs) == len(labels)

    prob_check = [1.0, 0.9999963045120239]
    pred_check = [5, 8]
    for idx, probs_ in enumerate(probs):
        prob_max = max(probs_)
        assert (
            np.abs(prob_max - prob_check[idx]) <= 1e-8 and preds[idx] == pred_check[idx]
        )


# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_patch_predictor():
    """Test for the patch predictor CLI."""
    file_parent_dir = pathlib.Path(__file__).parent
    dir_patches = file_parent_dir.joinpath("data/sample_patches/")

    runner = CliRunner()
    patch_predictor_dir = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--predefined_model",
            "resnet18-kather100K",
            "--img_input",
            dir_patches,
            "--batch_size",
            2,
            "--return_probs",
            False,
        ],
    )

    assert patch_predictor_dir.exit_code == 0

    list_paths = grab_files_from_dir(dir_patches, file_types="*.tif")
    single_path = list_paths[0]
    patch_predictor_single_path = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--predefined_model",
            "resnet18-kather100K",
            "--img_input",
            single_path,
            "--batch_size",
            2,
            "--return_probs",
            False,
        ],
    )

    assert patch_predictor_single_path.exit_code == 0


def test_command_line_patch_predictor_crash():
    """Test for the patch predictor CLI."""
    file_parent_dir = pathlib.Path(__file__).parent
    img_path = file_parent_dir.joinpath("data/sample_patches/kather1_unknown.tif")

    # test single image not exist
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--predefined_model",
            "resnet18-kather100K",
            "--img_input",
            img_path,
        ],
    )
    assert result.exit_code != 0

    # test not predefined model
    img_path = file_parent_dir.joinpath("data/sample_patches/kather1.tif")
    result = runner.invoke(
        cli.main,
        [
            "patch-predictor",
            "--predefined_model",
            "secret_model",
            "--img_input",
            img_path,
        ],
    )
    assert result.exit_code != 0
