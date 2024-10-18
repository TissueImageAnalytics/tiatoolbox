"""Test for predefined dataset within toolbox."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tiatoolbox import rcParam
from tiatoolbox.models import PatchDataset, WSIPatchDataset
from tiatoolbox.models.dataset import (
    DatasetInfoABC,
    KatherPatchDataset,
    PatchDatasetABC,
    predefined_preproc_func,
)
from tiatoolbox.utils import download_data, imread, imwrite, unzip_data
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.wsicore import WSIReader

RNG = np.random.default_rng()  # Numpy Random Generator


class Proto1(DatasetInfoABC):
    """Intentionally created to check error with new attribute a."""

    def __init__(self: Proto1) -> None:
        """Proto1 initialization."""
        self.a = "a"


class Proto2(DatasetInfoABC):
    """Intentionally created to check error with attribute inputs."""

    def __init__(self: Proto2) -> None:
        """Proto2 initialization."""
        self.inputs = "a"


class Proto3(DatasetInfoABC):
    """Intentionally created to check error with attribute inputs and labels."""

    def __init__(self: Proto3) -> None:
        """Proto3 initialization."""
        self.inputs = "a"
        self.labels = "a"


class Proto4(DatasetInfoABC):
    """Intentionally created to check error with attribute inputs and label names."""

    def __init__(self: Proto4) -> None:
        """Proto4 initialization."""
        self.inputs = "a"
        self.label_names = "a"


def test_dataset_abc() -> None:
    """Test for ABC."""
    # test defining a subclass of dataset info but not defining
    # enforcing attributes - should crash
    with pytest.raises(TypeError):
        Proto1()  # skipcq
    with pytest.raises(TypeError):
        Proto2()  # skipcq
    with pytest.raises(TypeError):
        Proto3()  # skipcq
    with pytest.raises(TypeError):
        Proto4()  # skipcq


@pytest.mark.skipif(toolbox_env.running_on_ci(), reason="Local test on local machine.")
def test_kather_dataset_default() -> None:
    """Test for kather patch dataset with default parameters."""
    # test Kather with default init
    dataset_path = rcParam["TIATOOLBOX_HOME"] / "dataset" / "kather100k-validation"
    shutil.rmtree(dataset_path, ignore_errors=True)

    _ = KatherPatchDataset()
    # kather with default data path skip download
    _ = KatherPatchDataset()

    # remove generated data
    shutil.rmtree(dataset_path, ignore_errors=False)


def test_kather_nonexisting_dir() -> None:
    """Pytest for not exist dir."""
    with pytest.raises(
        ValueError,
        match=r".*not exist.*",
    ):
        _ = KatherPatchDataset(save_dir_path="non-existing-path")


def test_kather_dataset(tmp_path: Path) -> None:
    """Test for kather patch dataset."""
    save_dir_path = tmp_path

    # save to temporary location
    # remove previously generated data
    if Path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    url = (
        "https://tiatoolbox.dcs.warwick.ac.uk/datasets"
        "/kather100k-train-nonorm-subset-90.zip"
    )
    save_zip_path = save_dir_path / "Kather.zip"
    download_data(url, save_path=save_zip_path)
    unzip_data(save_zip_path, save_dir_path)
    extracted_dir = save_dir_path / "NCT-CRC-HE-100K-NONORM/"
    dataset = KatherPatchDataset(save_dir_path=extracted_dir)
    assert dataset.inputs is not None
    assert dataset.labels is not None
    assert dataset.label_names is not None
    assert len(dataset.inputs) == len(dataset.labels)

    # to actually get the image, we feed it to PatchDataset
    actual_ds = PatchDataset(dataset.inputs, dataset.labels)
    sample_patch = actual_ds[89]
    assert isinstance(sample_patch["image"], np.ndarray)
    assert sample_patch["label"] is not None

    # remove generated data
    shutil.rmtree(save_dir_path, ignore_errors=True)


def test_patch_dataset_path_imgs(
    sample_patch1: str | Path,
    sample_patch2: str | Path,
) -> None:
    """Test for patch dataset with a list of file paths as input."""
    size = (224, 224, 3)

    dataset = PatchDataset([Path(sample_patch1), Path(sample_patch2)])

    for _, sample_data in enumerate(dataset):
        sampled_img_shape = sample_data["image"].shape
        assert sampled_img_shape[0] == size[0]
        assert sampled_img_shape[1] == size[1]
        assert sampled_img_shape[2] == size[2]


def test_patch_dataset_list_imgs(tmp_path: Path) -> None:
    """Test for patch dataset with a list of images as input."""
    save_dir_path = tmp_path

    size = (5, 5, 3)
    img = RNG.integers(low=0, high=255, size=size)
    list_imgs = [img, img, img]
    dataset = PatchDataset(list_imgs)

    dataset.preproc_func = lambda x: x

    for _, sample_data in enumerate(dataset):
        sampled_img_shape = sample_data["image"].shape
        assert sampled_img_shape[0] == size[0]
        assert sampled_img_shape[1] == size[1]
        assert sampled_img_shape[2] == size[2]

    # test for changing to another preproc
    dataset.preproc_func = lambda x: x - 10
    item = dataset[0]
    assert np.sum(item["image"] - (list_imgs[0] - 10)) == 0

    # * test for loading npy
    # remove previously generated data
    if Path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    Path.mkdir(save_dir_path, parents=True)
    np.save(
        str(save_dir_path / "sample2.npy"),
        RNG.integers(0, 255, (4, 4, 3)),
    )
    imgs = [
        save_dir_path / "sample2.npy",
    ]
    _ = PatchDataset(imgs)
    assert imgs[0] is not None
    # test for path object
    imgs = [
        save_dir_path / "sample2.npy",
    ]
    _ = PatchDataset(imgs)


def test_patch_datasetarray_imgs() -> None:
    """Test for patch dataset with a numpy array of a list of images."""
    size = (5, 5, 3)
    img = RNG.integers(0, 255, size=size)
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
        assert sampled_img_shape[0] == size[0]
        assert sampled_img_shape[1] == size[1]
        assert sampled_img_shape[2] == size[2]


def test_patch_dataset_crash(tmp_path: Path) -> None:
    """Test to make sure patch dataset crashes with incorrect input."""
    # all below examples should fail when input to PatchDataset
    save_dir_path = tmp_path

    # not supported input type
    imgs = {"a": RNG.integers(0, 255, (4, 4, 4))}
    with pytest.raises(
        ValueError,
        match=r".*Input must be either a list/array of images.*",
    ):
        _ = PatchDataset(imgs)

    # ndarray of mixed dtype
    imgs = np.array(
        # string array of the same shape
        [
            RNG.integers(0, 255, (4, 5, 3)),
            np.array(  # skipcq: PYL-E1121
                ["PatchDataset should crash here" for _ in range(4 * 5 * 3)],
            ).reshape(
                4,
                5,
                3,
            ),
        ],
        dtype=object,
    )
    with pytest.raises(ValueError, match="Provided input array is non-numerical."):
        _ = PatchDataset(imgs)

    # ndarray(s) of NHW images
    imgs = RNG.integers(0, 255, (4, 4, 4))
    with pytest.raises(ValueError, match=r".*array of the form HWC*"):
        _ = PatchDataset(imgs)

    # list of ndarray(s) with different sizes
    imgs = [
        RNG.integers(0, 255, (4, 4, 3)),
        RNG.integers(0, 255, (4, 5, 3)),
    ]
    with pytest.raises(ValueError, match="Images must have the same dimensions."):
        _ = PatchDataset(imgs)

    # list of ndarray(s) with HW and HWC mixed up
    imgs = [
        RNG.integers(0, 255, (4, 4, 3)),
        RNG.integers(0, 255, (4, 4)),
    ]
    with pytest.raises(
        ValueError,
        match="Each sample must be an array of the form HWC.",
    ):
        _ = PatchDataset(imgs)

    # list of mixed dtype
    imgs = [RNG.integers(0, 255, (4, 4, 3)), "you_should_crash_here", 123, 456]
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
    shutil.rmtree(save_dir_path, ignore_errors=True)
    save_dir_path.mkdir(parents=True)

    torch.save({"a": "a"}, save_dir_path / "sample1.tar")
    np.save(
        str(save_dir_path / "sample2.npy"),
        RNG.integers(0, 255, (4, 4, 3)),
    )

    imgs = [
        save_dir_path / "sample1.tar",
        save_dir_path / "sample2.npy",
    ]
    with pytest.raises(
        TypeError,
        match="Cannot load image data from",
    ):
        _ = PatchDataset(imgs)

    # preproc func for not defined dataset
    with pytest.raises(
        ValueError,
        match=r".* preprocessing .* does not exist.",
    ):
        predefined_preproc_func("secret-dataset")


def test_wsi_patch_dataset(  # noqa: PLR0915
    sample_wsi_dict: dict,
    tmp_path: Path,
) -> None:
    """A test for creation and bare output."""
    # convert to pathlib Path to prevent wsireader complaint
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    mini_wsi_jpg = Path(sample_wsi_dict["wsi2_4k_4k_jpg"])
    mini_wsi_msk = Path(sample_wsi_dict["wsi2_4k_4k_msk"])

    def reuse_init(img_path: Path = mini_wsi_svs, **kwargs: dict) -> WSIPatchDataset:
        """Testing function."""
        return WSIPatchDataset(img_path=img_path, **kwargs)

    def reuse_init_wsi(**kwargs: dict) -> WSIPatchDataset:
        """Testing function."""
        return reuse_init(mode="wsi", **kwargs)

    # test for ABC validate
    # intentionally created to check error
    # skipcq
    class Proto(PatchDatasetABC):
        def __init__(self: Proto) -> None:
            super().__init__()
            self.inputs = "CRASH"
            self._check_input_integrity("wsi")

        # skipcq
        def __getitem__(self: Proto, idx: int) -> object:
            """Get an item from the dataset."""

    with pytest.raises(
        ValueError,
        match=r".*`inputs` should be a list of patch coordinates.*",
    ):
        Proto()  # skipcq

    # invalid path input
    with pytest.raises(ValueError, match=r".*`img_path` must be a valid file path.*"):
        WSIPatchDataset(
            img_path="aaaa",
            mode="wsi",
            patch_input_shape=[512, 512],
            stride_shape=[256, 256],
            auto_get_mask=False,
        )

    # invalid mask path input
    with pytest.raises(ValueError, match=r".*`mask_path` must be a valid file path.*"):
        WSIPatchDataset(
            img_path=mini_wsi_svs,
            mask_path="aaaa",
            mode="wsi",
            patch_input_shape=[512, 512],
            stride_shape=[256, 256],
            resolution=1.0,
            units="mpp",
            auto_get_mask=False,
        )

    # invalid mode
    with pytest.raises(ValueError, match="`X` is not supported."):
        reuse_init(mode="X")

    # invalid patch
    with pytest.raises(ValueError, match="Invalid `patch_input_shape` value None."):
        reuse_init()
    with pytest.raises(
        ValueError,
        match=r"Invalid `patch_input_shape` value \[512 512 512\].",
    ):
        reuse_init_wsi(patch_input_shape=[512, 512, 512])
    with pytest.raises(
        ValueError,
        match=r"Invalid `patch_input_shape` value \['512' 'a'\].",
    ):
        reuse_init_wsi(patch_input_shape=[512, "a"])
    with pytest.raises(ValueError, match="Invalid `stride_shape` value None."):
        reuse_init_wsi(patch_input_shape=512)
    # invalid stride
    with pytest.raises(
        ValueError,
        match=r"Invalid `stride_shape` value \['512' 'a'\].",
    ):
        reuse_init_wsi(patch_input_shape=[512, 512], stride_shape=[512, "a"])
    with pytest.raises(
        ValueError,
        match=r"Invalid `stride_shape` value \[512 512 512\].",
    ):
        reuse_init_wsi(patch_input_shape=[512, 512], stride_shape=[512, 512, 512])
    # negative
    with pytest.raises(
        ValueError,
        match=r"Invalid `patch_input_shape` value \[ 512 -512\].",
    ):
        reuse_init_wsi(patch_input_shape=[512, -512], stride_shape=[512, 512])
    with pytest.raises(
        ValueError,
        match=r"Invalid `stride_shape` value \[ 512 -512\].",
    ):
        reuse_init_wsi(patch_input_shape=[512, 512], stride_shape=[512, -512])

    # * for wsi
    # dummy test for analysing the output
    # stride and patch size should be as expected
    patch_size = [512, 512]
    stride_size = [256, 256]
    ds = reuse_init_wsi(
        patch_input_shape=patch_size,
        stride_shape=stride_size,
        resolution=1.0,
        units="mpp",
        auto_get_mask=False,
    )
    reader = WSIReader.open(mini_wsi_svs)
    # tiling top to bottom, left to right
    ds_roi = ds[2]["image"]
    step_idx = 2  # manually calibrate
    start = (step_idx * stride_size[1], 0)
    end = (start[0] + patch_size[0], start[1] + patch_size[1])
    rd_roi = reader.read_bounds(
        start + end,
        resolution=1.0,
        units="mpp",
        coord_space="resolution",
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
        patch_input_shape=patch_size,
        stride_shape=stride_size,
        resolution=1.0,
        units="mpp",
        auto_get_mask=True,
    )
    assert len(ds) > 0
    ds = WSIPatchDataset(
        img_path=mini_wsi_svs,
        mask_path=mini_wsi_msk,
        mode="wsi",
        patch_input_shape=[512, 512],
        stride_shape=[256, 256],
        auto_get_mask=False,
        resolution=1.0,
        units="mpp",
    )
    negative_mask = imread(mini_wsi_msk)
    negative_mask = np.zeros_like(negative_mask)
    negative_mask_path = tmp_path / "negative_mask.png"
    imwrite(negative_mask_path, negative_mask)
    with pytest.raises(ValueError, match="No patch coordinates remain after filtering"):
        ds = WSIPatchDataset(
            img_path=mini_wsi_svs,
            mask_path=negative_mask_path,
            mode="wsi",
            patch_input_shape=[512, 512],
            stride_shape=[256, 256],
            auto_get_mask=False,
            resolution=1.0,
            units="mpp",
        )

    # * for tile
    reader = WSIReader.open(mini_wsi_jpg)
    tile_ds = WSIPatchDataset(
        img_path=mini_wsi_jpg,
        mode="tile",
        patch_input_shape=patch_size,
        stride_shape=stride_size,
        auto_get_mask=False,
    )
    step_idx = 3  # manually calibrate
    start = (step_idx * stride_size[1], 0)
    end = (start[0] + patch_size[0], start[1] + patch_size[1])
    roi2 = reader.read_bounds(
        start + end,
        resolution=1.0,
        units="baseline",
        coord_space="resolution",
    )
    roi1 = tile_ds[3]["image"]  # match with step_index
    correlation = np.corrcoef(
        cv2.cvtColor(roi1, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(roi2, cv2.COLOR_RGB2GRAY).flatten(),
    )
    assert roi1.shape[0] == roi2.shape[0]
    assert roi1.shape[1] == roi2.shape[1]
    assert np.min(correlation) > 0.9, correlation


def test_patch_dataset_abc() -> None:
    """Test for ABC methods.

    Test missing definition for abstract intentionally created to check error.

    """

    # skipcq
    class Proto(PatchDatasetABC):
        # skipcq
        def __init__(self: Proto) -> None:
            super().__init__()

    # crash due to undefined __getitem__
    with pytest.raises(TypeError):
        Proto()  # skipcq

    # skipcq
    class Proto(PatchDatasetABC):
        # skipcq
        def __init__(self: Proto) -> None:
            super().__init__()

        # skipcq
        def __getitem__(self: Proto, idx: int) -> None:
            """Get an item from the dataset."""

    ds = Proto()  # skipcq

    # test setter and getter
    assert ds.preproc_func(1) == 1
    ds.preproc_func = lambda x: x - 1  # skipcq: PYL-W0201
    assert ds.preproc_func(1) == 0
    assert ds.preproc(1) == 1, "Must be unchanged!"
    ds.preproc_func = None  # skipcq: PYL-W0201
    assert ds.preproc_func(2) == 2

    # test assign uncallable to preproc_func/postproc_func
    with pytest.raises(ValueError, match=r".*callable*"):
        ds.preproc_func = 1  # skipcq: PYL-W0201
