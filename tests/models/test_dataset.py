"""Tests for predefined dataset within toolbox."""

import os
import shutil

import numpy as np
import pytest

from tiatoolbox import rcParam
from tiatoolbox.models.dataset import DatasetInfoABC, KatherPatchDataset, PatchDataset
from tiatoolbox.utils import env_detection as toolbox_env
from tiatoolbox.utils.misc import download_data, unzip_data


class Proto1(DatasetInfoABC):
    """Intentionally created to check error with new attribute a."""

    def __init__(self):
        self.a = "a"


class Proto2(DatasetInfoABC):
    """Intentionally created to check error with attribute inputs."""

    def __init__(self):
        self.inputs = "a"


class Proto3(DatasetInfoABC):
    """Intentionally created to check error with attribute inputs and labels."""

    def __init__(self):
        self.inputs = "a"
        self.labels = "a"


class Proto4(DatasetInfoABC):
    """Intentionally created to check error with attribute inputs and label names."""

    def __init__(self):
        self.inputs = "a"
        self.label_names = "a"


def test_dataset_abc():
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
def test_kather_dataset_default(tmp_path):
    """Test for kather patch dataset with default parameters."""
    # test kather with default init
    _ = KatherPatchDataset()
    # kather with default data path skip download
    _ = KatherPatchDataset()

    # remove generated data
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"])


def test_kather_nonexisting_dir():
    """pytest for not exist dir."""
    with pytest.raises(
        ValueError,
        match=r".*not exist.*",
    ):
        _ = KatherPatchDataset(save_dir_path="non-existing-path")


def test_kather_dataset(tmp_path):
    """Test for kather patch dataset."""
    save_dir_path = tmp_path

    # save to temporary location
    # remove previously generated data
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    url = (
        "https://tiatoolbox.dcs.warwick.ac.uk/datasets"
        "/kather100k-train-nonorm-subset-90.zip"
    )
    save_zip_path = os.path.join(save_dir_path, "Kather.zip")
    download_data(url, save_zip_path)
    unzip_data(save_zip_path, save_dir_path)
    extracted_dir = os.path.join(save_dir_path, "NCT-CRC-HE-100K-NONORM/")
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
