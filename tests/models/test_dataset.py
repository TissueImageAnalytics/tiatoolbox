"""Test for predefined dataset within toolbox."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from tiatoolbox import rcParam
from tiatoolbox.models.dataset import DatasetInfoABC, KatherPatchDataset, PatchDataset
from tiatoolbox.utils import download_data, unzip_data
from tiatoolbox.utils import env_detection as toolbox_env


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
