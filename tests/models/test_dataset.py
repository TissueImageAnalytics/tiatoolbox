# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****
"""Tests for predefined dataset within toolbox."""

import os
import shutil

import numpy as np
import pytest

from tiatoolbox import rcParam
from tiatoolbox.models.dataset import DatasetInfoABC, KatherPatchDataset, PatchDataset
from tiatoolbox.utils.misc import download_data, unzip_data


def test_dataset_abc():
    """Test for ABC."""
    # test defining a subclass of dataset info but not defining
    # enforcing attributes - should crash
    with pytest.raises(TypeError):

        # intentionally created to check error
        # skipcq
        class Proto(DatasetInfoABC):
            def __init__(self):
                self.a = "a"

        # intentionally created to check error
        Proto()  # skipcq
    with pytest.raises(TypeError):

        # intentionally created to check error
        # skipcq
        class Proto(DatasetInfoABC):
            def __init__(self):
                self.inputs = "a"

        # intentionally created to check error
        Proto()  # skipcq
    with pytest.raises(TypeError):
        # intentionally created to check error
        # skipcq
        class Proto(DatasetInfoABC):
            def __init__(self):
                self.inputs = "a"
                self.labels = "a"

        # intentionally created to check error
        Proto()  # skipcq
    with pytest.raises(TypeError):

        # intentionally created to check error
        # skipcq
        class Proto(DatasetInfoABC):
            def __init__(self):
                self.inputs = "a"
                self.label_names = "a"

        # intentionally created to check error
        Proto()  # skipcq


@pytest.mark.skip(reason="Local test, not applicable for travis.")
def test_kather_dataset_default(tmp_path):
    """Test for kather patch dataset with default parameters."""

    # test kather with default init
    _ = KatherPatchDataset()
    # kather with default data path skip download
    _ = KatherPatchDataset()

    # remove generated data
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"])


def test_kather_nonexisting_dir():
    # pytest for not exist dir
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
