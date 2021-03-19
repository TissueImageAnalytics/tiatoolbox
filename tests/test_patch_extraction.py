from tiatoolbox.tools import patchextraction
from tiatoolbox.utils.exceptions import MethodNotSupported, FileNotSupported
from tiatoolbox.utils.misc import imread
from tiatoolbox.dataloader.wsireader import (
    OpenSlideWSIReader,
    OmnyxJP2WSIReader,
    VirtualWSIReader,
)

import pytest
import pathlib
import numpy as np


def read_points_patches(input_img, locations_list):
    patches = patchextraction.get_patch_extractor(
        input_img=input_img,
        locations_list=locations_list,
        method_name="point",
        patch_size=(20, 20),
    )

    data = np.empty([3, 20, 20, 3])
    data[0] = next(patches)
    data[1] = next(patches)
    data[2] = patches[23]

    return data


def test_get_patch_extractor():
    """Test get_patch_extractor returns the right object."""
    file_parent_dir = pathlib.Path(__file__).parent
    input_img = imread(file_parent_dir.joinpath("data/source_image.png"))
    locations_list = file_parent_dir.joinpath("data/sample_patch_extraction.csv")
    points = patchextraction.get_patch_extractor(
        input_img=input_img,
        locations_list=locations_list,
        method_name="point",
        patch_size=(200, 200),
    )

    assert isinstance(points, patchextraction.PointsPatchExtractor)

    with pytest.raises(MethodNotSupported):
        points.merge_patches()

    fixed_window = patchextraction.get_patch_extractor(
        input_img=input_img,
        method_name="fixedwindow",
        patch_size=(200, 200),
    )

    assert isinstance(fixed_window, patchextraction.FixedWindowPatchExtractor)

    variable_window = patchextraction.get_patch_extractor(
        input_img=input_img,
        method_name="variablewindow",
        patch_size=(200, 200),
    )

    assert isinstance(variable_window, patchextraction.VariableWindowPatchExtractor)

    with pytest.raises(MethodNotSupported):
        patchextraction.get_patch_extractor("unknown")


def test_points_patch_extractor_image_format(_sample_svs, _sample_jp2):
    """Test PointsPatchExtractor returns the right object."""
    file_parent_dir = pathlib.Path(__file__).parent
    input_img = pathlib.Path(file_parent_dir.joinpath("data/source_image.png"))
    locations_list = file_parent_dir.joinpath("data/sample_patch_extraction.csv")
    points = patchextraction.get_patch_extractor(
        input_img=input_img,
        locations_list=locations_list,
        method_name="point",
        patch_size=(200, 200),
    )

    assert isinstance(points.wsi, VirtualWSIReader)

    with pytest.raises(MethodNotSupported):
        points.merge_patches(patches=None)

    points = patchextraction.get_patch_extractor(
        input_img=pathlib.Path(_sample_svs),
        locations_list=locations_list,
        method_name="point",
        patch_size=(200, 200),
    )

    assert isinstance(points.wsi, OpenSlideWSIReader)

    points = patchextraction.get_patch_extractor(
        input_img=pathlib.Path(_sample_jp2),
        locations_list=locations_list,
        method_name="point",
        patch_size=(200, 200),
    )

    assert isinstance(points.wsi, OmnyxJP2WSIReader)

    with pytest.raises(FileNotSupported):
        false_image = pathlib.Path(file_parent_dir.joinpath("data/source_image.test"))
        _ = patchextraction.get_patch_extractor(
            input_img=false_image,
            locations_list=locations_list,
            method_name="point",
            patch_size=(200, 200),
        )


def test_points_patch_extractor():
    """Test PointsPatchExtractor returns the right value."""
    file_parent_dir = pathlib.Path(__file__).parent
    input_img = pathlib.Path(
        file_parent_dir.joinpath("data/TCGA-HE-7130-01Z-00-DX1.png")
    )
    saved_data = np.load(
        file_parent_dir.joinpath("data/sample_patch_extraction_read.npy")
    )

    locations_list = file_parent_dir.joinpath("data/sample_patch_extraction.csv")
    data = read_points_patches(input_img, locations_list)

    assert np.all(data == saved_data)

    locations_list = file_parent_dir.joinpath("data/sample_patch_extraction.npy")
    data = read_points_patches(input_img, locations_list)

    assert np.all(data == saved_data)

    locations_list = file_parent_dir.joinpath("data/sample_patch_extraction.json")
    data = read_points_patches(input_img, locations_list)

    assert np.all(data == saved_data)

    locations_list = file_parent_dir.joinpath(
        "data/sample_patch_extraction-noheader.csv"
    )
    data = read_points_patches(input_img, locations_list)

    assert np.all(data == saved_data)
