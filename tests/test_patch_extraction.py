from tiatoolbox.tools import patchextraction
from tiatoolbox.utils.exceptions import MethodNotSupported, FileNotSupported
from tiatoolbox.utils import misc
from tiatoolbox.wsicore.wsireader import (
    OpenSlideWSIReader,
    OmnyxJP2WSIReader,
    VirtualWSIReader,
)

import pytest
import pathlib
import numpy as np
import math


def read_points_patches(
    input_img,
    locations_list,
    patch_size=(20, 20),
    units="level",
    resolution=0.0,
    item=2,
):
    """Read patches with the help of PointsPatchExtractor using different formats."""
    patches = patchextraction.get_patch_extractor(
        input_img=input_img,
        locations_list=locations_list,
        method_name="point",
        patch_size=patch_size,
        units=units,
        resolution=resolution,
    )

    data = np.empty([3, patch_size[0], patch_size[1], 3])
    try:
        data[0] = next(patches)
    except StopIteration:
        raise StopIteration("Index out of bounds.")

    try:
        data[1] = next(patches)
    except StopIteration:
        raise StopIteration("Index out of bounds.")

    data[2] = patches[item]

    with pytest.raises(StopIteration):
        patches.n = 1870
        try:
            next(patches)
        except StopIteration:
            raise StopIteration("Index out of bounds.")

    with pytest.raises(IndexError):
        print(patches[1870])

    with pytest.raises(TypeError):
        print(patches[1.0])

    return data


def test_patch_extractor():
    """Test base class patch extractor."""
    file_parent_dir = pathlib.Path(__file__).parent
    input_img = misc.imread(file_parent_dir.joinpath("data/source_image.png"))
    patches = patchextraction.PatchExtractor(input_img=input_img, patch_size=(20, 20))
    next_patches = iter(patches)
    assert next_patches.n == 0


def test_get_patch_extractor():
    """Test get_patch_extractor returns the right object."""
    file_parent_dir = pathlib.Path(__file__).parent
    input_img = misc.imread(file_parent_dir.joinpath("data/source_image.png"))
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
    """Test PointsPatchExtractor for VirtualWSIReader."""
    file_parent_dir = pathlib.Path(__file__).parent
    input_img = pathlib.Path(
        file_parent_dir.joinpath("data/TCGA-HE-7130-01Z-00-DX1.png")
    )
    saved_data = np.load(
        str(file_parent_dir.joinpath("data/sample_patch_extraction_read.npy"))
    )

    locations_list = file_parent_dir.joinpath("data/sample_patch_extraction.csv")
    data = read_points_patches(input_img, locations_list, item=23)

    assert np.all(data == saved_data)

    locations_list = file_parent_dir.joinpath("data/sample_patch_extraction.npy")
    data = read_points_patches(input_img, locations_list, item=23)

    assert np.all(data == saved_data)

    locations_list = file_parent_dir.joinpath("data/sample_patch_extraction_2col.npy")
    data = read_points_patches(input_img, locations_list, item=23)

    assert np.all(data == saved_data)

    locations_list = file_parent_dir.joinpath("data/sample_patch_extraction.json")
    data = read_points_patches(input_img, locations_list, item=23)

    assert np.all(data == saved_data)

    locations_list = file_parent_dir.joinpath(
        "data/sample_patch_extraction-noheader.csv"
    )
    data = read_points_patches(input_img, locations_list, item=23)

    assert np.all(data == saved_data)


def test_points_patch_extractor_svs(_sample_svs):
    """Test PointsPatchExtractor for svs image."""
    file_parent_dir = pathlib.Path(__file__).parent
    locations_list = file_parent_dir.joinpath("data/sample_patch_extraction_svs.csv")
    saved_data = np.load(
        str(file_parent_dir.joinpath("data/sample_patch_extraction_svsread.npy"))
    )

    data = read_points_patches(
        pathlib.Path(_sample_svs),
        locations_list,
        item=2,
        patch_size=(100, 100),
        units="power",
        resolution=2.5,
    )

    assert np.all(data == saved_data)


def test_points_patch_extractor_jp2(_sample_jp2):
    """Test PointsPatchExtractor for jp2 image."""
    file_parent_dir = pathlib.Path(__file__).parent
    locations_list = file_parent_dir.joinpath("data/sample_patch_extraction_jp2.csv")
    saved_data = np.load(
        str(file_parent_dir.joinpath("data/sample_patch_extraction_jp2read.npy"))
    )

    data = read_points_patches(
        pathlib.Path(_sample_jp2),
        locations_list,
        item=2,
        patch_size=(100, 100),
        units="power",
        resolution=2.5,
    )

    assert np.all(data == saved_data)


def test_fixed_window_patch_extractor():
    """Test FixedWindowPatchExtractor for VF."""
    file_parent_dir = pathlib.Path(__file__).parent
    input_img = pathlib.Path(
        file_parent_dir.joinpath("data/TCGA-HE-7130-01Z-00-DX1.png")
    )

    stride = (20, 20)
    patch_size = (200, 200)
    img = misc.imread(input_img)

    img_h = img.shape[1]
    img_w = img.shape[0]

    num_patches_img_h = int(math.ceil((img_h - patch_size[1]) / stride[1] + 1))
    num_patches_img_w = int(math.ceil(((img_w - patch_size[0]) / stride[0] + 1)))
    num_patches_img = num_patches_img_h * num_patches_img_w
    iter_tot = 0

    img_patches = np.zeros(
        (num_patches_img, patch_size[1], patch_size[0], 3), dtype=img.dtype
    )

    for h in range(num_patches_img_h):
        for w in range(num_patches_img_w):
            start_h = h * stride[1]
            end_h = (h * stride[1]) + patch_size[1]
            start_w = w * stride[0]
            end_w = (w * stride[0]) + patch_size[0]
            if end_h > img_h:
                start_h = img_h - patch_size[1]
                end_h = img_h

            if end_w > img_w:
                start_w = img_w - patch_size[0]
                end_w = img_w

            img_patches[iter_tot, :, :, :] = img[start_h:end_h, start_w:end_w, :]

            iter_tot += 1

    patches = patchextraction.get_patch_extractor(
        input_img=input_img,
        method_name="fixedwindow",
        patch_size=patch_size,
        resolution=0,
        units="level",
        stride=stride,
    )

    assert np.all(img_patches[0] == patches[0])

    img_patches_test = []
    for patch in patches:
        img_patches_test.append(patch)

    img_patches_test = np.array(img_patches_test)

    assert np.all(img_patches == img_patches_test)

    # Test for integer (single) patch_size and stride input
    patches = patchextraction.get_patch_extractor(
        input_img=input_img,
        method_name="fixedwindow",
        patch_size=patch_size[0],
        resolution=0,
        units="level",
        stride=stride[0],
    )

    assert np.all(img_patches[0] == patches[0])

    img_patches_test = []
    for patch in patches:
        img_patches_test.append(patch)

    img_patches_test = np.array(img_patches_test)

    assert np.all(img_patches == img_patches_test)


def test_fixedwindow_patch_extractor_ndpi(_sample_ndpi):
    """Test FixedWindowPatchExtractor for ndpi image."""
    stride = (20, 20)
    patch_size = (200, 200)
    input_img = pathlib.Path(_sample_ndpi)

    patches = patchextraction.get_patch_extractor(
        input_img=input_img,
        method_name="fixedwindow",
        patch_size=patch_size,
        resolution=1,
        units="level",
        stride=stride,
    )

    wsi = OpenSlideWSIReader(input_img=input_img)
    x = 200
    y = 0
    patch = wsi.read_rect(
        location=(int(x), int(y)),
        size=patch_size,
        resolution=1,
        units="level",
    )

    assert np.all(patches[10] == patch)
