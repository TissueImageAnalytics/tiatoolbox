"""Tests for code related to patch extraction."""

import pathlib

import numpy as np
import pytest

from tiatoolbox.tools import patchextraction
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils import misc
from tiatoolbox.utils.exceptions import FileNotSupported, MethodNotSupported
from tiatoolbox.wsicore.wsireader import (
    OmnyxJP2WSIReader,
    OpenSlideWSIReader,
    VirtualWSIReader,
)


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
        pad_mode="constant",
        pad_constant_values=255,
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

    patches.n = 1870
    with pytest.raises(StopIteration):
        # skipcq
        next(patches)

    with pytest.raises(IndexError):
        print(patches[1870])

    with pytest.raises(TypeError):
        print(patches[1.0])

    return data


def test_patch_extractor(source_image):
    """Test base class patch extractor."""
    input_img = misc.imread(pathlib.Path(source_image))
    patches = patchextraction.PatchExtractor(input_img=input_img, patch_size=(20, 20))
    next_patches = iter(patches)
    assert next_patches.n == 0


def test_get_patch_extractor(source_image, patch_extr_csv):
    """Test get_patch_extractor returns the right object."""
    input_img = misc.imread(pathlib.Path(source_image))
    locations_list = pathlib.Path(patch_extr_csv)
    points = patchextraction.get_patch_extractor(
        input_img=input_img,
        locations_list=locations_list,
        method_name="point",
        patch_size=(200, 200),
    )

    assert isinstance(points, patchextraction.PointsPatchExtractor)

    sliding_window = patchextraction.get_patch_extractor(
        input_img=input_img,
        method_name="slidingwindow",
        patch_size=(200, 200),
    )

    assert isinstance(sliding_window, patchextraction.SlidingWindowPatchExtractor)

    with pytest.raises(MethodNotSupported):
        patchextraction.get_patch_extractor("unknown")


def test_points_patch_extractor_image_format(
    sample_svs, sample_jp2, source_image, patch_extr_csv
):
    """Test PointsPatchExtractor returns the right object."""
    file_parent_dir = pathlib.Path(__file__).parent
    locations_list = pathlib.Path(patch_extr_csv)

    points = patchextraction.get_patch_extractor(
        input_img=pathlib.Path(source_image),
        locations_list=locations_list,
        method_name="point",
        patch_size=(200, 200),
    )

    assert isinstance(points.wsi, VirtualWSIReader)

    points = patchextraction.get_patch_extractor(
        input_img=pathlib.Path(sample_svs),
        locations_list=locations_list,
        method_name="point",
        patch_size=(200, 200),
    )

    assert isinstance(points.wsi, OpenSlideWSIReader)

    points = patchextraction.get_patch_extractor(
        input_img=pathlib.Path(sample_jp2),
        locations_list=locations_list,
        method_name="point",
        patch_size=(200, 200),
    )

    assert isinstance(points.wsi, OmnyxJP2WSIReader)

    false_image = pathlib.Path(file_parent_dir.joinpath("data/source_image.test"))
    with pytest.raises(FileNotSupported):
        _ = patchextraction.get_patch_extractor(
            input_img=false_image,
            locations_list=locations_list,
            method_name="point",
            patch_size=(200, 200),
        )


def test_points_patch_extractor(
    patch_extr_vf_image,
    patch_extr_npy_read,
    patch_extr_csv,
    patch_extr_npy,
    patch_extr_2col_npy,
    patch_extr_json,
    patch_extr_csv_noheader,
):
    """Test PointsPatchExtractor for VirtualWSIReader."""
    input_img = pathlib.Path(patch_extr_vf_image)

    saved_data = np.load(str(pathlib.Path(patch_extr_npy_read)))

    locations_list = pathlib.Path(patch_extr_csv)
    data = read_points_patches(input_img, locations_list, item=23)

    assert np.all(data == saved_data)

    locations_list = pathlib.Path(patch_extr_npy)
    data = read_points_patches(input_img, locations_list, item=23)

    assert np.all(data == saved_data)

    locations_list = pathlib.Path(patch_extr_2col_npy)
    data = read_points_patches(input_img, locations_list, item=23)

    assert np.all(data == saved_data)

    locations_list = pathlib.Path(patch_extr_json)
    data = read_points_patches(input_img, locations_list, item=23)

    assert np.all(data == saved_data)

    locations_list = pathlib.Path(patch_extr_csv_noheader)
    data = read_points_patches(input_img, locations_list, item=23)

    assert np.all(data == saved_data)


def test_points_patch_extractor_svs(
    sample_svs, patch_extr_svs_csv, patch_extr_svs_npy_read
):
    """Test PointsPatchExtractor for svs image."""
    locations_list = pathlib.Path(patch_extr_svs_csv)
    saved_data = np.load(str(pathlib.Path(patch_extr_svs_npy_read)))

    data = read_points_patches(
        pathlib.Path(sample_svs),
        locations_list,
        item=2,
        patch_size=(100, 100),
        units="power",
        resolution=2,
    )

    assert np.all(data == saved_data)


def test_points_patch_extractor_jp2(
    sample_jp2, patch_extr_jp2_csv, patch_extr_jp2_read
):
    """Test PointsPatchExtractor for jp2 image."""
    locations_list = pathlib.Path(patch_extr_jp2_csv)
    saved_data = np.load(str(pathlib.Path(patch_extr_jp2_read)))

    data = read_points_patches(
        pathlib.Path(sample_jp2),
        locations_list,
        item=2,
        patch_size=(100, 100),
        units="power",
        resolution=2.5,
    )

    assert np.all(data == saved_data)


def test_sliding_windowpatch_extractor(patch_extr_vf_image):
    """Test SlidingWindowPatchExtractor for VF."""
    input_img = pathlib.Path(patch_extr_vf_image)

    stride = (20, 20)
    patch_size = (200, 200)
    img = misc.imread(input_img)

    img_h = img.shape[0]
    img_w = img.shape[1]

    coord_list = PatchExtractor.get_coordinates(
        image_shape=(img_w, img_h),
        patch_input_shape=patch_size,
        stride_shape=stride,
        input_within_bound=True,
    )

    num_patches_img = len(coord_list)
    img_patches = np.zeros(
        (num_patches_img, patch_size[1], patch_size[0], 3), dtype=img.dtype
    )

    for i, coord in enumerate(coord_list):
        start_w, start_h, end_w, end_h = coord
        img_patches[i, :, :, :] = img[start_h:end_h, start_w:end_w, :]

    patches = patchextraction.get_patch_extractor(
        input_img=input_img,
        method_name="slidingwindow",
        patch_size=patch_size,
        resolution=0,
        units="level",
        stride=stride,
        within_bound=True,
    )

    assert np.all(img_patches[0] == patches[0])

    img_patches_test = []
    for patch in patches:
        img_patches_test.append(patch)

    img_patches_test = np.array(img_patches_test)
    assert np.all(img_patches == img_patches_test)


def test_get_coordinates():
    """Test get tile cooordinates functionality."""
    expected_output = np.array(
        [
            [0, 0, 4, 4],
            [4, 0, 8, 4],
        ]
    )
    output = PatchExtractor.get_coordinates(
        image_shape=[9, 6],
        patch_input_shape=[4, 4],
        stride_shape=[4, 4],
        input_within_bound=True,
    )
    assert np.sum(expected_output - output) == 0

    expected_output = np.array(
        [
            [0, 0, 4, 4],
            [0, 4, 4, 8],
            [4, 0, 8, 4],
            [4, 4, 8, 8],
            [8, 0, 12, 4],
            [8, 4, 12, 8],
        ]
    )
    output = PatchExtractor.get_coordinates(
        image_shape=[9, 6],
        patch_input_shape=[4, 4],
        stride_shape=[4, 4],
        input_within_bound=False,
    )
    assert np.sum(expected_output - output) == 0
    # test when patch shape is larger than image
    output = PatchExtractor.get_coordinates(
        image_shape=[9, 6],
        patch_input_shape=[9, 9],
        stride_shape=[9, 9],
        input_within_bound=False,
    )
    # test when output patch shape is out of bound
    # but input is in bound
    input_bounds, output_bounds = PatchExtractor.get_coordinates(
        image_shape=[9, 6],
        patch_input_shape=[5, 5],
        patch_output_shape=[4, 4],
        stride_shape=[4, 4],
        output_within_bound=True,
        input_within_bound=False,
    )
    assert len(input_bounds) == 2
    assert len(output_bounds) == 2
    # test when patch shape is larger than image
    output = PatchExtractor.get_coordinates(
        image_shape=[9, 6],
        patch_input_shape=[9, 9],
        stride_shape=[9, 8],
        input_within_bound=True,
    )
    assert len(output) == 0

    # test error input form
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            image_shape=[9j, 6],
            patch_input_shape=[4, 4],
            stride_shape=[4, 4],
            input_within_bound=False,
        )
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            image_shape=[9, 6],
            patch_input_shape=[4, 4],
            stride_shape=[4, 4j],
            input_within_bound=False,
        )
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            image_shape=[9, 6],
            patch_input_shape=[4j, 4],
            stride_shape=[4, 4],
            input_within_bound=False,
        )
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            image_shape=[9, 6],
            patch_input_shape=[4, -1],
            stride_shape=[4, 4],
            input_within_bound=False,
        )
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            image_shape=[9, -6],
            patch_input_shape=[4, -1],
            stride_shape=[4, 4],
            input_within_bound=False,
        )
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            image_shape=[9, 6, 3],
            patch_input_shape=[4, 4],
            stride_shape=[4, 4],
            input_within_bound=False,
        )
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            image_shape=[9, 6],
            patch_input_shape=[4, 4, 3],
            stride_shape=[4, 4],
            input_within_bound=False,
        )
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            image_shape=[9, 6],
            patch_input_shape=[4, 4],
            stride_shape=[4, 4, 3],
            input_within_bound=False,
        )
    with pytest.raises(ValueError, match=r"stride.*> 1.*"):
        PatchExtractor.get_coordinates(
            image_shape=[9, 6],
            patch_input_shape=[4, 4],
            stride_shape=[0, 0],
            input_within_bound=False,
        )
    # * invalid shape for output
    with pytest.raises(ValueError, match=r".*input.*larger.*output.*"):
        PatchExtractor.get_coordinates(
            image_shape=[9, 6],
            stride_shape=[4, 4],
            patch_input_shape=[2, 2],
            patch_output_shape=[4, 4],
            input_within_bound=False,
        )
    with pytest.raises(ValueError, match=r"Invalid.*shape.*"):
        PatchExtractor.get_coordinates(
            image_shape=[9, 6],
            stride_shape=[4, 4],
            patch_input_shape=[4, 4],
            patch_output_shape=[2, -2],
            input_within_bound=False,
        )


def test_filter_coordinates():
    """Test different coordinate filtering functions for patch extraction"""
    bbox_list = np.array(
        [
            [0, 0, 4, 4],
            [0, 4, 4, 8],
            [4, 0, 8, 4],
            [4, 4, 8, 8],
            [8, 0, 12, 4],
            [8, 4, 12, 8],
        ]
    )
    mask = np.zeros([9, 6])
    mask[0:4, 3:8] = 1  # will flag first 2
    mask_reader = VirtualWSIReader(mask)

    # Tests for (original) filter_coordinates method
    # functionality test
    flag_list = PatchExtractor.filter_coordinates(
        mask_reader, bbox_list, resolution=1.0, units="baseline"
    )
    assert np.sum(flag_list - np.array([1, 1, 0, 0, 0, 0])) == 0

    # Test for bad mask input
    with pytest.raises(
        ValueError, match="`mask_reader` should be wsireader.VirtualWSIReader."
    ):
        PatchExtractor.filter_coordinates(
            mask, bbox_list, resolution=1.0, units="baseline"
        )

    # Test for bad bbox coordinate list in the input
    with pytest.raises(ValueError, match=r".*should be ndarray of integer type.*"):
        PatchExtractor.filter_coordinates(
            mask_reader, bbox_list.tolist(), resolution=1.0, units="baseline"
        )

    # Test for incomplete coordinate list
    with pytest.raises(ValueError, match=r".*`coordinates_list` of shape.*"):
        PatchExtractor.filter_coordinates(
            mask_reader, bbox_list[:, :2], resolution=1.0, units="baseline"
        )

    ###############################################
    # Tests for filter_coordinates_fast (new) method
    _info = mask_reader.info
    _info.mpp = 1.0
    mask_reader._m_info = _info

    # functionality test
    flag_list = PatchExtractor.filter_coordinates_fast(
        mask_reader,
        bbox_list,
        coordinate_resolution=1.0,
        coordinate_units="mpp",
        mask_resolution=1,
    )
    assert np.sum(flag_list - np.array([1, 1, 0, 0, 0, 0])) == 0
    flag_list = PatchExtractor.filter_coordinates_fast(
        mask_reader, bbox_list, coordinate_resolution=(1.0, 1.0), coordinate_units="mpp"
    )

    # Test for bad mask input
    with pytest.raises(
        ValueError, match="`mask_reader` should be wsireader.VirtualWSIReader."
    ):
        PatchExtractor.filter_coordinates_fast(
            mask,
            bbox_list,
            coordinate_resolution=1.0,
            coordinate_units="mpp",
        )

    # Test for bad bbox coordinate list in the input
    with pytest.raises(ValueError, match=r".*should be ndarray of integer type.*"):
        PatchExtractor.filter_coordinates_fast(
            mask_reader,
            bbox_list.tolist(),
            coordinate_resolution=1,
            coordinate_units="mpp",
        )

    # Test for incomplete coordinate list
    with pytest.raises(ValueError, match=r".*`coordinates_list` must be of shape.*"):
        PatchExtractor.filter_coordinates_fast(
            mask_reader,
            bbox_list[:, :2],
            coordinate_resolution=1,
            coordinate_units="mpp",
        )

    # Test for put of range min_mask_ratio
    with pytest.raises(ValueError, match="`min_mask_ratio` must be between 0 and 1."):
        PatchExtractor.filter_coordinates_fast(
            mask_reader,
            bbox_list,
            coordinate_resolution=1.0,
            coordinate_units="mpp",
            min_mask_ratio=-0.5,
        )
    with pytest.raises(ValueError, match="`min_mask_ratio` must be between 0 and 1."):
        PatchExtractor.filter_coordinates_fast(
            mask_reader,
            bbox_list,
            coordinate_resolution=1.0,
            coordinate_units="mpp",
            min_mask_ratio=1.1,
        )


def test_mask_based_patch_extractor_ndpi(sample_ndpi):
    """Test SlidingWindowPatchExtractor with mask for ndpi image."""
    res = 0
    patch_size = stride = (400, 400)
    input_img = pathlib.Path(sample_ndpi)
    wsi = OpenSlideWSIReader(input_img=input_img)
    slide_dimensions = wsi.info.slide_dimensions

    # Generating a test mask to read patches from
    mask_dim = (int(slide_dimensions[0] / 10), int(slide_dimensions[1] / 10))
    wsi_mask = np.zeros(mask_dim, dtype=np.uint8)
    # masking two column to extract patch from
    wsi_mask[:, :2] = 255

    # patch extraction based on the column mask
    patches = patchextraction.get_patch_extractor(
        input_img=input_img,
        input_mask=wsi_mask,
        method_name="slidingwindow",
        patch_size=patch_size,
        resolution=res,
        units="level",
        stride=None,
    )

    # read the patch from the second row (y) in the first column
    patch = wsi.read_rect(
        location=(0, int(patch_size[1])),
        size=patch_size,
        resolution=res,
        units="level",
    )

    # because we are using column mask to extract patches, we can expect
    # that the patches[1] is the from the second row (y) in the first column.
    assert np.all(patches[1] == patch)
    assert patches[0].shape == (patch_size[0], patch_size[1], 3)

    # Test None option for mask
    patches = patchextraction.get_patch_extractor(
        input_img=input_img,
        input_mask=None,
        method_name="slidingwindow",
        patch_size=patch_size,
        resolution=res,
        units="level",
        stride=stride[0],
    )

    # Test passing a VirtualWSI for mask
    mask_wsi = VirtualWSIReader(wsi_mask, info=wsi._m_info, mode="bool")
    patches = patchextraction.get_patch_extractor(
        input_img=wsi,
        input_mask=mask_wsi,
        method_name="slidingwindow",
        patch_size=patch_size,
        resolution=res,
        units="level",
        stride=None,
    )

    # Test `otsu` option for mask
    patches = patchextraction.get_patch_extractor(
        input_img=input_img,
        input_mask="otsu",
        method_name="slidingwindow",
        patch_size=patch_size[0],
        resolution=res,
        units="level",
        stride=stride,
    )

    patches = patchextraction.get_patch_extractor(
        input_img=wsi_mask,  # a numpy array to build VirtualSlideReader
        input_mask="morphological",
        method_name="slidingwindow",
        patch_size=patch_size,
        resolution=res,
        units="level",
        stride=stride,
    )

    # Test passing an empty mask
    wsi_mask = np.zeros(mask_dim, dtype=np.uint8)
    with pytest.warns(UserWarning, match=".*No candidate coordinates left.*"):
        _ = patchextraction.get_patch_extractor(
            input_img=input_img,
            input_mask=wsi_mask,
            method_name="slidingwindow",
            patch_size=patch_size,
            resolution=res,
            units="level",
            stride=stride,
        )
