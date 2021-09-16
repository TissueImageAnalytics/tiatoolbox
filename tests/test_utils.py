"""Tests for utils."""

import os
import random
import shutil
from pathlib import Path
from typing import Tuple
import hashlib

import cv2
import numpy as np
import pandas as pd
import pytest
import torch.cuda
from PIL import Image
from pytest import approx

from tiatoolbox import rcParam, utils
from tiatoolbox.utils import misc
from tiatoolbox.utils.exceptions import FileNotSupported
from tiatoolbox.utils.transforms import locsize2bounds


def sub_pixel_read(test_image, pillow_test_image, bounds, ow, oh):
    """sub_pixel_read test helper function."""
    output = utils.image.sub_pixel_read(test_image, bounds, (ow, oh))
    assert (ow, oh) == tuple(output.shape[:2][::-1])

    output = utils.image.sub_pixel_read(
        pillow_test_image, bounds, (ow, oh), stride=[1, 1]
    )
    assert (ow, oh) == tuple(output.shape[:2][::-1])


def test_imresize():
    """Test for imresize."""
    img = np.zeros((2000, 1000, 3))
    resized_img = utils.transforms.imresize(img, scale_factor=0.5)
    assert resized_img.shape == (1000, 500, 3)

    resized_img = utils.transforms.imresize(resized_img, scale_factor=2.0)
    assert resized_img.shape == (2000, 1000, 3)

    resized_img = utils.transforms.imresize(
        img,
        scale_factor=0.5,
        interpolation=cv2.INTER_CUBIC,
    )
    assert resized_img.shape == (1000, 500, 3)


def test_imresize_no_scale_factor():
    """Test for imresize with no scale_factor given."""
    img = np.zeros((2000, 1000, 3))
    resized_img = utils.transforms.imresize(img, output_size=(50, 100))
    assert resized_img.shape == (100, 50, 3)


def test_background_composite():
    """Test for background composite."""
    new_im = np.zeros((2000, 2000, 4)).astype("uint8")
    new_im[:1000, :, 3] = 255
    im = utils.transforms.background_composite(new_im)
    assert np.all(im[1000:, :, :] == 255)
    assert np.all(im[:1000, :, :] == 0)

    im = utils.transforms.background_composite(new_im, alpha=True)
    assert np.all(im[:, :, 3] == 255)

    new_im = Image.fromarray(new_im)
    im = utils.transforms.background_composite(new_im, alpha=True)
    assert np.all(im[:, :, 3] == 255)


def test_mpp2common_objective_power(_sample_svs):
    """Test approximate conversion of mpp to objective power."""
    mapping = [
        (0.05, 100),
        (0.07, 100),
        (0.10, 100),
        (0.12, 90),
        (0.15, 60),
        (0.29, 40),
        (0.30, 40),
        (0.49, 20),
        (0.60, 20),
        (1.00, 10),
        (1.20, 10),
        (2.00, 5),
        (2.40, 4),
        (3.00, 4),
        (4.0, 2.5),
        (4.80, 2),
        (8.00, 1.25),
        (9.00, 1),
    ]
    for mpp, result in mapping:
        assert utils.misc.mpp2common_objective_power(mpp) == result
        assert np.array_equal(
            utils.misc.mpp2common_objective_power([mpp] * 2), [result] * 2
        )


def test_assert_dtype_int():
    """Test AssertionError for dtype test."""
    utils.misc.assert_dtype_int(input_var=np.array([1, 2]))
    with pytest.raises(AssertionError):
        utils.misc.assert_dtype_int(
            input_var=np.array([1.0, 2]), message="Bounds must be integers."
        )


def test_safe_padded_read_non_int_bounds():
    """Test safe_padded_read with non-integer bounds."""
    data = np.zeros((16, 16))

    bounds = (1.5, 1, 5, 5)
    with pytest.raises(ValueError):
        utils.image.safe_padded_read(data, bounds)


def test_safe_padded_read_negative_padding():
    """Test safe_padded_read with non-integer bounds."""
    data = np.zeros((16, 16))

    bounds = (1, 1, 5, 5)
    with pytest.raises(ValueError):
        utils.image.safe_padded_read(data, bounds, padding=-1)


def test_safe_padded_read_padding_formats():
    """Test safe_padded_read with different padding argument formats."""
    data = np.zeros((16, 16))
    bounds = (0, 0, 8, 8)
    stride = (1, 1)
    for padding in [1, [1], (1,), [1, 1], (1, 1), [1] * 4]:
        region = utils.image.safe_padded_read(
            data,
            bounds,
            padding=padding,
            stride=stride,
        )
        assert region.shape == (8 + 2, 8 + 2)


def test_safe_padded_read_pad_kwargs(_source_image):
    """Test passing extra kwargs to safe_padded_read for np.pad."""
    data = utils.misc.imread(str(_source_image))
    bounds = (0, 0, 8, 8)
    padding = 2
    region = utils.image.safe_padded_read(
        data,
        bounds,
        pad_mode="reflect",
        padding=padding,
    )
    even_region = utils.image.safe_padded_read(
        data,
        bounds,
        pad_mode="reflect",
        padding=padding,
        pad_kwargs={
            "reflect_type": "even",
        },
    )
    assert np.all(region == even_region)

    odd_region = utils.image.safe_padded_read(
        data,
        bounds,
        pad_mode="reflect",
        padding=padding,
        pad_kwargs={
            "reflect_type": "odd",
        },
    )
    assert not np.all(region == odd_region)


def test_safe_padded_read_pad_constant_values():
    """Test safe_padded_read with custom pad constant values.

    This test creates an image of zeros and reads the whole image with a
    padding of 1 and constant values of 10 for padding. It then checks
    for a 1px border of 10s all the way around the zeros.
    """
    for side_len in range(1, 5):
        data = np.zeros((side_len, side_len))
        bounds = (0, 0, side_len, side_len)
        padding = 1
        region = utils.image.safe_padded_read(
            data,
            bounds,
            padding=padding,
            pad_constant_values=10,
        )

        assert np.sum(region == 10) == (4 * side_len) + 4


def test_fuzz_safe_padded_read_edge_padding():
    """Fuzz test for padding at edges of an image.

    This test creates a 16x16 image with a gradient from 1 to 17 across
    it. A region is read using safe_padded_read with a constant padding
    of 0 and an offset by some random 'shift' amount between 1 and 16.
    The resulting image is checked for the correct number of 0 values.
    """
    random.seed(0)
    for _ in range(1000):
        data = np.repeat([range(1, 17)], 16, axis=0)

        # Create bounds to fit the image and shift off by one
        # randomly in x or y
        sign = (-1) ** np.random.randint(0, 1)
        axis = random.randint(0, 1)
        shift = np.tile([1 - axis, axis], 2)
        shift_magnitude = random.randint(1, 16)
        bounds = np.array([0, 0, 16, 16]) + (shift * sign * shift_magnitude)

        region = utils.image.safe_padded_read(data, bounds)

        assert np.sum(region == 0) == (16 * shift_magnitude)


def test_safe_padded_read_padding_shape():
    """Test safe_padded_read for padding shape."""
    data = np.zeros((16, 16))

    bounds = (1, 1, 5, 5)
    with pytest.raises(ValueError):
        utils.image.safe_padded_read(data, bounds, padding=(1, 1, 1))


def test_safe_padded_read_stride_shape():
    """Test safe_padded_read for padding size."""
    data = np.zeros((16, 16))

    bounds = (1, 1, 5, 5)
    with pytest.raises(ValueError):
        utils.image.safe_padded_read(data, bounds, stride=(1, 1, 1))


def test_sub_pixel_read(_source_image):
    """Test sub-pixel numpy image reads with known tricky parameters."""
    image_path = Path(_source_image)
    assert image_path.exists()
    test_image = utils.misc.imread(image_path)
    pillow_test_image = Image.fromarray(test_image)

    x = 6
    y = -4
    w = 21.805648705868652
    h = 0.9280264518437986
    bounds = (x, y, x + w, y + h)
    ow = 88
    oh = 98

    sub_pixel_read(test_image, pillow_test_image, bounds, ow, oh)

    x = 13
    y = 15
    w = 29.46
    h = 6.92
    bounds = (x, y, x + w, y + h)
    ow = 93
    oh = 34

    sub_pixel_read(test_image, pillow_test_image, bounds, ow, oh)


def test_aligned_padded_sub_pixel_read(_source_image):
    """Test sub-pixel numpy image reads with pixel-aligned bounds."""
    image_path = Path(_source_image)
    assert image_path.exists()
    test_image = utils.misc.imread(image_path)

    x = 1
    y = 1
    w = 5
    h = 5
    padding = 1
    bounds = (x, y, x + w, y + h)
    ow = 4 + 2 * padding
    oh = 4 + 2 * padding
    output = utils.image.sub_pixel_read(test_image, bounds, (ow, oh), padding=padding)
    assert (ow + 2 * padding, oh + 2 * padding) == tuple(output.shape[:2][::-1])


def test_non_aligned_padded_sub_pixel_read(_source_image):
    """Test sub-pixel numpy image reads with non-pixel-aligned bounds."""
    image_path = Path(_source_image)
    assert image_path.exists()
    test_image = utils.misc.imread(image_path)

    x = 0.5
    y = 0.5
    w = 4.5
    h = 4.5
    padding = 1
    bounds = (x, y, x + w, y + h)
    ow = 4
    oh = 4
    output = utils.image.sub_pixel_read(test_image, bounds, (ow, oh), padding=padding)

    assert (ow + 2 * padding, oh + 2 * padding) == tuple(output.shape[:2][::-1])


def test_non_baseline_padded_sub_pixel_read(_source_image):
    """Test sub-pixel numpy image reads with baseline padding."""
    image_path = Path(_source_image)
    assert image_path.exists()
    test_image = utils.misc.imread(image_path)

    x = 0.5
    y = 0.5
    w = 4.5
    h = 4.5
    padding = 1
    bounds = (x, y, x + w, y + h)
    ow = 8
    oh = 8
    output = utils.image.sub_pixel_read(
        test_image, bounds, (ow, oh), padding=padding, pad_at_baseline=True
    )
    assert (ow + 2 * 2 * padding, oh + 2 * 2 * padding) == tuple(output.shape[:2][::-1])


def test_sub_pixel_read_invalid_interpolation():
    """Test sub_pixel_read with invalid interpolation."""
    data = np.zeros((16, 16))
    out_size = data.shape

    bounds = (1.5, 1, 5, 5)
    with pytest.raises(ValueError):
        utils.image.sub_pixel_read(data, bounds, out_size, interpolation="fizz")


def test_sub_pixel_read_invalid_bounds():
    """Test sub_pixel_read with invalid bounds."""
    data = np.zeros((16, 16))
    out_size = data.shape

    bounds = (0, 0, 0, 0)
    with pytest.raises(ValueError):
        utils.image.sub_pixel_read(data, bounds, out_size)

    bounds = (1.5, 1, 1.5, 0)
    with pytest.raises(ValueError):
        utils.image.sub_pixel_read(data, bounds, out_size)


def test_sub_pixel_read_pad_at_baseline():
    """Test sub_pixel_read with baseline padding."""
    data = np.zeros((16, 16))
    out_size = data.shape
    bounds = (0, 0, 8, 8)
    for padding in range(3):
        region = utils.image.sub_pixel_read(
            data, bounds, out_size, padding=padding, pad_at_baseline=True
        )
        assert region.shape == (16 + 4 * padding, 16 + 4 * padding)

    region = utils.image.sub_pixel_read(
        data,
        bounds,
        out_size,
        pad_at_baseline=True,
        read_func=utils.image.safe_padded_read,
    )
    assert region.shape == (16, 16)


def test_sub_pixel_read_padding_formats():
    """Test sub_pixel_read with different padding argument formats."""
    data = np.zeros((16, 16))
    out_size = data.shape
    bounds = (0, 0, 8, 8)
    for padding in [1, [1], (1,), [1, 1], (1, 1), [1] * 4]:
        region = utils.image.sub_pixel_read(
            data, bounds, out_size, padding=padding, pad_at_baseline=True
        )
        assert region.shape == (16 + 4, 16 + 4)
        region = utils.image.sub_pixel_read(data, bounds, out_size, padding=padding)
        assert region.shape == (16 + 2, 16 + 2)


def test_sub_pixel_read_negative_size_bounds(_source_image):
    """Test sub_pixel_read with different padding argument formats."""
    image_path = Path(_source_image)
    assert image_path.exists()
    test_image = utils.misc.imread(image_path)

    ow = 25
    oh = 25

    x = 5
    y = 5
    w = -4.5
    h = -4.5
    bounds = locsize2bounds((x, y), (w, h))
    output = utils.image.sub_pixel_read(test_image, bounds, (ow, oh))

    x = 0.5
    y = 0.5
    w = 4.5
    h = 4.5
    bounds = locsize2bounds((x, y), (w, h))
    print(bounds)
    flipped_output = utils.image.sub_pixel_read(test_image, bounds, (ow, oh))

    assert np.all(np.fliplr(np.flipud(flipped_output)) == output)


def test_fuzz_sub_pixel_read(_source_image):
    """Fuzz test for numpy sub-pixel image reads."""
    random.seed(0)

    image_path = Path(_source_image)
    assert image_path.exists()
    test_image = utils.misc.imread(image_path)

    for _ in range(10000):
        x = random.randint(-5, 32 - 5)
        y = random.randint(-5, 32 - 5)
        w = random.random() * random.randint(1, 32)
        h = random.random() * random.randint(1, 32)
        bounds = (x, y, x + w, y + h)
        ow = random.randint(4, 128)
        oh = random.randint(4, 128)
        output = utils.image.sub_pixel_read(
            test_image,
            bounds,
            (ow, oh),
            interpolation="linear",
        )
        assert (ow, oh) == tuple(output.shape[:2][::-1])


def test_fuzz_padded_sub_pixel_read(_source_image):
    """Fuzz test for numpy sub-pixel image reads with padding."""
    random.seed(0)

    image_path = Path(_source_image)
    assert image_path.exists()
    test_image = utils.misc.imread(image_path)

    for _ in range(10000):
        x = random.randint(-5, 32 - 5)
        y = random.randint(-5, 32 - 5)
        w = random.random() * random.randint(1, 32)
        h = random.random() * random.randint(1, 32)
        padding = random.randint(0, 2)
        bounds = (x, y, x + w, y + h)
        ow = random.randint(4, 128)
        oh = random.randint(4, 128)
        output = utils.image.sub_pixel_read(
            test_image,
            bounds,
            (ow, oh),
            interpolation="linear",
            padding=padding,
        )
        assert (ow + 2 * padding, oh + 2 * padding) == tuple(output.shape[:2][::-1])


def test_sub_pixel_read_interpolation_modes():
    """Test sub_pixel_read with different padding argument formats."""
    data = np.mgrid[:16:1, :16:1].sum(0).astype(np.uint8)
    out_size = data.shape
    bounds = (0, 0, 8, 8)
    for mode in ["nearest", "linear", "cubic", "lanczos"]:
        output = utils.image.sub_pixel_read(data, bounds, out_size, interpolation=mode)
        assert output.shape == out_size


def test_sub_pixel_read_incorrect_read_func_return():
    """Test for sub pixel reading with incorrect read func return."""
    bounds = (0, 0, 8, 8)
    image = np.ones((10, 10))

    def read_func(*args, **kwargs):
        return np.ones((5, 5))

    with pytest.raises(ValueError):
        utils.image.sub_pixel_read(
            image,
            bounds=bounds,
            output_size=(10, 10),
            read_func=read_func,
        )


def test_sub_pixel_read_empty_read_func_return():
    """Test for sub pixel reading with empty read func return."""
    bounds = (0, 0, 8, 8)
    image = np.ones((10, 10))

    def read_func(*args, **kwargs):
        return np.ones((0, 0))

    with pytest.raises(ValueError):
        utils.image.sub_pixel_read(
            image,
            bounds=bounds,
            output_size=(10, 10),
            read_func=read_func,
        )


def test_sub_pixel_read_empty_bounds():
    """Test for sub pixel reading with empty bounds."""
    bounds = (0, 0, 2, 2)
    image = np.ones((10, 10))

    with pytest.raises(ValueError):
        utils.image.sub_pixel_read(
            image,
            bounds=bounds,
            output_size=(2, 2),
            padding=-1,
        )


def test_fuzz_bounds2locsize():
    """Fuzz test for bounds2size."""
    random.seed(0)
    for _ in range(1000):
        size = (random.randint(-1000, 1000), random.randint(-1000, 1000))
        location = (random.randint(-1000, 1000), random.randint(-1000, 1000))
        bounds = (*location, *(sum(x) for x in zip(size, location)))
        assert utils.transforms.bounds2locsize(bounds)[1] == approx(size)


def test_fuzz_bounds2locsize_lower():
    """Fuzz test for bounds2size with origin lower."""
    random.seed(0)
    for _ in range(1000):
        loc = (np.random.rand(2) - 0.5) * 1000
        size = (np.random.rand(2) - 0.5) * 1000
        bounds = np.tile(loc, 2) + [
            0,
            *size[::-1],
            0,
        ]  # L T R B

        _, s = utils.transforms.bounds2locsize(bounds, origin="lower")

        assert s == approx(size)


def test_fuzz_roundtrip_bounds2size():
    """Fuzz roundtrip bounds2locsize and locsize2bounds."""
    random.seed(0)
    for _ in range(1000):
        loc = (np.random.rand(2) - 0.5) * 1000
        size = (np.random.rand(2) - 0.5) * 1000
        assert utils.transforms.bounds2locsize(
            utils.transforms.locsize2bounds(loc, size)
        )


def test_bounds2size_value_error():
    """Test bounds to size ValueError."""
    with pytest.raises(ValueError):
        utils.transforms.bounds2locsize((0, 0, 1, 1), origin="middle")


def test_contrast_enhancer():
    """Test contrast enhancement functionality."""
    # input array to the contrast_enhancer function
    input_array = np.array(
        [
            [[37, 244, 193], [106, 235, 128], [71, 140, 47]],
            [[103, 184, 72], [20, 188, 238], [126, 7, 0]],
            [[137, 195, 204], [32, 203, 170], [101, 77, 133]],
        ],
        dtype=np.uint8,
    )

    # expected output of the contrast_enhancer
    result_array = np.array(
        [
            [[35, 255, 203], [110, 248, 133], [72, 146, 46]],
            [[106, 193, 73], [17, 198, 251], [131, 3, 0]],
            [[143, 205, 215], [30, 214, 178], [104, 78, 139]],
        ],
        dtype=np.uint8,
    )

    with pytest.raises(AssertionError):
        # contrast_enhancer requires image input to be of dtype uint18
        utils.misc.contrast_enhancer(np.float32(input_array), low_p=2, high_p=98)

    # calculating the contrast enhanced version of input_array
    output_array = utils.misc.contrast_enhancer(input_array, low_p=2, high_p=98)
    # the out_put array should be equal to expected seult_array
    assert np.all(result_array == output_array)


def test_load_stain_matrix(tmp_path):
    """Test to load stain matrix."""
    with pytest.raises(FileNotSupported):
        utils.misc.load_stain_matrix("/samplefile.xlsx")

    with pytest.raises(TypeError):
        # load_stain_matrix requires numpy array as input providing list here
        utils.misc.load_stain_matrix([1, 2, 3])

    stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
    pd.DataFrame(stain_matrix).to_csv(Path(tmp_path).joinpath("sm.csv"), index=False)
    out_stain_matrix = utils.misc.load_stain_matrix(Path(tmp_path).joinpath("sm.csv"))
    assert np.all(out_stain_matrix == stain_matrix)

    np.save(str(Path(tmp_path).joinpath("sm.npy")), stain_matrix)
    out_stain_matrix = utils.misc.load_stain_matrix(Path(tmp_path).joinpath("sm.npy"))
    assert np.all(out_stain_matrix == stain_matrix)


def test_get_luminosity_tissue_mask():
    """Test get luminosity tissue mask."""
    with pytest.raises(ValueError):
        utils.misc.get_luminosity_tissue_mask(img=np.zeros((100, 100, 3)), threshold=0)


def test_read_point_annotations(
    tmp_path,
    _patch_extr_csv,
    _patch_extr_csv_noheader,
    _patch_extr_svs_csv,
    _patch_extr_svs_header,
    _patch_extr_npy,
    _patch_extr_json,
    _patch_extr_2col_json,
):
    """Test read point annotations reads csv, ndarray, npy and json correctly."""
    labels = Path(_patch_extr_csv)

    labels_table = pd.read_csv(labels)

    # Test csv read with header
    out_table = utils.misc.read_locations(labels)
    assert all(labels_table == out_table)
    assert out_table.shape[1] == 3

    # Test csv read without header
    labels = Path(_patch_extr_csv_noheader)
    out_table = utils.misc.read_locations(labels)
    assert all(labels_table == out_table)
    assert out_table.shape[1] == 3

    labels = Path(_patch_extr_svs_csv)
    out_table = utils.misc.read_locations(labels)
    assert out_table.shape[1] == 3

    labels = Path(_patch_extr_svs_header)
    out_table = utils.misc.read_locations(labels)
    assert out_table.shape[1] == 3

    # Test npy read
    labels = Path(_patch_extr_npy)
    out_table = utils.misc.read_locations(labels)
    assert all(labels_table == out_table)
    assert out_table.shape[1] == 3

    # Test pd dataframe read
    out_table = utils.misc.read_locations(labels_table)
    assert all(labels_table == out_table)
    assert out_table.shape[1] == 3

    labels_table_2 = labels_table.drop("class", axis=1)
    out_table = utils.misc.read_locations(labels_table_2)
    assert all(labels_table == out_table)
    assert out_table.shape[1] == 3

    # Test json read
    labels = Path(_patch_extr_json)
    out_table = utils.misc.read_locations(labels)
    assert all(labels_table == out_table)
    assert out_table.shape[1] == 3

    # Test json read 2 columns
    labels = Path(_patch_extr_2col_json)
    out_table = utils.misc.read_locations(labels)
    assert all(labels_table == out_table)
    assert out_table.shape[1] == 3

    # Test numpy array
    out_table = utils.misc.read_locations(labels_table.to_numpy())
    assert all(labels_table == out_table)
    assert out_table.shape[1] == 3

    out_table = utils.misc.read_locations(labels_table.to_numpy()[:, 0:2])
    assert all(labels_table == out_table)
    assert out_table.shape[1] == 3

    # Test if input array does not have 2 or 3 columns
    with pytest.raises(ValueError):
        _ = utils.misc.read_locations(labels_table.to_numpy()[:, 0:1])

    # Test if input npy does not have 2 or 3 columns
    labels = tmp_path.joinpath("test_gt_3col.npy")
    with open(labels, "wb") as f:
        np.save(f, np.zeros((3, 4)))

    with pytest.raises(ValueError):
        _ = utils.misc.read_locations(labels)

    # Test if input pd DataFrame does not have 2 or 3 columns
    with pytest.raises(ValueError):
        _ = utils.misc.read_locations(labels_table.drop(["y", "class"], axis=1))

    with pytest.raises(FileNotSupported):
        labels = Path("./sample_patch_extraction.test")
        _ = utils.misc.read_locations(labels)

    with pytest.raises(TypeError):
        _ = utils.misc.read_locations(["a", "b", "c"])


def test_grab_files_from_dir(_sample_visual_fields):
    """Test grab files from dir utils.misc."""
    file_parent_dir = Path(__file__).parent
    input_path = file_parent_dir.joinpath("data")

    file_types = "*.tif, *.png, *.jpg"

    out = utils.misc.grab_files_from_dir(
        input_path=_sample_visual_fields, file_types=file_types
    )
    assert len(out) == 5

    out = utils.misc.grab_files_from_dir(
        input_path=input_path.parent, file_types="test_utils*"
    )

    assert len(out) == 1
    assert str(Path(__file__)) == str(out[0])

    out = utils.misc.grab_files_from_dir(input_path=input_path, file_types="*.py")
    assert len(out) == 0


def test_download_unzip_data():
    """Test download and unzip data from utils.misc."""
    url = "https://tiatoolbox.dcs.warwick.ac.uk/testdata/utils/test_directory.zip"
    save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "tmp/")
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    os.makedirs(save_dir_path)
    save_zip_path = os.path.join(save_dir_path, "test_directory.zip")
    misc.download_data(url, save_zip_path)
    misc.download_data(url, save_zip_path, overwrite=True)  # do overwrite
    misc.unzip_data(save_zip_path, save_dir_path, del_zip=False)  # not remove
    assert os.path.exists(save_zip_path)
    misc.unzip_data(save_zip_path, save_dir_path)

    extracted_path = os.path.join(save_dir_path, "test_directory")
    # to avoid hidden files in case of MAC-OS or Windows (?)
    extracted_dirs = [f for f in os.listdir(extracted_path) if not f.startswith(".")]
    extracted_dirs.sort()  # ensure same ordering
    assert extracted_dirs == ["dir1", "dir2", "dir3"]

    shutil.rmtree(save_dir_path, ignore_errors=True)


def test_download_data():
    """Test download data from utils.misc."""
    url = "https://tiatoolbox.dcs.warwick.ac.uk/testdata/utils/test_directory.zip"
    save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "tmp/")
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    save_zip_path = os.path.join(save_dir_path, "test_directory.zip")

    misc.download_data(url, save_zip_path, overwrite=True)  # overwrite
    old_hash = hashlib.md5(open(save_zip_path, "rb").read()).hexdigest()
    # modify the content
    with open(save_zip_path, "wb") as fptr:
        fptr.write("dataXXX".encode())  # random data
    bad_hash = hashlib.md5(open(save_zip_path, "rb").read()).hexdigest()
    assert old_hash != bad_hash
    misc.download_data(url, save_zip_path, overwrite=True)  # overwrite
    new_hash = hashlib.md5(open(save_zip_path, "rb").read()).hexdigest()
    assert new_hash == old_hash

    # test not overiting
    # modify the content
    with open(save_zip_path, "wb") as fptr:
        fptr.write("dataXXX".encode())  # random data
    bad_hash = hashlib.md5(open(save_zip_path, "rb").read()).hexdigest()
    assert old_hash != bad_hash
    misc.download_data(url, save_zip_path, overwrite=False)  # data already exists
    new_hash = hashlib.md5(open(save_zip_path, "rb").read()).hexdigest()
    assert new_hash == bad_hash

    shutil.rmtree(save_dir_path, ignore_errors=True)  # remove data
    misc.download_data(url, save_zip_path)  # to test skip download
    assert os.path.exists(save_zip_path)
    shutil.rmtree(save_dir_path, ignore_errors=True)

    # URL not valid
    with pytest.raises(ConnectionError):
        # shouldn't use save_path if test runs correctly
        save_path = os.path.join(save_dir_path, "temp")
        misc.download_data(
            "https://tiatoolbox.dcs.warwick.ac.uk/invalid-url", save_path
        )


def test_parse_cv2_interpolaton():
    """Test parsing interpolation modes for cv2."""
    cases = [str.upper, str.lower, str.capitalize]
    mode_strings = ["cubic", "linear", "area", "lanczos"]
    mode_enums = [cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
    for string, cv2_enum in zip(mode_strings, mode_enums):
        for case in cases:
            assert utils.misc.parse_cv2_interpolaton(case(string)) == cv2_enum
            assert utils.misc.parse_cv2_interpolaton(cv2_enum) == cv2_enum

    with pytest.raises(ValueError):
        assert utils.misc.parse_cv2_interpolaton(1337)


def test_make_bounds_size_positive():
    """Test make_bounds_size_positive outputs positive bounds."""
    # Horizontal only
    bounds = (0, 0, -10, 10)
    pos_bounds, hflip, vflip = utils.image.make_bounds_size_positive(bounds)
    _, size = utils.transforms.bounds2locsize(pos_bounds)
    assert len(size) == 2
    assert size[0] > 0
    assert size[1] > 0
    assert hflip is True
    assert vflip is False

    # Vertical only
    bounds = (0, 0, 10, -10)
    pos_bounds, hflip, vflip = utils.image.make_bounds_size_positive(bounds)
    _, size = utils.transforms.bounds2locsize(pos_bounds)
    assert len(size) == 2
    assert size[0] > 0
    assert size[1] > 0
    assert hflip is False
    assert vflip is True

    # Both
    bounds = (0, 0, -10, -10)
    pos_bounds, hflip, vflip = utils.image.make_bounds_size_positive(bounds)
    _, size = utils.transforms.bounds2locsize(pos_bounds)
    assert len(size) == 2
    assert size[0] > 0
    assert size[1] > 0
    assert hflip is True
    assert vflip is True


def test_crop_and_pad_edges():
    """Test crop and pad util function."""
    slide_dimensions = (1024, 1024)

    def edge_mask(bounds: Tuple[int, int, int, int]) -> np.ndarray:
        """Produce a mask of regions outside of the slide dimensions."""
        l, t, r, b = bounds
        slide_width, slide_height = slide_dimensions
        X, Y = np.meshgrid(np.arange(l, r), np.arange(t, b), indexing="ij")
        under = np.logical_or(X < 0, Y < 0).astype(np.int)
        over = np.logical_or(X >= slide_width, Y >= slide_height).astype(np.int)
        return under, over

    loc = (-5, -5)
    size = (10, 10)
    bounds = utils.transforms.locsize2bounds(loc, size)
    under, over = edge_mask(bounds)
    region = -under + over
    region = np.sum(np.meshgrid(np.arange(10, 20), np.arange(10, 20)), axis=0)
    output = utils.image.crop_and_pad_edges(
        bounds=bounds,
        max_dimensions=slide_dimensions,
        region=region,
        pad_mode="constant",
    )

    assert np.all(np.logical_or(output >= 10, output == 0))
    assert output.shape == region.shape

    slide_width, slide_height = slide_dimensions

    loc = (slide_width - 5, slide_height - 5)
    bounds = utils.transforms.locsize2bounds(loc, size)
    under, over = edge_mask(bounds)
    region = np.sum(np.meshgrid(np.arange(10, 20), np.arange(10, 20)), axis=0)
    output = utils.image.crop_and_pad_edges(
        bounds=bounds,
        max_dimensions=slide_dimensions,
        region=region,
        pad_mode="constant",
    )

    assert np.all(np.logical_or(output >= 10, output == 0))
    assert output.shape == region.shape


def test_fuzz_crop_and_pad_edges_output_size():
    """Fuzz test crop and pad util function output size."""
    random.seed(0)

    for _ in range(1000):
        slide_dimensions = (random.randint(0, 50), random.randint(0, 50))

        loc = (-5, -5)
        size = (10, 10)
        bounds = utils.transforms.locsize2bounds(loc, size)
        region = np.sum(np.meshgrid(np.arange(10, 20), np.arange(10, 20)), axis=0)
        output = utils.image.crop_and_pad_edges(
            bounds=bounds,
            max_dimensions=slide_dimensions,
            region=region,
            pad_mode="constant",
        )

        assert output.shape == region.shape


def test_crop_and_pad_edges_negative_max_dims():
    """Test crop and pad edges for negative max dims."""
    for max_dims in [(-1, 1), (1, -1), (-1, -1)]:
        with pytest.raises(ValueError):
            utils.image.crop_and_pad_edges(
                bounds=(0, 0, 1, 1),
                max_dimensions=max_dims,
                region=np.zeros((10, 10)),
                pad_mode="constant",
            )

    # Zero dimensions
    utils.image.crop_and_pad_edges(
        bounds=(0, 0, 1, 1),
        max_dimensions=(0, 0),
        region=np.zeros((10, 10)),
        pad_mode="constant",
    )


def test_crop_and_pad_edges_non_positive_bounds_size():
    """Test crop and pad edges for non positive bound size."""
    with pytest.raises(ValueError):
        # Zero dimensions and negative bounds size
        utils.image.crop_and_pad_edges(
            bounds=(0, 0, -1, -1),
            max_dimensions=(0, 0),
            region=np.zeros((10, 10)),
            pad_mode="constant",
        )

    with pytest.raises(ValueError):
        # Zero dimensions and negative bounds size
        utils.image.crop_and_pad_edges(
            bounds=(0, 0, 0, 0),
            max_dimensions=(0, 0),
            region=np.zeros((10, 10)),
            pad_mode="constant",
        )


def test_normalise_padding_input_dims():
    """Test that normalise padding error with input dimensions > 1."""
    with pytest.raises(ValueError):
        utils.image.normalise_padding_size(((0, 0), (0, 0)))


def test_select_device():
    """Test if correct device is selected for models."""
    device = misc.select_device(on_gpu=True)
    assert device == "cuda"

    device = misc.select_device(on_gpu=False)
    assert device == "cpu"


def test_model_to():
    """Test for placing model on device."""
    import torch.nn as nn
    import torchvision.models as torch_models

    # test on GPU
    # no GPU on Travis so this will crash
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            model = torch_models.resnet18()
            _ = misc.model_to(on_gpu=True, model=model)

    # test on CPU
    model = torch_models.resnet18()
    model = misc.model_to(on_gpu=False, model=model)
    assert isinstance(model, nn.Module)


def test_save_as_json():
    """Test save data to json."""
    import json

    # dict with nested dict, list, and np.array
    key_dict = {
        "a1": {"name": "John", "age": 23, "sex": "male"},
        "a2": {"name": "John", "age": 23, "sex": "male"},
    }
    sample = {
        "a": [1, 1, 3, np.random.rand(2, 2, 2, 2), key_dict],
        "b": ["a1", "b1", "c1", {"a3": [1.0, 1, 3, np.random.rand(2, 2, 2, 2)]}],
        "c": {
            "a4": {"a5": {"a6": "a7", "c": [1, 1, 3, np.array([4, 5, 6.0])]}},
            "b1": {},
            "c1": [],
            True: [False, None],
        },
        "d": [key_dict, np.random.rand(2, 2)],
        "e": np.random.rand(16, 2),
    }
    not_jsonable = {"x86": lambda x: x}
    not_jsonable.update(sample)
    # should fail because key is not of primitive type [str, int, float, bool]
    with pytest.raises(ValueError, match=r".*Key.*.*not jsonified.*"):
        misc.save_as_json({frozenset(key_dict): sample}, "sample_json.json")
    with pytest.raises(ValueError, match=r".*Value.*.*not jsonified.*"):
        misc.save_as_json(not_jsonable, "sample_json.json")
    with pytest.raises(ValueError, match=r".*Value.*.*not jsonified.*"):
        misc.save_as_json(list(not_jsonable.values()), "sample_json.json")
    with pytest.raises(ValueError, match=r".*`data`.*.*not.*dict, list.*"):
        misc.save_as_json(np.random.rand(2, 2), "sample_json.json")
    # test complex nested dict
    print(sample)
    misc.save_as_json(sample, "sample_json.json")
    with open("sample_json.json", "r") as fptr:
        read_sample = json.load(fptr)
    # test read because == is useless when value is mutable
    assert read_sample["c"]["a4"]["a5"]["a6"] == "a7"
    assert read_sample["c"]["a4"]["a5"]["c"][-1][-1] == 6

    # test complex list of data
    misc.save_as_json(list(sample.values()), "sample_json.json")
    # test read because == is useless when value is mutable
    with open("sample_json.json", "r") as fptr:
        read_sample = json.load(fptr)
    assert read_sample[-3]["a4"]["a5"]["a6"] == "a7"
    assert read_sample[-3]["a4"]["a5"]["c"][-1][-1] == 6

    # test numpy generic
    misc.save_as_json([np.int32(1), np.float32(2)], "sample_json.json")
    misc.save_as_json({"a": np.int32(1), "b": np.float32(2)}, "sample_json.json")
    os.remove("sample_json.json")
