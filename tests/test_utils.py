from tiatoolbox import utils
from tiatoolbox.utils.exceptions import FileNotSupported
from tiatoolbox import TIATOOLBOX_HOME

import random
import pytest
from pytest import approx
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2


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

    bounds = (1.5, 1, -5, 5)
    with pytest.warns(UserWarning), pytest.raises(AssertionError):
        utils.image.sub_pixel_read(data, bounds, out_size)

    bounds = (1.5, 1, 1.5, 0)
    with pytest.raises(AssertionError):
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
        pad_for_interpolation=False,
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
            test_image, bounds, (ow, oh), interpolation="linear"
        )
        assert (ow, oh) == tuple(output.shape[:2][::-1])


def test_sub_pixel_read_interpolation_modes():
    """Test sub_pixel_read with different padding argument formats."""
    data = np.mgrid[:16:1, :16:1].sum(0).astype(np.uint8)
    out_size = data.shape
    bounds = (0, 0, 8, 8)
    for mode in ["nearest", "linear", "cubic", "lanczos"]:
        output = utils.image.sub_pixel_read(data, bounds, out_size, interpolation=mode)
        assert output.shape == out_size


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
    """ "Test contrast enhancement functionality."""
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
    with pytest.raises(FileNotSupported):
        utils.misc.load_stain_matrix("/samplefile.xlsx")

    with pytest.raises(TypeError):
        # load_stain_matrix requires numpy array as input providing list here
        utils.misc.load_stain_matrix([1, 2, 3])

    stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
    pd.DataFrame(stain_matrix).to_csv(Path(tmp_path).joinpath("sm.csv"), index=False)
    out_stain_matrix = utils.misc.load_stain_matrix(Path(tmp_path).joinpath("sm.csv"))
    assert np.all(out_stain_matrix == stain_matrix)

    np.save(Path(tmp_path).joinpath("sm.npy"), stain_matrix)
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
    file_parent_dir = Path(__file__).parent
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
        labels = file_parent_dir.joinpath("data/sample_patch_extraction.test")
        _ = utils.misc.read_locations(labels)

    with pytest.raises(TypeError):
        _ = utils.misc.read_locations(["a", "b", "c"])


def test_grab_files_from_dir():
    """Test grab files from dir utils.misc."""
    file_parent_dir = Path(__file__).parent
    input_path = file_parent_dir.joinpath("data")

    file_types = "*.tif, *.png, *.jpg"

    out = utils.misc.grab_files_from_dir(input_path=input_path, file_types=file_types)
    assert len(out) == 6

    out = utils.misc.grab_files_from_dir(
        input_path=input_path.parent, file_types="test_utils*"
    )

    assert len(out) == 1
    assert str(Path(__file__)) == str(out[0])

    out = utils.misc.grab_files_from_dir(input_path=input_path, file_types="*.py")
    assert len(out) == 0


def test_download_unzip_data():
    """Test download and unzip data from utils.misc."""
    url = "https://tiatoolbox.dcs.warwick.ac.uk/utils/test_directory.zip"
    save_dir_path = (os.path.join(TIATOOLBOX_HOME, "tmp/"),)
    os.mkdir(save_dir_path)
    utils.download_data(url, save_zip_path)
    utils.unzip_data(save_zip_path, save_dir_path)

    assert os.listdir(save_dir_path) == ["dir1", "dir2", "dir3"]

    os.rmdir(save_dir_path)
