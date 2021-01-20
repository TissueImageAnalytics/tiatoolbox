from tiatoolbox import utils
from tiatoolbox.utils.exceptions import FileNotSupported

import random
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


def test_imresize():
    """Test for imresize."""
    img = np.zeros((2000, 1000, 3))
    resized_img = utils.transforms.imresize(img, scale_factor=0.5)
    assert resized_img.shape == (1000, 500, 3)

    resized_img = utils.transforms.imresize(resized_img, scale_factor=2.0)
    assert resized_img.shape == (2000, 1000, 3)


def test_background_composite():
    """Test for background composite."""
    new_im = np.zeros((2000, 2000, 4)).astype("uint8")
    new_im[:1000, :, 3] = 255
    im = utils.transforms.background_composite(new_im)
    assert np.all(im[1000:, :, :] == 255)
    assert np.all(im[:1000, :, :] == 0)

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
    for padding in [1, [1], (1,), [1, 1], (1, 1), [1] * 4]:
        region = utils.image.safe_padded_read(
            data,
            bounds,
            padding=padding,
        )
        assert region.shape == (8 + 2, 8 + 2)


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


def test_sub_pixel_read():
    """Test sub-pixel numpy image reads with known tricky parameters."""
    image_path = Path(__file__).parent / "data" / "source_image.png"
    assert image_path.exists()
    test_image = utils.misc.imread(image_path)

    x = 6
    y = -4
    w = 21.805648705868652
    h = 0.9280264518437986
    bounds = (x, y, x + w, y + h)
    ow = 88
    oh = 98
    output = utils.image.sub_pixel_read(test_image, bounds, (ow, oh))
    assert (ow, oh) == tuple(output.shape[:2][::-1])

    x = 13
    y = 15
    w = 29.46
    h = 6.92
    bounds = (x, y, x + w, y + h)
    ow = 93
    oh = 34
    output = utils.image.sub_pixel_read(test_image, bounds, (ow, oh))
    assert (ow, oh) == tuple(output.shape[:2][::-1])


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
    with pytest.warns(UserWarning):
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


def test_fuzz_sub_pixel_read():
    """Fuzz test for numpy sub-pixel image reads."""
    random.seed(0)

    image_path = Path(__file__).parent / "data" / "source_image.png"
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


def test_fuzz_bounds2size():
    """Fuzz test for bounds2size."""
    random.seed(0)
    for _ in range(1000):
        size = (random.randint(-1000, 1000), random.randint(-1000, 1000))
        location = (random.randint(-1000, 1000), random.randint(-1000, 1000))
        bounds = (*location, *(sum(x) for x in zip(size, location)))
        assert np.array_equal(utils.transforms.bounds2size(bounds), size)


def test_fuzz_bounds2size_lower():
    """Fuzz test for bounds2size with origin lower."""
    random.seed(0)
    for _ in range(1000):
        size = (random.randint(-1000, 1000), random.randint(-1000, 1000))
        starts = (random.randint(-1000, 1000), random.randint(-1000, 1000))
        ends = [sum(x) for x in zip(size, starts)]
        bounds = (starts[0], ends[1], ends[0], starts[1])

        assert np.array_equal(
            utils.transforms.bounds2size(bounds, origin="lower"), size
        )


def test_bounds2size_value_error():
    """Test bounds to size ValueError."""
    with pytest.raises(ValueError):
        utils.transforms.bounds2size((0, 0, 1, 1), origin="middle")


def test_contrast_enhancer():
    """"Test contrast enhancement funcitionality."""
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


def test_grab_files_from_dir():
    """Test grab files from dir utils.misc."""
    file_parent_dir = Path(__file__).parent
    input_path = file_parent_dir.joinpath("data")

    file_types = "*.tif, *.png, *.jpg"

    out = utils.misc.grab_files_from_dir(input_path=input_path, file_types=file_types)
    assert len(out) == 5

    out = utils.misc.grab_files_from_dir(
        input_path=input_path.parent, file_types="test_utils*"
    )

    assert len(out) == 1
    assert str(Path(__file__)) == str(out[0])

    out = utils.misc.grab_files_from_dir(input_path=input_path, file_types="*.py")
    assert len(out) == 0
