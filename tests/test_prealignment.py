import urllib

import PIL.Image as Image
import numpy as np
import pytest

from tiatoolbox.models.engine.semantic_segmentor import (
	IOSegmentorConfig,
	SemanticSegmentor,
)
from tiatoolbox.tools.registration.prealignment import prealignment


def test_prealignment():
    """Test for prealignment of an image pair"""
    main_url = "https://tiatoolbox.dcs.warwick.ac.uk/testdata/registration/"
    urllib.request.urlretrieve(main_url + "HE_1_level8_gray.png", "sample.png")
    fixed_img = np.asarray(Image.open("sample.png"))
    urllib.request.urlretrieve(main_url + "HE_2_level8_gray.png", "sample.png")
    moving_img = np.asarray(Image.open("sample.png"))
    urllib.request.urlretrieve(main_url + "HE_1_level8_mask.png", "sample.png")
    fixed_mask = np.asarray(Image.open("sample.png"))
    urllib.request.urlretrieve(main_url + "HE_2_level8_mask.png", "sample.png")
    moving_mask = np.asarray(Image.open("sample.png"))

    transform = prealignment(fixed_img, moving_img, fixed_mask, moving_mask)
    assert transform.shape == (3, 3)

    no_fixed_mask = np.zeros(shape=fixed_img.shape, dtype=int)
    no_moving_mask = np.zeros(shape=moving_img.shape, dtype=int)
    with pytest.raises(ValueError, match=r".*The foreground is missing in the mask.*"):
        _ = prealignment(fixed_img, moving_img, no_fixed_mask, no_moving_mask)

    with pytest.raises(
        ValueError,
        match=r".*Mismatch of shape between image and its corresponding mask.*",
    ):
        _ = prealignment(fixed_img, moving_img, moving_mask, fixed_mask)

    fixed_img_3d = np.repeat(np.expand_dims(fixed_img, axis=2), 3, axis=2)
    moving_img_3d = np.repeat(np.expand_dims(moving_img, axis=2), 3, axis=2)
    with pytest.raises(
        ValueError, match=r".*The input images should be grayscale images.*"
    ):
        _ = prealignment(fixed_img_3d, moving_img_3d, fixed_mask, moving_mask)


def test_rotation_step_range():
    """Test if the value of rotation step is within the range"""
    fixed_img = np.random.randint(20, size=(256, 256))
    moving_img = np.random.randint(20, size=(256, 256))
    fixed_mask = np.random.randint(2, size=(256, 256))
    moving_mask = np.random.randint(2, size=(256, 256))

    _ = prealignment(fixed_img, moving_img, fixed_mask, moving_mask, 15)

    with pytest.raises(
        ValueError, match=r".*Please select the rotation step in between 10 and 20.*"
    ):
        _ = prealignment(fixed_img, moving_img, fixed_mask, moving_mask, 9)

    with pytest.raises(
        ValueError, match=r".*Please select the rotation step in between 10 and 20.*"
    ):
        _ = prealignment(fixed_img, moving_img, fixed_mask, moving_mask, 21)


def test_tissue_segmentation():
    #     """Test for coarse registration of an image pair using masks
    #     generated with a pretrained model."""

    main_url = "https://tiatoolbox.dcs.warwick.ac.uk/testdata/registration/"
    urllib.request.urlretrieve(main_url + "HE_1_level8_gray.png", "fixed.png")
    urllib.request.urlretrieve(main_url + "HE_2_level8_gray.png", "moving.png")

    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {"units": "baseline", "resolution": 1.0},
        ],
        output_resolutions=[
            {"units": "baseline", "resolution": 1.0},
        ],
        patch_input_shape=[1024, 1024],
        patch_output_shape=[512, 512],
        stride_shape=[512, 512],
        save_resolution={"units": "baseline", "resolution": 1.0},
    )

    segmentor = SemanticSegmentor(
        pretrained_model="unet_tissue_mask_tsef",
        num_loader_workers=4,
        batch_size=4,
    )

    output = segmentor.predict(
        ["fixed.png", "moving.png"],
        save_dir=None,
        mode="tile",
        on_gpu=False,
        ioconfig=ioconfig,
        crash_on_exception=True,
    )

    fixed_img = np.asarray(Image.open("fixed.png"))
    moving_img = np.asarray(Image.open("moving.png"))
    fixed_mask = np.load(output[0][1] + ".raw.0.npy")
    assert len(fixed_mask.shape) == 3
    fixed_mask = fixed_mask[:, :, 2] > 0.5
    moving_mask = np.load(output[1][1] + ".raw.0.npy")[:, :, 2] > 0.5
    _ = prealignment(fixed_img, moving_img, fixed_mask, moving_mask)
