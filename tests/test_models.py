"""Tests for code related to model usage."""

# %%
import os
import pathlib
import shutil
import cv2
from cv2 import data

import numpy as np
import pytest
import torch
from click.testing import CliRunner

import sys
# sys.path.append('.')
# sys.path.append('..')

from sklearn import metrics
from tiatoolbox import rcParam
from tiatoolbox.models.backbone import get_model
from tiatoolbox.models.classification.abc import ModelBase
from tiatoolbox.models.classification import CNNPatchModel, CNNPatchPredictor
from tiatoolbox.models.dataset import (
    ABCDatasetInfo,
    KatherPatchDataset,

    PatchDataset,
    WSIPatchDataset,
    predefined_preproc_func,
)
from tiatoolbox.utils.misc import download_data, unzip_data, imread
from tiatoolbox import cli
from tiatoolbox.wsicore import wsireader

from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIMeta
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, get_wsireader

# def test_sync_read():
    # wsi_path = '/home/tialab-dang/local/project/tiatoolbox/tests/CMU-1_mini.svs'
    # wsi_mask_path = '/home/tialab-dang/local/project/tiatoolbox/tests/CMU-1_mini_thumb_mask.png'
    # mask = imread(wsi_mask_path)
    # wsi_reader = get_wsireader(wsi_path)
    # wsi_metadata = wsi_reader.info
    # mask_reader = VirtualWSIReader(mask)
    # mask_reader.attach_to_reader(wsi_reader.info)
    # #
    # wsi_lv0_coords = np.array([4500, 9500, 6500, 11500])
    # roi_img = wsi_reader.read_bounds(
    #             wsi_lv0_coords,
    #             resolution=1.0,
    #             units='mpp'
    #         )
    # roi_msk = mask_reader.read_bounds(
    #             wsi_lv0_coords / mask_reader.info.level_downsamples[0],
    #             resolution=1.0,
    #             units='mpp'
    #         )

    # wsi_path = '/home/tialab-dang/local/project/tiatoolbox/tests/CMU-1_mini.jpg'
    # wsi_mask_path = '/home/tialab-dang/local/project/tiatoolbox/tests/CMU-1_mini_thumb_mask.png'
    # wsi = imread(wsi_path)
    # wsi_reader = VirtualWSIReader(wsi_path, WSIMeta(
    #     mpp=np.array([0.25, 0.25]),
    #     slide_dimensions=np.array(wsi.shape[:2][::-1]),
    #     level_downsamples=[1.0],
    #     level_dimensions=[np.array(wsi.shape[:2][::-1])]
    # ))

    # mask = imread(wsi_mask_path)
    # mask_reader = VirtualWSIReader(mask)
    # mask_reader.attach_to_reader(wsi_reader.info)
    # #
    # wsi_lv0_coords = np.array([4500, 9500, 6500, 11500])
    # roi_img = wsi_reader.read_bounds(
    #             wsi_lv0_coords,
    #             resolution=1.0,
    #             units='mpp'
    #         )
    # roi_msk = mask_reader.read_bounds(
    #             wsi_lv0_coords / mask_reader.info.level_downsamples[0],
    #             resolution=1.0,
    #             units='mpp'
    #         )

    # import matplotlib.pyplot as plt
    # plt.subplot(1,2,1)
    # plt.imshow(roi_img)
    # plt.subplot(1,2,2)
    # plt.imshow(roi_msk)
    # plt.savefig('dump.png')

# %%

# def test_wsi_patch_predictor():
#     """Test for patch predictor with resnet50 on Kather 100K dataset."""
#     # API 1, also test with return_labels
#     sample_pyramid = "CMU-1_mini.svs"
#     # dataset = WSIPatchDataset(sample_pyramid)
#     results = _get_outputs_api1(sample_pyramid, "resnet18-kather100K", mode="wsi")
#     # probabilities, predictions, labels


# def test_wsi_patch_predictor(_sample_crc_tile):
#     """Test for patch predictor with resnet50 on Kather 100K dataset."""
#     # API 1, also test with return_labels
#     results = _get_outputs_api1(
#         pathlib.Path(_sample_crc_tile), "resnet18-kather100K", mode="wsi"
#     )
#     # probabilities, predictions, labels


# def _get_outputs_api1(data, pretrained_model, mode):
#     """Helper function to get the model output using API 1."""
#     # API 1, also test with return_labels
#     predictor = CNNPatchPredictor(pretrained_model=pretrained_model, batch_size=2)
#     # don't run test on GPU
#     output = predictor.predict(
#         data,
#         mode=mode,
#         return_probabilities=True,
#         return_labels=True,
#         on_gpu=True,

#         patch_shape=np.array([224, 224])
#     )
#     probabilities = output["probabilities"]
#     predictions = output["predictions"]

#     if mode == "wsi" or mode == "tile":
#         coordinates = output["coordinates"]
#         return probabilities, predictions, coordinates
#     else:
#         labels = output["labels"]
#         return probabilities, predictions, labels


# def test_tile(_sample_crc_tile):
    # __sample = _sample_crc_tile
# __sample = pathlib.Path('/home/tialab-dang/local/project/tiatoolbox/tests/mini_tile.tif')
# predictor = CNNPatchPredictor(
#                 pretrained_model="resnet18-kather100K",
#                 batch_size=2)
# # don't run test on GPU
# output = predictor.predict(
#     [__sample],
#     mode='tile',
#     return_probabilities=True,
#     return_labels=True,
#     on_gpu=False,

#     patch_shape=np.array([224, 224])
# )

#     # don't run test on GPU
#     output = predictor.predict(
#         __sample,
#         mode='tile',
#         return_probabilities=True,
#         return_labels=True,
#         on_gpu=False,

#         patch_shape=np.array([224, 224]),
#         stride_shape=np.array([112, 112])
#     )


# def test_wsi(_sample_svs):
# __sample_wsi = pathlib.Path('/home/tialab-dang/local/project/tiatoolbox/tests/CMU-1_mini.svs')
# __sample_msk = pathlib.Path('/home/tialab-dang/local/project/tiatoolbox/tests/CMU-1_mini_thumb_mask.png')

# # __sample = _sample_svs
# predictor = CNNPatchPredictor(
#                 pretrained_model="resnet18-kather100K", 
#                 batch_size=2)
# # don't run test on GPU
# output = predictor.predict(
#     [__sample_wsi],
#     mask_list=[__sample_msk],
#     mode='wsi',
#     return_probabilities=True,
#     return_labels=True,
#     on_gpu=False,

#     patch_shape=np.array([224, 224]),
#     resolution=1.0,
#     units='mpp'
# )

#     # # don't run test on GPU
#     # output = predictor.predict(
#     #     __sample,
#     #     mode='wsi',
#     #     return_probabilities=True,
#     #     return_labels=True,
#     #     on_gpu=False,

#     #     patch_shape=np.array([224, 224]),
#     #     resolution=2.0,
#     #     units='mpp'
#     # )

# dummy_test_wsi()

# %%

# results = _get_outputs_api1(
#     pathlib.Path('/home/tialab-dang/local/project/tiatoolbox/tests/CMU-1.ndpi'), 
#     "resnet18-kather100K", mode="wsi"
# )

# def _get_outputs_api2(data, pretrained_model, mode):
#     """Helper function to get the model output using API 2."""
#     # API 2
#     pretrained_weight_url = (
#         "https://tiatoolbox.dcs.warwick.ac.uk/models/resnet18-kather100K-pc.pth"
#     )

#     save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "tmp_api2")
#     # remove prev generated data - just a test!
#     if os.path.exists(save_dir_path):
#         shutil.rmtree(save_dir_path, ignore_errors=True)
#     os.makedirs(save_dir_path)

#     pretrained_weight = os.path.join(
#         rcParam["TIATOOLBOX_HOME"], "tmp_api2", "resnet18-kather100K-pc.pth"
#     )
#     download_data(pretrained_weight_url, pretrained_weight)

#     predictor = CNNPatchPredictor(
#         pretrained_model=pretrained_model,
#         pretrained_weight=pretrained_weight,
#         batch_size=1,
#     )
#     # don't run test on GPU
#     output = predictor.predict(
#         data,
#         mode=mode,
#         return_probabilities=True,
#         return_labels=True,
#         on_gpu=False,
#     )
#     probabilities = output["probabilities"]
#     predictions = output["predictions"]
#     labels = output["labels"]

#     if mode == "wsi" or mode == "tile":
#         coordinates = output["coordinates"]
#         return probabilities, predictions, labels, coordinates
#     else:
#         return probabilities, predictions, labels


# def _get_outputs_api3(data, backbone, mode, num_classes=9):
#     """Helper function to get the model output using API 3."""
#     # API 3
#     model = CNNPatchModel(backbone=backbone, num_classes=num_classes)

#     # coverage setter check
#     model.set_preproc_func(lambda x: x - 1)  # do this for coverage
#     assert model.get_preproc_func()(1) == 0
#     # coverage setter check
#     model.set_preproc_func(None)  # do this for coverage
#     assert model.get_preproc_func()(1) == 1

#     predictor = CNNPatchPredictor(model=model, batch_size=1, verbose=False)
#     # don't run test on GPU
#     output = predictor.predict(
#         data,
#         mode=mode,
#         return_probabilities=True,
#         return_labels=True,
#         on_gpu=False,
#     )

#     probabilities = output["probabilities"]
#     predictions = output["predictions"]
#     labels = output["labels"]

#     if mode == "wsi" or mode == "tile":
#         coordinates = output["coordinates"]
#         return probabilities, predictions, labels, coordinates
#     else:
#         return probabilities, predictions, labels


# def test_create_backbone():
#     """Test for creating backbone."""
#     backbone_list = [
#         "alexnet",
#         "resnet18",
#         "resnet34",
#         "resnet50",
#         "resnet101",
#         "resnext50_32x4d",
#         "resnext101_32x8d",
#         "wide_resnet50_2",
#         "wide_resnet101_2",
#         "densenet121",
#         "densenet161",
#         "densenet169",
#         "densenet201",
#         "googlenet",
#         "mobilenet_v2",
#         "mobilenet_v3_large",
#         "mobilenet_v3_small",
#     ]
#     for backbone in backbone_list:
#         try:
#             get_model(backbone, pretrained=False)
#         except ValueError:
#             assert False, "Model %s failed." % backbone

#     # test for model not defined
#     with pytest.raises(ValueError, match=r".*not supported.*"):
#         get_model("secret_model", pretrained=False)


# def test_predictor_crash():
#     """Test for crash when making predictor."""
#     # test abc
#     with pytest.raises(NotImplementedError):
#         ModelBase()
#     with pytest.raises(NotImplementedError):
#         ModelBase.infer_batch(1, 2, 3)

#     # without providing any model
#     with pytest.raises(ValueError, match=r"Must provide.*"):
#         CNNPatchPredictor()

#     # provide wrong unknown pretrained model
#     with pytest.raises(ValueError, match=r"Pretrained .* does not exist"):
#         CNNPatchPredictor(pretrained_model="secret_model")

#     # provide wrong model of unknown type, deprecated later with type hint
#     with pytest.raises(ValueError, match=r".*must be a string.*"):
#         CNNPatchPredictor(pretrained_model=123)


# def test_set_root_dir():
#     """Test for setting new root dir."""
#     # skipcq
#     from tiatoolbox import rcParam

#     old_root_dir = rcParam["TIATOOLBOX_HOME"]
#     test_dir_path = os.path.join(os.getcwd(), "tmp_check/")
#     # clean up prev test
#     if os.path.exists(test_dir_path):
#         os.rmdir(test_dir_path)
#     rcParam["TIATOOLBOX_HOME"] = test_dir_path
#     # reimport to see if it overwrites
#     # silence Deep Source because this is intentional check
#     # skipcq
#     from tiatoolbox import rcParam

#     os.makedirs(rcParam["TIATOOLBOX_HOME"])
#     if not os.path.exists(test_dir_path):
#         assert False, "`%s` != `%s`" % (rcParam["TIATOOLBOX_HOME"], test_dir_path)
#     shutil.rmtree(rcParam["TIATOOLBOX_HOME"], ignore_errors=True)
#     rcParam["TIATOOLBOX_HOME"] = old_root_dir  # reassign for subsequent test

def test_WSIPatchDataset(_mini_wsi1_svs, _mini_wsi1_jpg):
    _mini_wsi1_svs = '/home/tialab-dang/local/project/tiatoolbox/tests/CMU-1_mini.svs'
    _mini_wsi1_jpg = '/home/tialab-dang/local/project/tiatoolbox/tests/CMU-1_mini.jpg'

    # to prevent wsireader complaint
    # _mini_wsi1_svs = pathlib.Path(_mini_wsi1_svs)
    # _mini_wsi1_jpg = pathlib.Path(_mini_wsi1_jpg)

    def reuse_init(**kwargs):
        return WSIPatchDataset(
                    wsi_path=_mini_wsi1_svs,
                    **kwargs)

    def reuse_init_wsi(**kwargs):
        return reuse_init(mode='wsi', **kwargs)
    # invalid mode
    with pytest.raises(ValueError):
        reuse_init(mode='X')

    # invalid patch
    with pytest.raises(ValueError):
        reuse_init()
    with pytest.raises(ValueError):
        reuse_init_wsi(patch_shape=[512, 512, 512])
    with pytest.raises(ValueError):
        reuse_init_wsi(patch_shape=[512, 'a'])
    with pytest.raises(ValueError):
        reuse_init_wsi(patch_shape=512)
    # invalid stride
    with pytest.raises(ValueError):
        reuse_init_wsi(
            patch_shape=[512, 512],
            stride_shape=[512, 'a'])
    with pytest.raises(ValueError):
        reuse_init_wsi(
            patch_shape=[512, 512],
            stride_shape=[512, 512, 512])

    # * dummy test for output correctness
    # * striding and patch should be as expected
    # * so we just need to do a manual retrieval and do sum check (hopefully)
    # * correct tiling or will be test in another way
    patch_shape = [4096, 4096]
    stride_shape = [2048, 2048]
    ds = reuse_init_wsi(
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            resolution=1.0,
            units='baseline')
    # tiling top to bottom, left to right
    ds_roi = ds[2]['image']
    step_idx = 2  # manual calibrate
    reader = get_wsireader(_mini_wsi1_svs)
    start = (0, step_idx * stride_shape[1])
    end = (start[0]+patch_shape[0], start[1]+patch_shape[1])
    rd_roi = reader.read_bounds(
        start + end,
        resolution=1.0,
        units='baseline'
    )
    correlation = np.corrcoef(
        cv2.cvtColor(ds_roi, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(rd_roi, cv2.COLOR_RGB2GRAY).flatten())

    assert ds_roi.shape[0] == rd_roi.shape[0]
    assert ds_roi.shape[1] == rd_roi.shape[1]
    assert np.min(correlation) > 0.9, correlation
    # uncomment these for internal viz check
    # import matplotlib.pyplot as plt
    # plt.subplot(1,2,1)
    # plt.imshow(ds_roi)
    # plt.subplot(1,2,2)
    # plt.imshow(rd_roi)
    # plt.savefig('dump.png')

    # ** repeated above test for tile at the same resolution as baseline
    # ** but is not pyramidal
    wsi_ds = WSIPatchDataset(
            wsi_path=_mini_wsi1_svs,
            mode='wsi',
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            resolution=1.0,
            units='baseline')
    tile_ds = WSIPatchDataset(
            wsi_path=_mini_wsi1_jpg,
            mode='tile',
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            resolution=1.0,
            units='baseline')
    assert len(tile_ds) == len(wsi_ds), '%s vs %s' % (len(tile_ds), len(wsi_ds))
    roi1 = wsi_ds[3]['image']
    roi2 = tile_ds[3]['image']
    correlation = np.corrcoef(
        cv2.cvtColor(roi1, cv2.COLOR_RGB2GRAY).flatten(),
        cv2.cvtColor(roi2, cv2.COLOR_RGB2GRAY).flatten())
    assert roi1.shape[0] == roi2.shape[0]
    assert roi1.shape[1] == roi2.shape[1]
    assert np.min(correlation) > 0.9, correlation

    # import matplotlib.pyplot as plt
    # plt.subplot(1,2,1)
    # plt.imshow(roi1)
    # plt.subplot(1,2,2)
    # plt.imshow(roi2)
    # plt.savefig('dump.png')


@pytest.mark.skip(reason="working, skip to run other test")
def test_PatchDatasetpath_imgs(_sample_patch1, _sample_patch2):
    """Test for patch dataset with a list of file paths as input."""
    size = (224, 224, 3)

    dataset = PatchDataset([pathlib.Path(_sample_patch1), pathlib.Path(_sample_patch2)])

    dataset.preproc_func = lambda x: x

    for _, sample_data in enumerate(dataset):
        sampled_img_shape = sample_data['image'].shape
        assert (
            sampled_img_shape[0] == size[0]
            and sampled_img_shape[1] == size[1]
            and sampled_img_shape[2] == size[2]
        )


@pytest.mark.skip(reason="working, skip to run other test")
def test_PatchDatasetlist_imgs():
    """Test for patch dataset with a list of images as input."""
    size = (5, 5, 3)
    img = np.random.randint(0, 255, size=size)
    list_imgs = [img, img, img]
    dataset = PatchDataset(list_imgs)

    dataset.preproc_func = lambda x: x

    for _, sample_data in enumerate(dataset):
        sampled_img_shape = sample_data['image'].shape
        assert (
            sampled_img_shape[0] == size[0]
            and sampled_img_shape[1] == size[1]
            and sampled_img_shape[2] == size[2]
        )

    # test for changing to another preproc
    dataset.preproc_func = lambda x : x - 10
    item = dataset[0]
    assert np.sum(item['image'] - (list_imgs[0] - 10)) == 0

    # test for loading npy
    save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "tmp_check/")
    # remove prev generated data - just a test!
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    os.makedirs(save_dir_path)
    np.save(
        os.path.join(save_dir_path, "sample2.npy"), np.random.randint(0, 255, (4, 4, 3))
    )
    img_list = [
        os.path.join(save_dir_path, "sample2.npy"),
    ]
    _ = PatchDataset(img_list)
    assert img_list[0] is not None
    # test for path object
    img_list = [
        pathlib.Path(os.path.join(save_dir_path, "sample2.npy")),
    ]
    _ = PatchDataset(img_list)
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"])


@pytest.mark.skip(reason="working, skip to run other test")
def test_PatchDatasetarray_imgs():
    """Test for patch dataset with a numpy array of a list of images."""
    size = (5, 5, 3)
    img = np.random.randint(0, 255, size=size)
    list_imgs = [img, img, img]
    label_list = [1, 2, 3]
    array_imgs = np.array(list_imgs)

    # test different setter for label
    dataset = PatchDataset(array_imgs, label_list=label_list)
    an_item = dataset[2]
    assert an_item['label'] == 3
    dataset = PatchDataset(array_imgs, label_list=None)
    an_item = dataset[2]
    assert 'label' not in an_item

    dataset = PatchDataset(array_imgs)
    for _, sample_data in enumerate(dataset):
        sampled_img_shape = sample_data['image'].shape
        assert (
            sampled_img_shape[0] == size[0]
            and sampled_img_shape[1] == size[1]
            and sampled_img_shape[2] == size[2]
        )


@pytest.mark.skip(reason="working, skip to run other test")
def test_PatchDatasetcrash():
    """Test to make sure patch dataset crashes with incorrect input."""
    # all examples below should fail when input to PatchDataset

    # not supported input type
    img_list = {"a": np.random.randint(0, 255, (4, 4, 4))}
    with pytest.raises(
        ValueError, match=r".*Input must be either a list/array of images.*"
    ):
        _ = PatchDataset(img_list)

    # ndarray of mixed dtype
    img_list = np.array([np.random.randint(0, 255, (4, 5, 3)), "Should crash"])
    with pytest.raises(ValueError, match="Provided input array is non-numerical."):
        _ = PatchDataset(img_list)

    # ndarrays of NHW images
    img_list = np.random.randint(0, 255, (4, 4, 4))
    with pytest.raises(ValueError, match=r".*array of images of the form NHWC.*"):
        _ = PatchDataset(img_list)

    # list of ndarrays with different sizes
    img_list = [
        np.random.randint(0, 255, (4, 4, 3)),
        np.random.randint(0, 255, (4, 5, 3)),
    ]
    with pytest.raises(ValueError, match="Images must have the same dimensions."):
        _ = PatchDataset(img_list)

    # list of ndarrays with HW and HWC mixed up
    img_list = [
        np.random.randint(0, 255, (4, 4, 3)),
        np.random.randint(0, 255, (4, 4)),
    ]
    with pytest.raises(
        ValueError, match="Each sample must be an array of the form HWC."
    ):
        _ = PatchDataset(img_list)

    # list of mixed dtype
    img_list = [np.random.randint(0, 255, (4, 4, 3)), "you_should_crash_here", 123, 456]
    with pytest.raises(
        ValueError,
        match="Input must be either a list/array of images or a list of "
        "valid image paths.",
    ):
        _ = PatchDataset(img_list)

    # list of mixed dtype
    img_list = ["you_should_crash_here", 123, 456]
    with pytest.raises(
        ValueError,
        match="Input must be either a list/array of images or a list of "
        "valid image paths.",
    ):
        _ = PatchDataset(img_list)

    # list not exist paths
    with pytest.raises(
        ValueError,
        match=r".*valid image paths.*",
    ):
        _ = PatchDataset(["img.npy"])

    # ** test different extenstion parser
    # save dummy data to temporary location
    save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "tmp_check/")
    # remove prev generated data - just a test!
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    os.makedirs(save_dir_path)
    torch.save({"a": "a"}, os.path.join(save_dir_path, "sample1.tar"))
    np.save(
        os.path.join(save_dir_path, "sample2.npy"), np.random.randint(0, 255, (4, 4, 3))
    )

    img_list = [
        os.path.join(save_dir_path, "sample1.tar"),
        os.path.join(save_dir_path, "sample2.npy"),
    ]
    with pytest.raises(
        ValueError,
        match=r"Can not load data of .*",
    ):
        _ = PatchDataset(img_list)
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"])

    # preproc func for not defined dataset
    with pytest.raises(
        ValueError,
        match=r".* preprocessing .* does not exist.",
    ):
        predefined_preproc_func("secret_dataset")


@pytest.mark.skip(reason="working, skip to run other test")
def test_DatasetInfo():  # Working
    """Test for kather patch dataset."""
    # test defining a subclas of dataset info but not defining
    # enforcing attributes, should crash
    with pytest.raises(TypeError):
        class Proto(ABCDatasetInfo):
            def __init__(self):
                self.a = 'a'
        Proto()

    # test kather with default init
    dataset = KatherPatchDataset()
    # kather with default data path skip download
    dataset = KatherPatchDataset()
    # pytest for not exist dir
    with pytest.raises(
        ValueError,
        match=r".*not exist.*",
    ):
        _ = KatherPatchDataset(save_dir_path="unknown_place")
    # save to temporary location
    save_dir_path = os.path.join(rcParam["TIATOOLBOX_HOME"], "tmp_check/")
    # remove prev generated data - just a test!
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path, ignore_errors=True)
    url = (
        "https://zenodo.org/record/53169/files/"
        "Kather_texture_2016_image_tiles_5000.zip"
    )
    save_zip_path = os.path.join(save_dir_path, "Kather.zip")
    download_data(url, save_zip_path)
    unzip_data(save_zip_path, save_dir_path)
    extracted_dir = os.path.join(save_dir_path, "Kather_texture_2016_image_tiles_5000/")
    dataset = KatherPatchDataset(save_dir_path=extracted_dir)
    assert dataset.input_list is not None
    assert dataset.label_list is not None
    assert dataset.label_name is not None
    assert len(dataset.input_list) == len(dataset.label_list)

    # remove generated data - just a test!
    shutil.rmtree(save_dir_path, ignore_errors=True)
    shutil.rmtree(rcParam["TIATOOLBOX_HOME"])


# # -------------------------------------------------------------------------------------
# # Command Line Interface
# # -------------------------------------------------------------------------------------


# def test_command_line_patch_predictor(_dir_sample_patches, _sample_patch1):
#     """Test for the patch predictor CLI."""
#     runner = CliRunner()
#     patch_predictor_dir = runner.invoke(
#         cli.main,
#         [
#             "patch-predictor",
#             "--pretrained_model",
#             "resnet18-kather100K",
#             "--img_input",
#             str(pathlib.Path(_dir_sample_patches)),
#             "--output_path",
#             "tmp_output",
#             "--batch_size",
#             2,
#             "--mode",
#             "patch",
#             "--return_probabilities",
#             False,
#         ],
#     )

#     assert patch_predictor_dir.exit_code == 0
#     shutil.rmtree("tmp_output", ignore_errors=True)

#     patch_predictor_single_path = runner.invoke(
#         cli.main,
#         [
#             "patch-predictor",
#             "--pretrained_model",
#             "resnet18-kather100K",
#             "--img_input",
#             pathlib.Path(_sample_patch1),
#             "--output_path",
#             "tmp_output",
#             "--batch_size",
#             2,
#             "--mode",
#             "patch",
#             "--return_probabilities",
#             False,
#         ],
#     )

#     assert patch_predictor_single_path.exit_code == 0
#     shutil.rmtree("tmp_output", ignore_errors=True)


# def test_command_line_patch_predictor_crash(_sample_patch1):
#     """Test for the patch predictor CLI."""
#     # test single image not exist
#     runner = CliRunner()
#     result = runner.invoke(
#         cli.main,
#         [
#             "patch-predictor",
#             "--pretrained_model",
#             "resnet18-kather100K",
#             "--img_input",
#             "imaginary_img.tif",
#             "--mode",
#             "patch",
#         ],
#     )
#     assert result.exit_code != 0

#     # test not pretrained model
#     result = runner.invoke(
#         cli.main,
#         [
#             "patch-predictor",
#             "--pretrained_model",
#             "secret_model",
#             "--img_input",
#             pathlib.Path(_sample_patch1),
#             "--mode",
#             "patch",
#         ],
#     )
#     assert result.exit_code != 0
