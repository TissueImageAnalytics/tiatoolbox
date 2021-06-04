import os
import pathlib
import shutil
import sys

import cv2
import numpy as np
import pytest
import torch

# sys.path.append('.')
sys.path.append('..')

from tiatoolbox.utils.misc import imwrite
from tiatoolbox.wsicore.wsireader import get_wsireader, VirtualWSIReader
from tiatoolbox.utils.misc import download_data, imread, unzip_data

import matplotlib.pyplot as plt

# root_dir = '/home/tialab-dang/workstation_storage_1/workspace/tiatoolbox/'
# wsi_path = '%s/tests/local_samples/PCA_mini.svs' % root_dir
# assert os.path.exists(wsi_path)

# wsi_reader = get_wsireader(wsi_path)
# bound_x = np.array([1024, 1024, 2048, 2048])
# bound_y = np.array([ 512,  512, 1024, 1024])
# roi_x = wsi_reader.read_bounds(bound_x, resolution=1.0, units='level')
# roi_y = wsi_reader.read_bounds_at_requested(bound_y, resolution=1.0, units='level')

# read_mpp = 0.75
# mpp_fx = wsi_reader.info.mpp[0] / read_mpp
# bound_x = (np.array([1024, 1024, 2048, 2048])).astype(np.int64)
# bound_y = (np.array([1024, 1024, 2048, 2048]) * mpp_fx).astype(np.int64)
# roi_x = wsi_reader.read_bounds(bound_x, resolution=read_mpp, units='mpp')
# roi_y = wsi_reader.read_bounds_at_requested(bound_y, resolution=read_mpp, units='mpp')

# read_mpp = 0.75
# mpp_fx = wsi_reader.info.mpp[0] / read_mpp
# bound_x = (np.array([1024, 1024, 2048, 2048])).astype(np.int64)
# bound_y = (np.array([1024, 1024, 2048, 2048]) * mpp_fx).astype(np.int64)
# roi_x = wsi_reader.read_bounds(bound_x, resolution=read_mpp, units='mpp')
# roi_y = wsi_reader.read_bounds(bound_y, resolution=read_mpp, units='mpp', location_is_at_requested=True)

# plt.subplot(1,2,1)
# plt.imshow(roi_x)
# plt.subplot(1,2,2)
# plt.imshow(roi_y)
# plt.savefig('dump.png')

# def test_sync_VirtualReader_read(_mini_wsi1_svs, _mini_wsi1_jpg, _mini_wsi1_msk):
    # """Test synchronize read for VirtualReader"""
    # _mini_wsi1_svs = pathlib.Path(_mini_wsi1_svs)
    # _mini_wsi1_msk = pathlib.Path(_mini_wsi1_msk)
    # _mini_wsi1_jpg = pathlib.Path(_mini_wsi1_jpg)

_mini_wsi1_svs = pathlib.Path('/home/tialab-dang/workstation_storage_1/workspace/tiatoolbox/tests/local_samples/CMU-mini.svs')
_mini_wsi1_msk = pathlib.Path('/home/tialab-dang/workstation_storage_1/workspace/tiatoolbox/tests/local_samples/CMU-mask.png')
_mini_wsi1_jpg = pathlib.Path('/home/tialab-dang/workstation_storage_1/workspace/tiatoolbox/tests/local_samples/CMU-mini.jpg')

wsi_reader = get_wsireader(_mini_wsi1_svs)

msk = imread(_mini_wsi1_msk)
msk_reader = VirtualWSIReader(msk)
old_metadata = msk_reader.info
msk_reader.attach_to_reader(wsi_reader.info)
# check that attach altered vreader metadata
assert np.any(old_metadata.mpp != msk_reader.info.mpp)

# now check sync read by comparing the RoI with different base
# the output should be at same resolution even if source is of different base
bigger_msk = cv2.resize(msk, (0, 0), fx=4.0, fy=4.0,
                        interpolation=cv2.INTER_NEAREST)
bigger_msk_reader = VirtualWSIReader(bigger_msk)
# * must set mpp metadata to not None else wont work
# error checking first
ref_metadata = bigger_msk_reader.info
ref_metadata.mpp = 1.0
ref_metadata.objective_power = None
with pytest.raises(ValueError, match=r".*objective.*None.*"):
    msk_reader.attach_to_reader(ref_metadata)
ref_metadata.mpp = None
ref_metadata.objective_power = 1.0
with pytest.raises(ValueError, match=r".*mpp.*None.*"):
    msk_reader.attach_to_reader(ref_metadata)

# must set mpp metadata to not None else wont
# !?! why do this doesn modify ?, but modify
# !!! reference above seem to work? @John
ref_metadata.mpp = 1.0
ref_metadata.objective_power = 1.0
msk_reader.attach_to_reader(ref_metadata)

shape2 = bigger_msk_reader.slide_dimensions(resolution=0.75, units='mpp')
shape1 = msk_reader.slide_dimensions(resolution=0.75, units='mpp')
assert shape1[0] - shape2[0] < 10  # offset may happen if shape is not multiple
assert shape1[1] - shape2[1] < 10  # offset may happen if shape is not multiple
shape2 = bigger_msk_reader.slide_dimensions(resolution=0.75, units='power')
shape1 = msk_reader.slide_dimensions(resolution=0.75, units='power')
assert shape1[0] - shape2[0] < 10  # offset may happen if shape is not multiple
assert shape1[1] - shape2[1] < 10  # offset may happen if shape is not multiple

# ! box should be within image
read_coords = np.array([3500, 3000, 5500, 7000])
# with mpp
roi1 = bigger_msk_reader.read_bounds(
            read_coords,
            resolution=0.25,
            units='mpp',
            location_is_at_requested=True
        )
roi2 = msk_reader.read_bounds(
            read_coords,
            resolution=0.25,
            units='mpp',
            location_is_at_requested=True,
        )
print(roi1.shape)
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow(roi1)
plt.subplot(1, 2, 2)
plt.imshow(roi2)
plt.savefig('dump1.png')
assert roi1.shape[0] == roi2.shape[0]
assert roi1.shape[1] == roi2.shape[1]
cc = np.corrcoef(roi1[..., 0].flatten(), roi2[..., 0].flatten())
assert np.min(cc) > 0.95, cc
# with objective
roi1 = bigger_msk_reader.read_bounds(
            read_coords,
            resolution=3.0,
            units='power',
            location_is_at_requested=True
        )
roi2 = msk_reader.read_bounds(
            read_coords,
            resolution=3.0,
            units='power',
            location_is_at_requested=True
        )
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow(roi1)
plt.subplot(1, 2, 2)
plt.imshow(roi2)
plt.savefig('dump2.png')
assert roi1.shape[0] == roi2.shape[0]
assert roi1.shape[1] == roi2.shape[1]
cc = np.corrcoef(roi1[..., 0].flatten(), roi2[..., 0].flatten())
assert np.min(cc) > 0.95, cc

    # * now check attaching and read to WSIReader and varying resolution
    # need to think how to check correctness
    # lv0_coords = np.array([4500, 9500, 6500, 11500])
    # msk_reader.attach_to_reader(wsi_reader.info)
    # msk_reader.read_bounds(
    #         lv0_coords / scale_wrt_ref,
    #         resolution=15.0,
    #         units='power'
    #     )
    # msk_reader.read_bounds(
    #         lv0_coords / scale_wrt_ref,
    #         resolution=1.0,
    #         units='mpp'
    #     )
    # msk_reader.read_bounds(
    #         lv0_coords / scale_wrt_ref,
    #         resolution=1.0,
    #         units='baseline'
    #     )

    # patch_shape = [512, 512]
    # # now check normal reading for dataset with mask
    # item_list = []
    # ds = WSIPatchDataset(
    #     _mini_wsi1_svs,
    #     mode='wsi',
    #     mask_path=_mini_wsi1_msk,
    #     patch_shape=patch_shape,
    #     stride_shape=patch_shape,
    #     resolution=1.0,
    #     units='mpp')
    # item_list.append(ds[10])
    # ds = WSIPatchDataset(
    #     _mini_wsi1_svs,
    #     mode='wsi',
    #     mask_path=_mini_wsi1_msk,
    #     patch_shape=patch_shape,
    #     stride_shape=patch_shape,
    #     resolution=1.0,
    #     units='baseline')
    # item_list.append(ds[10])
    # ds = WSIPatchDataset(
    #     _mini_wsi1_svs,
    #     mode='wsi',
    #     mask_path=_mini_wsi1_msk,
    #     patch_shape=patch_shape,
    #     stride_shape=patch_shape,
    #     resolution=15.0,
    #     units='power')
    # item_list.append(ds[10])

    # # * now check sync read for tile ans wsi
    # patch_shape = np.array([2048, 2048])
    # wds = WSIPatchDataset(
    #     _mini_wsi1_svs,
    #     mask_path=_mini_wsi1_msk,
    #     mode="wsi",
    #     patch_shape=patch_shape,
    #     stride_shape=patch_shape,
    #     resolution=1.0,
    #     units="baseline",
    # )
    # tds = WSIPatchDataset(
    #     _mini_wsi1_jpg,
    #     mask_path=_mini_wsi1_msk,
    #     mode="tile",
    #     patch_shape=patch_shape,
    #     stride_shape=patch_shape,
    #     resolution=1.0,
    #     units="baseline",
    # )
    # assert len(wds) == len(tds)
    # # now loop over each read and ensure they look similar
    # num_sample = len(wds)
    # for idx in range(num_sample):
    #     cc = np.corrcoef(
    #             cv2.cvtColor(wds[idx]['image'], cv2.COLOR_RGB2GRAY).flatten(),
    #             cv2.cvtColor(tds[idx]['image'], cv2.COLOR_RGB2GRAY).flatten())
    #     assert np.min(cc) > 0.95, (cc, idx)