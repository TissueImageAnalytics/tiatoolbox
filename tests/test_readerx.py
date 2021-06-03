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
from tiatoolbox.wsicore.wsireader import get_wsireader

import matplotlib.pyplot as plt

root_dir = '/home/tialab-dang/workstation_storage_1/workspace/tiatoolbox/'
wsi_path = '%s/tests/local_samples/PCA_mini.svs' % root_dir
assert os.path.exists(wsi_path)

wsi_reader = get_wsireader(wsi_path)
# bound_x = np.array([1024, 1024, 2048, 2048])
# bound_y = np.array([ 512,  512, 1024, 1024])
# roi_x = wsi_reader.read_bounds(bound_x, resolution=1.0, units='level')
# roi_y = wsi_reader.read_bounds_at_requested(bound_y, resolution=1.0, units='level')

read_mpp = 0.75
mpp_fx = wsi_reader.info.mpp[0] / read_mpp
bound_x = (np.array([1024, 1024, 2048, 2048])).astype(np.int64)
bound_y = (np.array([1024, 1024, 2048, 2048]) * mpp_fx).astype(np.int64)
roi_x = wsi_reader.read_bounds(bound_x, resolution=read_mpp, units='mpp')
roi_y = wsi_reader.read_bounds_at_requested(bound_y, resolution=read_mpp, units='mpp')

plt.subplot(1,2,1)
plt.imshow(roi_x)
plt.subplot(1,2,2)
plt.imshow(roi_y)
plt.savefig('dump.png')