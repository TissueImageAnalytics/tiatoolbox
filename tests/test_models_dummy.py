import random
import os
import glob
import cv2
import os
import numpy as np

import torch
import torch.utils.data as torch_data
import torchvision.transforms as transforms

import sys

sys.path.append(".")


from tiatoolbox.models.classification import CNN_Patch_Predictor
from tiatoolbox.models.dataset import Patch_Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test Dataset
np.random.seed(5)
set_of_img_list = [
    # ndarray same shape test
    np.random.randint(0, 255, (4, 512, 512, 3)),
    # ndarray of mixed type
    np.array([
        np.random.randint(0, 255, (4, 5, 3)),
        'Should crash',
    ]),
    # list of same shape ndarray test
    [np.random.randint(0, 255, (4, 4, 3)),
     np.random.randint(0, 255, (4, 4, 3)),],
    # list of random shape test
    [np.random.randint(0, 255, (4, 4, 3)),
     np.random.randint(0, 255, (4, 5, 3)),],
    # list of mixed dtype test
    [np.random.randint(0, 255, (4, 4, 3)),
    'you_should_crash_here', 123, 456],
    # list of mixed dtype test
    ['you_should_crash_here', 123, 456],
]
# path to same shape img

# path to variable shape img

# valid path test
flag_list = []
for img_list in set_of_img_list:
    try:
        Patch_Dataset(img_list)
    except:
        flag_list.append(1)
        continue
    # Patch_Dataset(set_of_img_list[2])
    flag_list.append(0)
print('here')

# getitem test

# kather download test

# kather return label test
# random.seed(5)
# model = CNN_Patch_Predictor(
#     backbone="resnet18", pretrained="kather", batch_size=16, nr_loader_worker=4
# )
# output = model.predict(img_list, return_names=True)
# print(output)
