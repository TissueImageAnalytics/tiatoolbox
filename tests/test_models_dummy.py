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

sys.path.append("..")


from tiatoolbox.models.classification import CNN_Patch_Predictor
from tiatoolbox.models.dataset import Patch_Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test Dataset
np.random.seed(5)
# same shape test
img_list = np.random.randint(0, 255, (4, 512, 512, 3))
# list of nd.array test
img_list = [np.random.randint(0, 255, (4, 4, 3)),
            np.random.randint(0, 255, (4, 4, 3)),]
# random shape test
img_list = [np.random.randint(0, 255, (4, 4, 3)),
            np.random.randint(0, 255, (4, 5, 3)),]
# mixed dtype test
img_list = [np.random.randint(0, 255, (4, 4, 3)),
            'you_should_crash_here', 123, 456]
# mixed dtype test
img_list = ['you_should_crash_here', 123, 456]
# mixed dtype test
img_list = [np.random.randint(0, 255, (4, 4, 3)),
            'you_should_crash_here',]
# path to same shape img

# path to variable shape img

# valid path test
Patch_Dataset(img_list)

# getitem test

# kather download test

# kather return label test
random.seed(5)
model = CNN_Patch_Predictor(
    backbone="resnet18", pretrained="kather", batch_size=16, nr_loader_worker=4
)
output = model.predict("Kather_Sample/all/", return_names=True)
print(output)
