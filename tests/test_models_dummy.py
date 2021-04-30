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
from tiatoolbox.models.dataset import Patch_Dataset, Kather_Patch_Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# getitem test

# kather download test

# kather return label test
random.seed(5)
# model = CNN_Patch_Predictor(
#     backbone="resnet18", pretrained="kather", batch_size=16, nr_loader_worker=4
# )

dataset = Kather_Patch_Dataset()
model = CNN_Patch_Predictor(pretrained_model="resnet18_kather_pc", batch_size=16)
output = model.predict(dataset, return_labels=True)


model = get_patch_predictor(pretrained_model="resnet18_kather_pc", batch_size=16)
