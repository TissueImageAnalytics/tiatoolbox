import random
import os
import glob
import cv2
import os

import torch
import torch.utils.data as torch_data
import torchvision.transforms as transforms

import sys

sys.path.append("..")


from tiatoolbox.models.classification import CNN_Patch_Predictor


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

random.seed(5)
model = CNN_Patch_Predictor(
    backbone="resnet18", pretrained="kather", batch_size=16, nr_loader_worker=4
)
output = model.predict("Kather_Sample/all/", return_names=True)
print(output)
