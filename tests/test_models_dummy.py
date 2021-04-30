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


from tiatoolbox.models.classification import CNN_Patch_Predictor, CNN_Patch_Model
from tiatoolbox.models.dataset import Patch_Dataset, Kather_Patch_Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# getitem test

# kather download test

# API call style 0 (do we need this?, the ultimate lazy way?)
# predictor = get_predictor('resnet18#kather', )
#   internally this will perform name mapping to create API call style 1

# API call style 1
# dataset = Kather_Patch_Dataset()
# predictor = CNN_Patch_Predictor(predefined_model="resnet18#kather", 
#                                 batch_size=16)
# output = predictor.predict(dataset, return_labels=True)

# API call style 2
# my_special_sauce_weights = '/home/tialab-dang/local/project/tiatoolbox/resnet18_kather.pth'
# dataset = Kather_Patch_Dataset()
# predictor = CNN_Patch_Predictor(predefined_model="resnet18#kather", 
#                                 pretrained_weight=my_special_sauce_weights, 
#                                 batch_size=16)
# output = predictor.predict(dataset, return_labels=True)

# # API call style 3 (advance)
dataset = Kather_Patch_Dataset()
model = CNN_Patch_Model(backbone='resnet50', nr_classes=9)
predictor = CNN_Patch_Predictor(model=model, batch_size=16)
output = predictor.predict(dataset, return_labels=True)

print('here')
