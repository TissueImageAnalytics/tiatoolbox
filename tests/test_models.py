import random
import os
import glob
import cv2
import os

import torch
import torch.utils.data as torch_data
import torchvision.transforms as transforms

import sys

sys.path.append(".")


from tiatoolbox.models.classification import CNN_Patch_Predictor


class Kather_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, preproc=None):
        super().__init__()
        if preproc is None:
            self.preproc = lambda x: x
        else:
            self.preproc = preproc
        label_code_list = [
            "01_TUMOR ",
            "02_STROMA",
            "03_COMPLEX" "04_LYMPHO",
            "05_DEBRIS",
            "06_MUCOSA",
            "07_ADIPOSE" "08_EMPTY",
        ]
        all_path_list = []
        for label_id, label_code in enumerate(label_code_list):
            path_list = glob.glob("%s/%s/*.tif" % (root_dir, label_code))
            path_list = [[v, label_id] for v in path_list]
            path_list.sort()
            all_path_list.extend(path_list)

        random.shuffle(all_path_list)  # HACK to creat randomness
        self.path_list = all_path_list[:512]  # HACK to test 32 images for now
        return

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path, label = self.path_list[idx]
        image = cv2.imread(path)
        image = self.preproc(image)
        return image


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ds = Kather_Dataset(root_dir="dataset/sample_exp/data/Kather_Sample/")

from tiatoolbox.models.backbone import get_model
pretrained_path = 'dataset/sample_exp/models/Kather_resnet18.h5.pth'
pretrained = torch.load(pretrained_path)['model']

new_dict = {}
for k, v in pretrained.items():
    new_dict[k[2:]] = v
    
model = get_model('resnet18')
model.load_state_dict(new_dict)

random.seed(5)
model = CNN_Patch_Predictor(
    backbone="resnet18", nr_class=9, batch_size=16, nr_loader_worker=4
)
model.load_model("toolbox_weights/resnet18_kather.pth")
output = model.predict_dataset(ds)
