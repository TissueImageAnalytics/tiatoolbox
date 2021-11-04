# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****
"""Defines CNNs as used in IDaRS for prediction of molecular pathways and mutations."""


import numpy as np
from PIL import Image
from torchvision import transforms

from tiatoolbox import rcParam
from tiatoolbox.models.architecture.vanilla import CNNModel
from tiatoolbox.tools.stainnorm import get_normaliser
from tiatoolbox.utils.misc import download_data, imread

# fit Vahadane stain normalisation
TARGET_URL = "https://tiatoolbox.dcs.warwick.ac.uk/models/idars/target.jpg"
stain_normaliser = None
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.1, 0.1, 0.1]),
    ]
)
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.RandomResizedCrop(224),
#     transforms.RandomCrop(224),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.5, 0.5, 0.5],
#         std=[0.1, 0.1, 0.1])
# ])

download_data(TARGET_URL, save_path=f"{rcParam['TIATOOLBOX_HOME']}/idars_target.jpg")

target = imread(f"{rcParam['TIATOOLBOX_HOME']}/idars_target.jpg")
# stain_normaliser = get_normaliser(method_name="vahadane")
# stain_normaliser.fit(target)

import staintools

METHOD = 'vahadane'
STANDARDIZE_BRIGHTNESS = True
normalizer = staintools.StainNormalizer(method=METHOD)
ref = staintools.read_image('/home/tialab-dang/local/project/tiatoolbox/local/idars/source/target.jpg')
normalizer.fit(ref)


class CNNModel1(CNNModel):
    """Retrieve the model and add custom preprocessing.

    This is named CNNModel1 in line with the original IDaRS paper and
    is used for tumour segmentation.

    Args:
        backbone (str): Model name.
        num_classes (int): Number of classes output by model.

    """

    def __init__(self, backbone, num_classes=1):
        super().__init__(backbone, num_classes=num_classes)

    @staticmethod
    def preproc(img):
        img = transform(img)
        # toTensor will turn image to CHW so we transpose again
        img = img.permute(1, 2, 0)

        return img


class CNNModel2(CNNModel):
    """Retrieve the model and add custom preprocessing, including
      Vahadane stain normalisation.

      This is named CNNModel2 in line with the original IDaRS paper and
      is used for prediction of molecular pathways and key mutations.

    Args:
        backbone (str): Model name.
        num_classes (int): Number of classes output by model.

    """

    def __init__(self, backbone, num_classes=1):
        super().__init__(backbone, num_classes=num_classes)

    @staticmethod
    def preproc(img):
        # apply stain normalisation
        # try:
        #     luminous_normed = staintools.LuminosityStandardizer.standardize(img)
        #     stain_normed = normalizer.transform(luminous_normed)
        # except:
        #     stain_normed = img

        # # toTensor will turn image to CHW so we transpose again
        # img = img.permute(1, 2, 0)
        # transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomCrop(224),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.5, 0.5, 0.5],
        #         std=[0.1, 0.1, 0.1])
        # ])

        # img = Image.fromarray(stain_normed.copy())
        # img = stain_normaliser.transform(img.copy())

        img = transform(img)
        # toTensor will turn image to CHW so we transpose again
        img = img.permute(1, 2, 0)

        return img

