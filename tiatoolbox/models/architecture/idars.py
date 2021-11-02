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
"""Defines vanilla CNNs with torch backbones, mainly for patch classification."""


import numpy as np
from torchvision import transforms

from tiatoolbox.models.architecture.vanilla import CNNModel
from tiatoolbox.utils.misc import download_data, imread
from tiatoolbox.tools.stainnorm import get_normaliser

from tiatoolbox import rcParam


# fit Vahadane stain normalisation
TARGET_URL = "https://tiatoolbox.dcs.warwick.ac.uk/models/idars/target.jpg"
stain_normaliser = None
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1]),
    ]
)


class CNNModel1(CNNModel):
    def __init__(self, backbone, num_classes=1):
        super().__init__(backbone, num_classes=num_classes)

    @staticmethod
    def preproc(img):
        img = transform(img)
        # toTensor will turn image to CHW so we transpose again
        img = img.permute(1, 2, 0)

        return img


# class CNNModel2(CNNModel):
#     def __init__(self, backbone, num_classes=1):
#         super().__init__(backbone, num_classes=num_classes)

#         global stain_normaliser
#         stain_normaliser = None

#     @staticmethod
#     def preproc(img):
#         global stain_normaliser
#         if stain_normaliser is None:
#             target = download_data(
#                 TARGET_URL, save_path=f"{rcParam['TIATOOLBOX_HOME']}/idars_target.jpg"
#             )
#         target = imread(f"{rcParam['TIATOOLBOX_HOME']}/idars_target.jpg")
#         target = target.copy()  # create a copy of the target image
#         target = target.astype("uint8")
#         img = img.copy()
#         img = img.astype("uint8")
#         print(img.shape, target.shape)
#         stain_normaliser = get_normaliser(method_name="vahadane")
#         stain_normaliser.fit(target)
#         # apply stain normalisation
#         img = stain_normaliser.transform(img)

#         img = transform(img)
#         # toTensor will turn image to CHW so we transpose again
#         img = img.permute(1, 2, 0)

#         return img


class CNNModel2(CNNModel):
    def __init__(self, backbone, num_classes=1):
        super().__init__(backbone, num_classes=num_classes)

        global target
        # download stain normalisation target image used for IDaRS
        download_data(
            TARGET_URL, save_path=f"{rcParam['TIATOOLBOX_HOME']}/idars_target.jpg"
        )
        target = imread(f"{rcParam['TIATOOLBOX_HOME']}/idars_target.jpg")

    @staticmethod
    def preproc(img):
        global target

        # get stain normalisation method
        stain_normaliser = get_normaliser(method_name="vahadane")
        stain_normaliser.fit(target)
        # apply stain normalisation
        img = stain_normaliser.transform(img.copy())

        img = transform(img)
        # toTensor will turn image to CHW so we transpose again
        img = img.permute(1, 2, 0)

        return img
