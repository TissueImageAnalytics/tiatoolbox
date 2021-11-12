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


from torchvision import transforms

from tiatoolbox import rcParam
from tiatoolbox.models.architecture.vanilla import CNNModel
from tiatoolbox.tools.stainnorm import get_normaliser
from tiatoolbox.utils.misc import download_data, imread

# fit Vahadane stain normalisation
TARGET_URL = "https://tiatoolbox.dcs.warwick.ac.uk/models/idars/target.jpg"
TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1]),
    ]
)

download_data(TARGET_URL, save_path=f"{rcParam['TIATOOLBOX_HOME']}/idars_target.jpg")
TARGET = imread(f"{rcParam['TIATOOLBOX_HOME']}/idars_target.jpg")
STAIN_NORMALIZER = get_normaliser(method_name="vahadane")
STAIN_NORMALIZER.fit(TARGET)


class CNNTumor(CNNModel):
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
        img = TRANSFORM(img)
        # toTensor will turn image to CHW so we transpose again
        img = img.permute(1, 2, 0)

        return img


class CNNMutation(CNNModel):
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
        img = img.copy()

        # apply stain normalisation
        # try:
        #     stain_normed = STAIN_NORMALIZER.transform(img)
        # # skipcq:
        # except:  # noqa
        #     # bad behavior when encountering blank image
        #     # which leads to numerical error problems,
        #     # we do not stain-normalize these cases
        #     stain_normed = img
        # stain_normed = STAIN_NORMALIZER.transform(img)
        # img = TRANSFORM(stain_normed)

        img = TRANSFORM(img)
        # toTensor will turn image to CHW so we transpose again
        img = img.permute(1, 2, 0)

        return img
