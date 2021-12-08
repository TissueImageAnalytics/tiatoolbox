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
from torchvision import transforms

from tiatoolbox.models.architecture.vanilla import CNNModel


TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1]),
    ]
)


class IDaRS(CNNModel):
    """Retrieve the model and add custom preprocessing used in IDaRS paper.

    Args:
        backbone (str): Model name.
        num_classes (int): Number of classes output by model.

    """

    def __init__(self, backbone, num_classes=1):
        super().__init__(backbone, num_classes=num_classes)

    @staticmethod
    # skipcq: PYL-W0221
    def preproc(img: np.ndarray):
        """Define preprocessing steps.

        Args:
            img (np.ndarray): An image of shape HWC.

        Return:
            img (torch.Tensor): An image of shape HWC.

        """
        img = img.copy()
        img = TRANSFORM(img)
        # toTensor will turn image to CHW so we transpose again
        img = img.permute(1, 2, 0)

        return img
