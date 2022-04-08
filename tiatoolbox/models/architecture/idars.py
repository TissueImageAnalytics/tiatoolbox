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
