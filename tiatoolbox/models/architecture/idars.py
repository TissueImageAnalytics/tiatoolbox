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
    """Initialise IDaRS and add custom preprocessing as used in the original paper [1].

    The tiatoolbox model should produce the following results:

    .. list-table:: IDaRS performance measured by AUROC.
       :widths: 15 15 15 15 15 15 15
       :header-rows: 1

       * -
         - MSI
         - TP53
         - BRAF
         - CIMP
         - CIN
         - HM
       * - Bilal et al.
         - 0.828
         - 0.755
         - 0.813
         - 0.853
         - 0.860
         - 0.846
       * - TIAToolbox
         - 0.870
         - 0.747
         - 0.750
         - 0.748
         - 0.810
         - 0.790

    Args:
        backbone (str):
            Model name.
        num_classes (int):
            Number of classes output by model.

    References:
        [1] Bilal, Mohsin, et al. "Development and validation of a weakly supervised
        deep learning framework to predict the status of molecular pathways and key
        mutations in colorectal cancer from routine histology images: a retrospective
        study." The Lancet Digital Health 3.12 (2021): e763-e772.

    """

    def __init__(self, backbone, num_classes=1):
        super().__init__(backbone, num_classes=num_classes)

    @staticmethod
    # noqa: E800
    def preproc(image: np.ndarray):
        """Define preprocessing steps.

        Args:
            img (:class:`numpy.ndarray`):
                An image of shape HWC.

        Return:
            img (:class:`torch.Tensor`):
                An image of shape HWC.

        """
        image = image.copy()
        image = TRANSFORM(image)
        # toTensor will turn image to CHW so we transpose again
        return image.permute(1, 2, 0)
