"""Define CNNs as used in IDaRS for prediction of molecular pathways and mutations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchvision import transforms

from tiatoolbox.models.architecture.vanilla import CNNModel

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import torch

Transforms = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1]),
]

TRANSFORM = transforms.Compose(
    Transforms,
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

    def __init__(self: IDaRS, backbone: str, num_classes: int = 1) -> None:
        """Initialize :class:`IDaRS`."""
        super().__init__(backbone, num_classes=num_classes)

    @staticmethod
    def preproc(image: np.ndarray) -> torch.Tensor:
        """Define preprocessing steps.

        Args:
            image (:class:`numpy.ndarray`):
                An image of shape HWC.

        Return:
            image (:class:`torch.Tensor`):
                An image of shape HWC.

        """
        image = image.copy()
        image = TRANSFORM(image)
        # toTensor will turn image to CHW so we transpose again
        return image.permute(1, 2, 0)
