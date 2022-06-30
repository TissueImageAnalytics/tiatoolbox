from typing import List

import torchvision
from torchvision.models._utils import IntermediateLayerGetter


class RegisterationConfig:
    """Contains information required for performing wsi registration

            Args:
                    patch_shape (class:`numpy.ndarray`, list(int)):
                            Shape of the input in (height, width).
                    resolution (dict):
                            Resolution for performing registration using DFBR approach
                    number_of_rotations (int):
                            Number of rotations for the pre-alignment step

    Examples:
            >>> reg_config = RegisterationConfig(
            ...     resolution=[{"units": "mpp", "resolution": 0.3125}],
            ...     number_of_rotations = 10,
            ...     patch_shape=[224, 224]
            ... )
    """

    model = torchvision.models.vgg16(True)
    feature_extractor = IntermediateLayerGetter(
        model.features,
        return_layers={"16": "block3_pool", "23": "block4_pool", "30": "block5_pool"},
    )
    # We pre-define to follow enforcement, actual initialisation in init
    resolution = None
    number_of_rotations = None
    patch_shape = None

    def __init__(
        self,
        resolution: float = {"units": "mpp", "resolution": 0.03125},
        number_of_rotations: int = 10,
        patch_shape: List[int] = [224, 224],
    ):
        self.resolution = resolution
        self.number_of_rotations = number_of_rotations
        self.patch_shape = patch_shape
        self._validate()

    def _validate(self):
        """Validate the data format."""
        units = self.resolution[0]["units"]
        if units not in [
            "power",
            "baseline",
            "mpp",
        ]:
            raise ValueError(f"Invalid resolution units `{units}`.")
