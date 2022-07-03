from numbers import Number
from typing import List, Tuple, Union

import numpy as np
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

Resolution = Union[Number, Tuple[Number, Number], np.ndarray]


class RegisterationConfig:
    """Contains information required for performing wsi registration

            Args:
                    patch_shape (class:`numpy.ndarray`, list(int)):
                            Shape of the input in (height, width).
                    resolution (int or float or tuple(float)):
                            Resolution to perform registration at,
                            default = 0.01325 (objective power).
                    units (str):
                            resolution units, default="power"
                    number_of_rotations (int):
                            Number of rotations for the pre-alignment step

    Examples:
            >>> reg_config = RegisterationConfig(
            ...     resolution = 0.3125,
            ...		units = 'power',
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
    units = None
    number_of_rotations = None
    patch_shape = None

    def __init__(
        self,
        resolution: Resolution = 0.03125,
        units: str = "power",
        number_of_rotations: int = 10,
        patch_shape: List[int] = None,
    ):
        self.resolution = resolution
        self.units = units
        self.number_of_rotations = number_of_rotations
        if patch_shape is None:
            self.patch_shape = []
            patch_shape = [224, 224]
        self.patch_shape = patch_shape
        self._validate()

    def _validate(self):
        """Validate the data format."""
        if self.units not in [
            "power",
            "baseline",
            "mpp",
        ]:
            raise ValueError(f"Invalid resolution units `{self.units}`.")
