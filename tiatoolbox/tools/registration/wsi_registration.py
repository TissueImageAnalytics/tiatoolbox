from numbers import Number
from typing import Tuple, Union

import cv2
import numpy as np
import scipy.ndimage as ndi
import torchvision
from skimage import exposure, filters, morphology
from skimage.util import img_as_float
from torchvision.models._utils import IntermediateLayerGetter

Resolution = Union[Number, Tuple[Number, Number], np.ndarray]


def preprocess(fixed_img, moving_img):
    """This function performs normalization to unify the appearance of fixed
     and moving images

    Args:
            fixed_img (:class:`numpy.ndarray`):
                    A grayscale fixed image
            moving_img (:class:`numpy.ndarray`):
                    A grayscale moving image

    Returns:
            :class:`numpy.ndarray`:
                    A normalized grayscale fixed image
            :class:`numpy.ndarray`:
                    A normalized grayscale moving image

    """
    if len(fixed_img.shape) != 2 or len(moving_img.shape) != 2:
        raise ValueError(f'{"The input images should be grayscale images."}')

    moving_entropy, fixed_entropy = filters.rank.entropy(
        moving_img, morphology.disk(3)
    ), filters.rank.entropy(fixed_img, morphology.disk(3))
    if np.mean(fixed_entropy) > np.mean(moving_entropy):
        moving_img = exposure.match_histograms(moving_img, fixed_img)
    else:
        fixed_img = exposure.match_histograms(fixed_img, moving_img)

    return fixed_img, moving_img


def dice(mask_1, mask_2):
    """This function computes dice coefficent between two masks."""
    if mask_1.shape != mask_2.shape:
        x_size = max(mask_1.shape[1], mask_2.shape[1])
        y_size = max(mask_1.shape[0], mask_2.shape[0])

        right_y, right_x = y_size - mask_1.shape[0], x_size - mask_1.shape[1]
        mask_1 = np.pad(mask_1, [(0, right_y), (0, right_x)], mode="constant")

        right_y, right_x = y_size - mask_2.shape[0], x_size - mask_2.shape[1]
        mask_2 = np.pad(mask_2, [(0, right_y), (0, right_x)], mode="constant")

    mask_1 = mask_1.astype(np.bool)
    mask_2 = mask_2.astype(np.bool)
    return 2 * np.logical_and(mask_1, mask_2).sum() / (mask_1.sum() + mask_2.sum())


def prealignment(fixed_img, moving_img, fixed_mask, moving_mask):
    """This function performs coarse registration of a pair of images

    Args:
            fixed_img (:class:`numpy.ndarray`):
                    A grayscale fixed image
            moving_img (:class:`numpy.ndarray`):
                    A grayscale moving image
            fixed_mask (:class:`numpy.ndarray`):
                    A binary tissue mask for the fixed image
            moving_mask (:class:`numpy.ndarray`):
                    A binary tissue mask for the moving image

    Returns:
            :class:`numpy.ndarray`:
                    A transform matrix

    """
    if len(fixed_img.shape) != 2 or len(moving_img.shape) != 2:
        raise ValueError(f'{"The input images should be grayscale images."}')

    if fixed_img.shape != fixed_img.shape or moving_img.shape != moving_mask.shape:
        raise ValueError(
            f'{"Mismatch of shape between image and its corresponding mask."}'
        )

    fixed_img = exposure.rescale_intensity(img_as_float(fixed_img), in_range=(0, 1))
    moving_img = exposure.rescale_intensity(img_as_float(moving_img), in_range=(0, 1))

    cy, cx = ndi.center_of_mass((1 - fixed_img) * fixed_mask)
    fixed_com = [cx, cy]

    cy, cx = ndi.center_of_mass((1 - moving_img) * moving_mask)
    moving_com = [cx, cy]

    com_transform = np.array(
        [
            [1, 0, fixed_com[0] - moving_com[0]],
            [0, 1, fixed_com[1] - moving_com[1]],
            [0, 0, 1],
        ]
    )
    origin_transform_com_ = [[1, 0, -fixed_com[0]], [0, 1, -fixed_com[1]], [0, 0, 1]]
    origin_transform_com = [[1, 0, fixed_com[0]], [0, 1, fixed_com[1]], [0, 0, 1]]

    all_dice = []
    all_transform = []
    list_angles = np.arange(10, 360, 10).tolist()
    for i in range(len(list_angles)):
        theta = np.radians(list_angles[i])
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

        transform = np.matmul(
            np.matmul(
                np.matmul(origin_transform_com, rotation_matrix), origin_transform_com_
            ),
            com_transform,
        )
        warped_moving_mask = cv2.warpAffine(
            moving_mask, transform[0:-1][:], fixed_img.shape[:2][::-1]
        )
        dice_com = dice(fixed_mask, warped_moving_mask)

        all_dice.append(dice_com)
        all_transform.append(transform)

    return all_transform[all_dice.index(max(all_dice))]


class RegistrationConfig:
    """Contains information required for performing wsi registration

    Args:
            input_image_size (tuple):
                    Size of the image input to the feature_extractor.
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
            ...     input_image_size = (224, 224)
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
    input_image_size = None

    def __init__(
        self,
        resolution: Resolution = 0.03125,
        units: str = "power",
        number_of_rotations: int = 10,
        input_image_size: Tuple[int, int] = (224, 224),
    ):
        self.resolution = resolution
        self.units = units
        self.number_of_rotations = number_of_rotations
        self.input_image_size = input_image_size
        self._validate()

    def _validate(self):
        """Validate the data format."""
        if self.units not in [
            "power",
            "baseline",
            "mpp",
        ]:
            raise ValueError(f"Invalid resolution units `{self.units}`.")
