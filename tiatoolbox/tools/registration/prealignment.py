import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage import exposure
from skimage.util import img_as_float

from tiatoolbox.utils.metrics import dice


def prealignment(fixed_img, moving_img, fixed_mask, moving_mask, rotation_steps=10):
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
    if len(fixed_mask.shape) is not 2:
        fixed_mask = fixed_mask[:, :, 0]
    if len(moving_mask.shape) is not 2:
        moving_mask = moving_mask[:, :, 0]

    fixed_mask, moving_mask = np.uint8(fixed_mask > 0), np.uint8(moving_mask > 0)

    if len(np.unique(fixed_mask)) == 1 or len(np.unique(fixed_mask)) == 1:
        raise ValueError(f'{"The foreground is missing in the mask."}')

    if len(fixed_img.shape) is not 2 or len(moving_img.shape) is not 2:
        raise ValueError(f'{"The input images should be grayscale images."}')

    if fixed_img.shape is not fixed_mask.shape or moving_img.shape is not moving_mask.shape:
        raise ValueError(
            f'{"Mismatch of shape between image and its corresponding mask."}'
        )

    if rotation_steps < 10 or rotation_steps > 20:
        raise ValueError(f'{"Please select the rotation step in between 10 and 20."}')

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
    for angle in np.arange(0, 360, rotation_steps).tolist():
        theta = np.radians(angle)
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
