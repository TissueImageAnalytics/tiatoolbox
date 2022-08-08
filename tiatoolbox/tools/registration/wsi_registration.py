import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage import exposure
from skimage.util import img_as_float

from tiatoolbox.utils.metrics import dice


def _check_dimen(fixed_img, moving_img, fixed_mask, moving_mask):
    """Checking dimensionality of images and mask."""
    if len(np.unique(fixed_mask)) == 1 or len(np.unique(fixed_mask)) == 1:
        raise ValueError("The foreground is missing in the mask.")

    if len(fixed_img.shape) != 2 or len(moving_img.shape) != 2:
        raise ValueError("The input images should be grayscale images.")

    if (
        fixed_img.shape[:] != fixed_mask.shape[:]
        or moving_img.shape[:] != moving_mask.shape[:]
    ):
        raise ValueError("Mismatch of shape between image and its corresponding mask.")


def prealignment(
    fixed_img, moving_img, fixed_mask, moving_mask, dice_overlap=0.5, rotation_step=10
):
    """Coarse registration of an image pair.

    This function performs initial alignment of a moving image with respect to a
    fixed image. This can be used as a prealignment step before final refinement.

    Args:
        fixed_img (:class:`numpy.ndarray`):
            A grayscale fixed image.
        moving_img (:class:`numpy.ndarray`):
            A grayscale moving image.
        fixed_mask (:class:`numpy.ndarray`):
            A binary tissue mask for the fixed image.
        moving_mask (:class:`numpy.ndarray`):
            A binary tissue mask for the moving image.
        dice_overlap (float):
            A dice ratio used for the selection of the best
            transformation matrix.
        rotation_step (int):
            Rotation_step defines an increment in the rotation angles.
    Returns:
        :class:`numpy.ndarray`:
            A transform matrix.

    """
    if len(fixed_mask.shape) != 2:
        fixed_mask = fixed_mask[:, :, 0]
    if len(moving_mask.shape) != 2:
        moving_mask = moving_mask[:, :, 0]

    fixed_img = np.squeeze(fixed_img)
    moving_img = np.squeeze(moving_img)

    fixed_mask = np.uint8(fixed_mask > 0)
    moving_mask = np.uint8(moving_mask > 0)

    if rotation_step < 10 or rotation_step > 20:
        raise ValueError("Please select the rotation step in between 10 and 20.")

    if dice_overlap < 0 or dice_overlap > 1:
        raise ValueError("The dice_overlap should be in between 0 and 1.0.")

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
    for angle in np.arange(0, 360, rotation_step).tolist():
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

    if max(all_dice) >= dice_overlap:
        return all_transform[all_dice.index(max(all_dice))]

    print(
        "***** Not able to find the best transformation. Try changing the values"
        "for dice_overlap and rotation_step. *****"
    )
    return None
