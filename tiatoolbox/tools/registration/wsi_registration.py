import warnings

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
import torchvision
from skimage import exposure, filters
from skimage.util import img_as_float
from torchvision.models._utils import IntermediateLayerGetter

from tiatoolbox.utils.metrics import dice
from tiatoolbox.utils.transforms import imresize


def _check_dims(fixed_img, moving_img, fixed_mask, moving_mask):
    """Check the dimensionality of images and mask.

    Args:
        fixed_img (:class:`numpy.ndarray`):
            A grayscale fixed image.
        moving_img (:class:`numpy.ndarray`):
            A grayscale moving image.
        fixed_mask (:class:`numpy.ndarray`):
            A binary tissue mask for the fixed image.
        moving_mask (:class:`numpy.ndarray`):
            A binary tissue mask for the moving image.

    Returns:
        None

    """
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
            Dice ratio used for the selection of the best
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

    _check_dims(fixed_img, moving_img, fixed_mask, moving_mask)

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

    warnings.warn(
        "Not able to find the best transformation. Try changing the values for"
        " 'dice_overlap' and 'rotation_step'."
    )
    return None


def match_histograms(image_a, image_b, kernel_size=7):
    """Image normalization function.

    This function performs histogram equalization to unify the
    appearance of an image pair.

    Args:
        image_a (:class:`numpy.ndarray`):
            A grayscale image.
        image_b (:class:`numpy.ndarray`):
            A grayscale image.
        kernel_size (int):
            The size of the ellipse-shaped footprint.

    Returns:
        :class:`numpy.ndarray`:
            A normalized grayscale image.
        :class:`numpy.ndarray`:
            A normalized grayscale image.

    """

    image_a, image_b = np.squeeze(image_a), np.squeeze(image_b)
    if len(image_a.shape) == 3 or len(image_b.shape) == 3:
        raise ValueError("The input images should be grayscale images.")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    entropy_a, entropy_b = filters.rank.entropy(image_a, kernel), filters.rank.entropy(
        image_b, kernel
    )
    if np.mean(entropy_a) > np.mean(entropy_b):
        image_b = exposure.match_histograms(image_b, image_a).astype(np.uint8)
    else:
        image_a = exposure.match_histograms(image_a, image_b).astype(np.uint8)

    return image_a, image_b


class DFBRegistrtation:
    r"""Implements Deep Feature based Registration.

    This class implements a CNN feature based registration,
    as proposed in a paper titled `Deep Feature based Cross-slide Registration
    <https://arxiv.org/pdf/2202.09971.pdf>`_.

    """

    def __init__(self):
        self.patch_size = (224, 224)
        self.Xscale, self.Yscale = [], []
        model = torchvision.models.vgg16(True)
        return_layers = {"16": "block3_pool", "23": "block4_pool", "30": "block5_pool"}
        self.FeatureExtractor = IntermediateLayerGetter(
            model.features, return_layers=return_layers
        )

    # Make this function private when full pipeline is implemented.
    def extract_features(self, fixed_img, moving_img):
        """CNN based feature extraction for registration.

        This function extracts multiscale features from a pre-trained
        VGG-16 model for an image pair.

        Args:
            fixed_img (:class:`numpy.ndarray`):
                A fixed image.
            moving_img (:class:`numpy.ndarray`):
                A moving image.

        Returns:
            dict:
                A dictionary containing the multiscale features.
                The expected format is {layer_name: features}.

        """
        if len(fixed_img.shape) != 3 or len(moving_img.shape) != 3:
            raise ValueError(
                "The required shape for fixed and moving images is n x m x 3."
            )

        if fixed_img.shape[2] != 3 or moving_img.shape[2] != 3:
            raise ValueError("The input images are expected to have 3 channels.")

        self.Xscale = 1.0 * np.array(fixed_img.shape[:2]) / self.patch_size
        self.Yscale = 1.0 * np.array(moving_img.shape[:2]) / self.patch_size
        fixed_cnn = imresize(
            fixed_img, output_size=self.patch_size, interpolation="linear"
        )
        moving_cnn = imresize(
            moving_img, output_size=self.patch_size, interpolation="linear"
        )

        fixed_cnn = fixed_cnn / 255.0
        moving_cnn = moving_cnn / 255.0

        fixed_cnn = np.moveaxis(fixed_cnn, -1, 0)
        moving_cnn = np.moveaxis(moving_cnn, -1, 0)

        fixed_cnn = np.expand_dims(fixed_cnn, axis=0)
        moving_cnn = np.expand_dims(moving_cnn, axis=0)
        cnn_input = np.concatenate((fixed_cnn, moving_cnn), axis=0)

        x = torch.from_numpy(cnn_input).type(torch.float32)
        return self.FeatureExtractor(x)
