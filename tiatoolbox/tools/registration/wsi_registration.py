import numpy as np
import torch
import torchvision
from skimage import exposure, filters, morphology
from torchvision.models._utils import IntermediateLayerGetter

from tiatoolbox.utils.transforms import imresize


class DFBRegistrtation:
    r"""Deep Feature based Registration

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


def match_histograms(image_a, image_b, disk_radius=3):
    """Image normalization function.

    This function performs histogram equalization to unify the
    appearance of an image pair.

    Args:
        image_a (:class:`numpy.ndarray`):
            A grayscale image.
        image_b (:class:`numpy.ndarray`):
            A grayscale image.
        disk_radius (int):
            The radius of the disk-shaped footprint.

    Returns:
        :class:`numpy.ndarray`:
            A normalized grayscale image.
        :class:`numpy.ndarray`:
            A normalized grayscale image.

    """

    image_a, image_b = np.squeeze(image_a), np.squeeze(image_b)
    if len(image_a.shape) == 3 or len(image_b.shape) == 3:
        raise ValueError("The input images should be grayscale images.")

    entropy_a, entropy_b = filters.rank.entropy(
        image_a, morphology.disk(disk_radius)
    ), filters.rank.entropy(image_b, morphology.disk(disk_radius))
    if np.mean(entropy_a) > np.mean(entropy_b):
        image_b = exposure.match_histograms(image_b, image_a).astype(np.uint8)
    else:
        image_a = exposure.match_histograms(image_a, image_b).astype(np.uint8)

    return image_a, image_b
