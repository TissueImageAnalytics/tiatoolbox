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

    @staticmethod
    def finding_match(feature_dist):
        """Computes matching points.

        This function computes all the possible matching points
        between fixed and moving images.

        Args:
            feature_dist (:class:`numpy.ndarray`):
                A feature distance array.

        Returns:
            :class:`numpy.ndarray`:
                An array of matching points.
            :class:`numpy.ndarray`:
                An array of floating numbers representing quality
                of each matching points.

        """
        seq = np.arange(feature_dist.shape[0])
        indx_first_min = np.argmin(feature_dist, axis=1)
        first_min = feature_dist[seq, indx_first_min]
        mask = np.zeros_like(feature_dist)
        mask[seq, indx_first_min] = 1
        masked = np.ma.masked_array(feature_dist, mask)
        second_min = np.amin(masked, axis=1)
        return np.array([seq, indx_first_min]).transpose(), np.array(
            second_min / first_min
        )

    @staticmethod
    def compute_feature_distance(feature_x, feature_y, factor):
        """Computes feature distance.

        This function computes Euclidean distance between features of
        fixed and moving images.

        Args:
            feature_x (:class:`numpy.ndarray`):
                Features computed for a fixed image.
            feature_y (:class:`numpy.ndarray`):
                Features computed for a moving image.
            factor (int):
                A number multiplied by the feature size
                for getting the referenced feature size.

        Returns:
            :class:`numpy.ndarray`:
                A feature distance array.

        """
        feature_distance = np.linalg.norm(
            np.repeat(np.expand_dims(feature_x, axis=0), feature_y.shape[0], axis=0)
            - np.repeat(np.expand_dims(feature_y, axis=1), feature_x.shape[0], axis=1),
            axis=len(feature_x.shape),
        )

        feature_size_2d = np.int(np.sqrt(feature_distance.shape[0]))
        ref_feature_size_2d = factor * feature_size_2d
        feature_size, ref_feature_size = feature_size_2d**2, ref_feature_size_2d**2
        feature_grid = np.kron(
            np.arange(feature_size).reshape([feature_size_2d, feature_size_2d]),
            np.ones([factor, factor], dtype="int32"),
        )
        row_indx = np.repeat(
            feature_grid.reshape([ref_feature_size, 1]), ref_feature_size, axis=1
        )
        col_indx = np.repeat(
            feature_grid.reshape([1, ref_feature_size]), ref_feature_size, axis=0
        )
        return feature_distance[row_indx, col_indx]

    def feature_mapping(self, features, num_matching_points=128):
        """CNN based feature extraction for registration.

        This function extracts multiscale features from a pre-trained
        VGG-16 model for an image pair.

        Args:
            features (dict):
                Multiscale CNN features.
            num_matching_points (int):
                Number of required matching points.

        Returns:

        """
        pool3_feat = features["block3_pool"].detach().numpy()
        pool4_feat = features["block4_pool"].detach().numpy()
        pool5_feat = features["block5_pool"].detach().numpy()
        ref_feature_size = pool3_feat.shape[2]

        fixed_feat1, moving_feat1 = np.reshape(
            pool3_feat[0, :, :, :], [-1, 256]
        ), np.reshape(pool3_feat[1, :, :, :], [-1, 256])
        fixed_feat2, moving_feat2 = np.reshape(
            pool4_feat[0, :, :, :], [-1, 512]
        ), np.reshape(pool4_feat[1, :, :, :], [-1, 512])
        fixed_feat3, moving_feat3 = np.reshape(
            pool5_feat[0, :, :, :], [-1, 512]
        ), np.reshape(pool5_feat[1, :, :, :], [-1, 512])

        fixed_feat1, moving_feat1 = fixed_feat1 / np.std(
            fixed_feat1
        ), moving_feat1 / np.std(moving_feat1)
        fixed_feat2, moving_feat2 = fixed_feat2 / np.std(
            fixed_feat2
        ), moving_feat2 / np.std(moving_feat2)
        fixed_feat3, moving_feat3 = fixed_feat3 / np.std(
            fixed_feat3
        ), moving_feat3 / np.std(moving_feat3)

        feature_dist1 = self.compute_feature_distance(fixed_feat1, moving_feat1, 1)
        feature_dist2 = self.compute_feature_distance(fixed_feat2, moving_feat2, 2)
        feature_dist3 = self.compute_feature_distance(fixed_feat3, moving_feat3, 4)
        feature_dist = 1.414 * feature_dist1 + feature_dist2 + feature_dist3

        del (
            fixed_feat1,
            moving_feat1,
            fixed_feat2,
            moving_feat2,
            fixed_feat3,
            moving_feat3,
            feature_dist1,
            feature_dist2,
            feature_dist3,
        )

        seq = np.array(
            [[i, j] for i in range(ref_feature_size) for j in range(ref_feature_size)],
            dtype="int32",
        )
        fixed_points = np.array(seq, dtype="float32") * 8.0 + 4.0
        moving_points = np.array(seq, dtype="float32") * 8.0 + 4.0

        fixed_points = (fixed_points - 112.0) / 224.0
        moving_points = (moving_points - 112.0) / 224.0

        matching_points, quality = self.finding_match(feature_dist)
        max_quality = np.max(quality)
        while np.where(quality >= max_quality)[0].shape[0] <= num_matching_points:
            max_quality -= 0.01

        matching_points = matching_points[np.where(quality >= max_quality)]
        count_matching_points = matching_points.shape[0]

        fixed_points, moving_points = (
            fixed_points[matching_points[:, 1]],
            moving_points[matching_points[:, 0]],
        )
        feature_dist = feature_dist[
            np.repeat(
                np.reshape(matching_points[:, 1], [count_matching_points, 1]),
                count_matching_points,
                axis=1,
            ),
            np.repeat(
                np.reshape(matching_points[:, 0], [1, count_matching_points]),
                count_matching_points,
                axis=0,
            ),
        ]

        fixed_points, moving_points = ((fixed_points * 224.0) + 112.0) * self.Xscale, (
            (moving_points * 224.0) + 112.0
        ) * self.Yscale
        return fixed_points, moving_points, np.amin(feature_dist, axis=1)


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
