import warnings
from collections import OrderedDict
from typing import Tuple

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
import torchvision
from skimage import exposure, filters
from skimage.util import img_as_float

from tiatoolbox.utils.metrics import dice
from tiatoolbox.utils.transforms import imresize


def _check_dims(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> None:
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
    if len(np.unique(fixed_mask)) == 1 or len(np.unique(moving_mask)) == 1:
        raise ValueError("The foreground is missing in the mask.")

    if len(fixed_img.shape) != 2 or len(moving_img.shape) != 2:
        raise ValueError("The input images should be grayscale images.")

    if (
        fixed_img.shape[:] != fixed_mask.shape[:]
        or moving_img.shape[:] != moving_mask.shape[:]
    ):
        raise ValueError("Mismatch of shape between image and its corresponding mask.")


def prealignment(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    dice_overlap: float = 0.5,
    rotation_step: int = 10,
) -> np.ndarray:
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


def match_histograms(
    image_a: np.ndarray, image_b: np.ndarray, kernel_size: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
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
        tuple:
            A normalized pair of images for performing registration.
            - np.ndarray - A normalized grayscale image.
            - np.ndarray - A normalized grayscale image.

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


class DFBRFeatureExtractor(torch.nn.Module):
    """Feature extractor for Deep Feature based Registration (DFBR).

    This class extracts features from three different layers of VGG16.
    These features are processed in DFBRegister class for registration
    of a pair of images.

    """

    def __init__(self):
        super().__init__()
        output_layers_id: list[str] = ["16", "23", "30"]
        output_layers_key: list[str] = ["block3_pool", "block4_pool", "block5_pool"]
        self.features: OrderedDict = OrderedDict.fromkeys(output_layers_key, None)
        self.pretrained: torch.nn.Sequential = torchvision.models.vgg16(
            pretrained=True
        ).features
        self.f_hooks = []

        for i, l in enumerate(output_layers_id):
            self.f_hooks.append(
                getattr(self.pretrained, l).register_forward_hook(
                    self.forward_hook(output_layers_key[i])
                )
            )

    def forward_hook(self, layer_name: str) -> None:
        """Register a hook.

        Args:
            layer_name (str):
                User-defined name for a layer.

        Returns:
            None

        """

        def hook(
            _module: torch.nn.MaxPool2d,
            _module_input: tuple[torch.Tensor],
            module_output: torch.Tensor,
        ) -> None:
            """Forward hook for feature extraction.

            Args:
                _module:
                    Unused argument for the module.
                _module_input:
                    Unused argument for the modules' input.
                module_output (torch.Tensor):
                    Output (features) of the module.

            Returns:
                None

            """
            self.features[layer_name] = module_output

        return hook

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Forward pass for feature extraction.

        Args:
            x (torch.Tensor):
                Batch of input images.

        Returns:
            OrderedDict:
                A dictionary containing the multiscale features.
                The expected format is {layer_name: features}.

        """
        _ = self.pretrained(x)
        return self.features


class DFBRegister:
    r"""Deep Feature based Registration (DFBR).

    This class implements a CNN feature based method for
    registering a pair of histology images, as presented
    in the paper [1]. This work is adapted from [2].

    References:
        [1] Awan, R., Raza, S.E.A., Lotz, J. and Rajpoot, N.M., 2022.
        `Deep Feature based Cross-slide Registration
        <https://arxiv.org/pdf/2202.09971.pdf>`_. arXiv preprint
        arXiv:2202.09971.

        [2] Yang, Z., Dan, T. and Yang, Y., 2018. Multi-temporal remote
        sensing image registration using deep convolutional features.
        Ieee Access, 6, pp.38544-38555.

    """

    def __init__(self):
        self.patch_size: Tuple[int, int] = (224, 224)
        self.x_scale, self.y_scale = [], []
        self.feature_extractor = DFBRFeatureExtractor()

    # Make this function private when full pipeline is implemented.
    def extract_features(
        self, fixed_img: np.ndarray, moving_img: np.ndarray
    ) -> OrderedDict[str, torch.Tensor]:
        """CNN based feature extraction for registration.

        This function extracts multiscale features from a pre-trained
        VGG-16 model for an image pair.

        Args:
            fixed_img (:class:`numpy.ndarray`):
                A fixed image.
            moving_img (:class:`numpy.ndarray`):
                A moving image.

        Returns:
            OrderedDict:
                A dictionary containing the multiscale features.
                The expected format is {layer_name: features}.

        """
        if len(fixed_img.shape) != 3 or len(moving_img.shape) != 3:
            raise ValueError(
                "The required shape for fixed and moving images is n x m x 3."
            )

        if fixed_img.shape[2] != 3 or moving_img.shape[2] != 3:
            raise ValueError("The input images are expected to have 3 channels.")

        self.x_scale = 1.0 * np.array(fixed_img.shape[:2]) / self.patch_size
        self.y_scale = 1.0 * np.array(moving_img.shape[:2]) / self.patch_size
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
        return self.feature_extractor(x)

    @staticmethod
    def finding_match(feature_dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes matching points.

        This function computes all the possible matching points
        between fixed and moving images.

        Args:
            feature_dist (:class:`numpy.ndarray`):
                A feature distance array.

        Returns:
            tuple:
                - np.ndarray - An array of matching points.
                - np.ndarray - An array of floating numbers representing
                               quality of each matching points.

        """
        seq = np.arange(feature_dist.shape[0])
        ind_first_min = np.argmin(feature_dist, axis=1)
        first_min = feature_dist[seq, ind_first_min]
        mask = np.zeros_like(feature_dist)
        mask[seq, ind_first_min] = 1
        masked = np.ma.masked_array(feature_dist, mask)
        second_min = np.amin(masked, axis=1)
        return np.array([seq, ind_first_min]).transpose(), np.array(
            second_min / first_min
        )

    @staticmethod
    def compute_feature_distances(
        features_x: np.ndarray, features_y: np.ndarray, factor: int
    ) -> np.ndarray:
        """Compute feature distance.

        This function computes Euclidean distance between features of
        fixed and moving images.

        Args:
            features_x (:class:`numpy.ndarray`):
                Features computed for a fixed image.
            features_y (:class:`numpy.ndarray`):
                Features computed for a moving image.
            factor (int):
                A number multiplied by the feature size
                for getting the referenced feature size.

        Returns:
            :class:`numpy.ndarray`:
                A feature distance array.

        """
        feature_distance = np.linalg.norm(
            np.repeat(np.expand_dims(features_x, axis=0), features_y.shape[0], axis=0)
            - np.repeat(
                np.expand_dims(features_y, axis=1), features_x.shape[0], axis=1
            ),
            axis=len(features_x.shape),
        )

        feature_size_2d = np.int(np.sqrt(feature_distance.shape[0]))
        ref_feature_size_2d = factor * feature_size_2d
        feature_size, ref_feature_size = feature_size_2d**2, ref_feature_size_2d**2
        feature_grid = np.kron(
            np.arange(feature_size).reshape([feature_size_2d, feature_size_2d]),
            np.ones([factor, factor], dtype="int32"),
        )
        row_ind = np.repeat(
            feature_grid.reshape([ref_feature_size, 1]), ref_feature_size, axis=1
        )
        col_ind = np.repeat(
            feature_grid.reshape([1, ref_feature_size]), ref_feature_size, axis=0
        )
        return feature_distance[row_ind, col_ind]

    def feature_mapping(
        self, features: OrderedDict[str, torch.Tensor], num_matching_points: int = 128
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find mapping between CNN features.

        This function maps features of a fixed image to that of
        a moving image on the basis of Euclidean distance between
        them.

        Args:
            features (OrderedDict):
                Multiscale CNN features.
            num_matching_points (int):
                Number of required matching points.

        Returns:
            tuple:
                Parameters for estimating transformation parameters.
                - np.ndarray - A matching 2D point set in the fixed image.
                - np.ndarray - A matching 2D point set in the moving image.
                - np.ndarray - A 1D array, where each element represents
                               quality of each matching point.

        """
        if len(features) != 3:
            raise ValueError("The feature mapping step expects 3 blocks of features.")

        pool3_feat = features["block3_pool"].detach().numpy()
        pool4_feat = features["block4_pool"].detach().numpy()
        pool5_feat = features["block5_pool"].detach().numpy()
        ref_feature_size = pool3_feat.shape[2]

        fixed_feat1 = np.reshape(pool3_feat[0, :, :, :], [-1, 256])
        moving_feat1 = np.reshape(pool3_feat[1, :, :, :], [-1, 256])
        fixed_feat2 = np.reshape(pool4_feat[0, :, :, :], [-1, 512])
        moving_feat2 = np.reshape(pool4_feat[1, :, :, :], [-1, 512])
        fixed_feat3 = np.reshape(pool5_feat[0, :, :, :], [-1, 512])
        moving_feat3 = np.reshape(pool5_feat[1, :, :, :], [-1, 512])

        fixed_feat1 = fixed_feat1 / np.std(fixed_feat1)
        moving_feat1 = moving_feat1 / np.std(moving_feat1)
        fixed_feat2 = fixed_feat2 / np.std(fixed_feat2)
        moving_feat2 = moving_feat2 / np.std(moving_feat2)
        fixed_feat3 = fixed_feat3 / np.std(fixed_feat3)
        moving_feat3 = moving_feat3 / np.std(moving_feat3)

        feature_dist1 = self.compute_feature_distances(fixed_feat1, moving_feat1, 1)
        feature_dist2 = self.compute_feature_distances(fixed_feat2, moving_feat2, 2)
        feature_dist3 = self.compute_feature_distances(fixed_feat3, moving_feat3, 4)
        feature_dist = np.sqrt(2) * feature_dist1 + feature_dist2 + feature_dist3

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

        fixed_points = ((fixed_points * 224.0) + 112.0) * self.x_scale
        moving_points = ((moving_points * 224.0) + 112.0) * self.y_scale
        return fixed_points, moving_points, np.amin(feature_dist, axis=1)

    @staticmethod
    def estimate_affine_transform(
        points_0: np.ndarray, points_1: np.ndarray
    ) -> np.ndarray:
        """Compute affine transformation matrix.

        This function estimates transformation parameters
        using linear least squares for a given set of matched
        points.

        Args:
            points_0 (:class:`numpy.ndarray`):
                An Nx2 array of points in a fixed image.
            points_1 (:class:`numpy.ndarray`):
                An Nx2 array of points in a moving image.

        Returns:
            :class:`numpy.ndarray`:
                A 3x3 transformation matrix.

        """
        num_points = min(len(points_0), len(points_1))
        x = np.hstack([points_0[:num_points], np.ones((num_points, 1))])
        y = np.hstack([points_1[:num_points], np.ones((num_points, 1))])

        matrix = np.linalg.lstsq(x, y, rcond=-1)[0].T
        matrix[-1, :] = [0, 0, 1]

        return matrix
