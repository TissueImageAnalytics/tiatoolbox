import itertools
import warnings
from numbers import Number
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
import torchvision
from numpy.linalg import inv
from skimage import exposure, filters
from skimage.registration import phase_cross_correlation
from skimage.util import img_as_float

from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils.metrics import dice
from tiatoolbox.utils.transforms import imresize
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader

Resolution = Union[Number, Tuple[Number, Number], np.ndarray]
IntBounds = Tuple[int, int, int, int]


def _check_dims(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Check the dimensionality of images and mask.

    This function verify the dimensionality of images and their corresponding masks.
    If the input images are RGB images, it converts them to grayscale images.

    Args:
        fixed_img (:class:`numpy.ndarray`):
            A fixed image.
        moving_img (:class:`numpy.ndarray`):
            A moving image.
        fixed_mask (:class:`numpy.ndarray`):
            A binary tissue mask for the fixed image.
        moving_mask (:class:`numpy.ndarray`):
            A binary tissue mask for the moving image.

    Returns:
        tuple:
            - :class:`numpy.ndarray` - A grayscale fixed image.
            - :class:`numpy.ndarray` - A grayscale moving image.

    """
    if len(np.unique(fixed_mask)) == 1 or len(np.unique(moving_mask)) == 1:
        raise ValueError("The foreground is missing in the mask.")

    if (
        fixed_img.shape[:2] != fixed_mask.shape
        or moving_img.shape[:2] != moving_mask.shape
    ):
        raise ValueError("Mismatch of shape between image and its corresponding mask.")

    if len(fixed_img.shape) == 3:
        fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_BGR2GRAY)

    if len(moving_img.shape) == 3:
        moving_img = cv2.cvtColor(moving_img, cv2.COLOR_BGR2GRAY)

    return fixed_img, moving_img


def compute_center_of_mass(mask: np.ndarray) -> tuple:
    """Compute center of mass.

    Args:
        mask: (:class:`numpy.ndarray`):
            A binary mask.

    Returns:
        :py:obj:`tuple` - x- and y- coordinates representing center of mass.
            - :py:obj:`int` - X coordinate.
            - :py:obj:`int` - Y coordinate.

    """
    moments = cv2.moments(mask)
    x_coord_center = moments["m10"] / moments["m00"]
    y_coord_center = moments["m01"] / moments["m00"]
    return (x_coord_center, y_coord_center)


def prealignment(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    dice_overlap: float = 0.5,
    rotation_step: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Coarse registration of an image pair.

    This function performs initial alignment of a moving image with respect to a
    fixed image. This can be used as a prealignment step before final refinement.

    Args:
        fixed_img (:class:`numpy.ndarray`):
            A fixed image.
        moving_img (:class:`numpy.ndarray`):
            A moving image.
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
        tuple:
            - :class:`numpy.ndarray` - A rigid transform matrix.
            - :class:`numpy.ndarray` - Transformed moving image.
            - :class:`numpy.ndarray` - Transformed moving mask.
            - :py:obj:`float` - Dice overlap.

    Examples:
        >>> from tiatoolbox.tools.registration.wsi_registration import prealignment
        >>> transform, transformed_image, transformed_mask, dice_overlap = prealignment(
        ...     fixed_thumbnail, moving_thumbnail, fixed_mask, moving_mask
        ... )

    """
    orig_fixed_img, orig_moving_img = fixed_img, moving_img

    if len(fixed_mask.shape) != 2:
        fixed_mask = fixed_mask[:, :, 0]
    if len(moving_mask.shape) != 2:
        moving_mask = moving_mask[:, :, 0]

    fixed_mask = np.uint8(fixed_mask > 0)
    moving_mask = np.uint8(moving_mask > 0)

    fixed_img = np.squeeze(fixed_img)
    moving_img = np.squeeze(moving_img)
    fixed_img, moving_img = _check_dims(fixed_img, moving_img, fixed_mask, moving_mask)

    if rotation_step < 10 or rotation_step > 20:
        raise ValueError("Please select the rotation step in between 10 and 20.")

    if dice_overlap < 0 or dice_overlap > 1:
        raise ValueError("The dice_overlap should be in between 0 and 1.0.")

    fixed_img = exposure.rescale_intensity(img_as_float(fixed_img), in_range=(0, 1))
    moving_img = exposure.rescale_intensity(img_as_float(moving_img), in_range=(0, 1))

    # Resizing of fixed and moving masks so that dice can be computed
    height = np.max((fixed_mask.shape[0], moving_mask.shape[0]))
    width = np.max((fixed_mask.shape[1], moving_mask.shape[1]))
    padded_fixed_mask = np.pad(
        fixed_mask,
        pad_width=[(0, height - fixed_mask.shape[0]), (0, width - fixed_mask.shape[1])],
        mode="constant",
    )
    padded_moving_mask = np.pad(
        moving_mask,
        pad_width=[
            (0, height - moving_mask.shape[0]),
            (0, width - moving_mask.shape[1]),
        ],
        mode="constant",
    )
    dice_before = dice(padded_fixed_mask, padded_moving_mask)

    fixed_com = compute_center_of_mass((1 - fixed_img) * fixed_mask)
    moving_com = compute_center_of_mass((1 - moving_img) * moving_mask)

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

        # Apply transformation
        warped_moving_mask = cv2.warpAffine(
            moving_mask, transform[0:-1][:], fixed_img.shape[:2][::-1]
        )
        dice_com = dice(fixed_mask, warped_moving_mask)

        all_dice.append(dice_com)
        all_transform.append(transform)

    if max(all_dice) >= dice_overlap:
        dice_after = max(all_dice)
        pre_transform = all_transform[all_dice.index(dice_after)]

        # Apply transformation to both image and mask
        moving_img = cv2.warpAffine(
            orig_moving_img, pre_transform[0:-1][:], orig_fixed_img.shape[:2][::-1]
        )
        moving_mask = cv2.warpAffine(
            moving_mask, pre_transform[0:-1][:], fixed_img.shape[:2][::-1]
        )
        return pre_transform, moving_img, moving_mask, dice_after

    warnings.warn(
        "Not able to find the best transformation for pre-alignment. "
        "Try changing the values for 'dice_overlap' and 'rotation_step'."
    )
    return np.eye(3), moving_img, moving_mask, dice_before


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

            - :class:`numpy.ndarray` - A normalized grayscale image.
            - :class:`numpy.ndarray` - A normalized grayscale image.

    Examples:
        >>> from tiatoolbox.tools.registration.wsi_registration import match_histograms
        >>> norm_image_a, norm_image_b = match_histograms(gray_image_a, gray_image_b)

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
        self.features: dict = dict.fromkeys(output_layers_key, None)
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
            _module_input: Tuple[torch.Tensor],
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for feature extraction.

        Args:
            x (torch.Tensor):
                Batch of input images.

        Returns:
            dict:
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

    Examples:
        >>> from tiatoolbox.tools.registration.wsi_registration import DFBRegister
        >>> import cv2
        >>> df = DFBRegister()
        >>> fixed_image = np.repeat(np.expand_dims(fixed_gray, axis=2), 3, axis=2)
        >>> moving_image = np.repeat(np.expand_dims(moving_gray, axis=2), 3, axis=2)
        >>> transform = df.register(fixed_image, moving_image, fixed_mask, moving_mask)
        >>> registered = cv2.warpAffine(
        ...     moving_gray, transform[0:-1], fixed_gray.shape[:2][::-1]
        ... )

    """

    def __init__(self, patch_size: Tuple[int, int] = (224, 224)):
        self.patch_size = patch_size
        self.x_scale, self.y_scale = [], []
        self.feature_extractor = DFBRFeatureExtractor()

    # Make this function private when full pipeline is implemented.
    def extract_features(
        self, fixed_img: np.ndarray, moving_img: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """CNN based feature extraction for registration.

        This function extracts multiscale features from a pre-trained
        VGG-16 model for an image pair.

        Args:
            fixed_img (:class:`numpy.ndarray`):
                A fixed image.
            moving_img (:class:`numpy.ndarray`):
                A moving image.

        Returns:
            Dict:
                A dictionary containing the multiscale features.
                The expected format is {layer_name: features}.

        """
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
                - :class:`numpy.ndarray` - An array of matching points.
                - :class:`numpy.ndarray` - An array of floating numbers representing
                  quality of each matching point.

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
        self, features: Dict[str, torch.Tensor], num_matching_points: int = 128
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find mapping between CNN features.

        This function maps features of a fixed image to that of
        a moving image on the basis of Euclidean distance between
        them.

        Args:
            features (Dict):
                Multiscale CNN features.
            num_matching_points (int):
                Number of required matching points.

        Returns:
            tuple:
                Parameters for estimating transformation parameters.

                - :class:`numpy.ndarray` - A matching 2D point set in the fixed image.
                - :class:`numpy.ndarray` - A matching 2D point set in the moving image.
                - :class:`numpy.ndarray` - A 1D array, where each element represents
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
        fixed_points, moving_points = fixed_points[:, [1, 0]], moving_points[:, [1, 0]]
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

    @staticmethod
    def get_tissue_regions(
        fixed_image: np.ndarray,
        fixed_mask: np.ndarray,
        moving_image: np.ndarray,
        moving_mask: np.ndarray,
    ) -> Tuple[np.array, np.array, np.array, np.array, IntBounds]:
        """Extract tissue region.

        This function uses binary mask for extracting tissue
        region from the image.

        Args:
            fixed_image (:class:`numpy.ndarray`):
                A fixed image.
            fixed_mask (:class:`numpy.ndarray`):
                A binary tissue mask for the fixed image.
            moving_image (:class:`numpy.ndarray`):
                A moving image.
            moving_mask (:class:`numpy.ndarray`):
                A binary tissue mask for the moving image.

        Returns:
            tuple:
                - :class:`numpy.ndarray` - A cropped image containing tissue region
                  from fixed image.
                - :class:`numpy.ndarray` - A cropped image containing tissue mask
                  from fixed image.
                - :class:`numpy.ndarray` - A cropped image containing tissue region
                  from moving image.
                - :class:`numpy.ndarray` - A cropped image containing tissue mask
                  from moving image.
                - :py:obj:`tuple` - Bounds of the tissue region.
                    - :py:obj:`int` - Top (start y value)
                    - :py:obj:`int` - Left (start x value)
                    - :py:obj:`int` - Bottom (end y value)
                    - :py:obj:`int` - Right (end x value)

        """
        fixed_minc, fixed_min_r, width, height = cv2.boundingRect(fixed_mask)
        fixed_max_c, fixed_max_r = fixed_minc + width, fixed_min_r + height
        moving_minc, moving_min_r, width, height = cv2.boundingRect(moving_mask)
        moving_max_c, moving_max_r = moving_minc + width, moving_min_r + height

        minc, max_c, min_r, max_r = (
            np.min([fixed_minc, moving_minc]),
            np.max([fixed_max_c, moving_max_c]),
            np.min([fixed_min_r, moving_min_r]),
            np.max([fixed_max_r, moving_max_r]),
        )

        fixed_tissue_image = fixed_image[min_r:max_r, minc:max_c]
        fixed_tissue_mask = fixed_mask[min_r:max_r, minc:max_c]
        moving_tissue_image = moving_image[min_r:max_r, minc:max_c]
        moving_tissue_mask = moving_mask[min_r:max_r, minc:max_c]
        moving_tissue_image[np.all(moving_tissue_image == (0, 0, 0), axis=-1)] = (
            243,
            243,
            243,
        )
        return (
            fixed_tissue_image,
            fixed_tissue_mask,
            moving_tissue_image,
            moving_tissue_mask,
            (min_r, minc, max_r, max_c),
        )

    @staticmethod
    def find_points_inside_boundary(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Find indices of points lying inside the boundary.

        This function returns indices of points which are
        enclosed by an area indicated by a binary mask.

        Args:
            mask (:class:`numpy.ndarray`):
                A binary tissue mask
            points (:class:`numpy.ndarray`):
                 (N, 2) array of point coordinates.

        Returns:
            :class:`numpy.ndarray`:
                Indices of points enclosed by a boundary.

        """
        kernel = np.ones((25, 25), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask_reader = VirtualWSIReader(mask)

        # convert coordinates of shape [N, 2] to [N, 4]
        end_x_y = points[:, 0:2] + 1
        bbox_coord = np.c_[points, end_x_y].astype(int)
        return PatchExtractor.filter_coordinates_fast(
            mask_reader, bbox_coord, 1.0, "baseline", 1.0
        )

    def filtering_matching_points(
        self,
        fixed_mask: np.ndarray,
        moving_mask: np.ndarray,
        fixed_matched_points: np.ndarray,
        moving_matched_points: np.ndarray,
        quality: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter the matching points.

        This function removes the duplicated points and the points
        which are identified outside the tissue region.

        Args:
            fixed_mask (:class:`numpy.ndarray`):
                A binary tissue mask for the fixed image.
            moving_mask (:class:`numpy.ndarray`):
                A binary tissue mask for the moving image.
            fixed_matched_points (:class:`numpy.ndarray`):
                (N, 2) array of coordinates.
            moving_matched_points (:class:`numpy.ndarray`):
                (N, 2) array of coordinates.
            quality (:class:`numpy.ndarray`):
                An array representing quality of each matching point.

        Returns:
            tuple:
                - np.ndarray - Filtered matching points for a fixed image.
                - np.ndarray - Filtered matching points for a moving image.
                - np.ndarray - Quality of matching points.

        """
        included_index = self.find_points_inside_boundary(
            fixed_mask, fixed_matched_points
        )
        fixed_matched_points, moving_matched_points, quality = (
            fixed_matched_points[included_index, :],
            moving_matched_points[included_index, :],
            quality[included_index],
        )
        included_index = self.find_points_inside_boundary(
            moving_mask, moving_matched_points
        )
        fixed_matched_points, moving_matched_points, quality = (
            fixed_matched_points[included_index, :],
            moving_matched_points[included_index, :],
            quality[included_index],
        )

        # remove duplicate matching points
        duplicate_ind = []
        unq, count = np.unique(fixed_matched_points, axis=0, return_counts=True)
        repeated_points = unq[count > 1]
        for repeated_point in repeated_points:
            repeated_idx = np.argwhere(
                np.all(fixed_matched_points == repeated_point, axis=1)
            )
            duplicate_ind = np.hstack([duplicate_ind, repeated_idx.ravel()])

        unq, count = np.unique(moving_matched_points, axis=0, return_counts=True)
        repeated_points = unq[count > 1]
        for repeated_point in repeated_points:
            repeated_idx = np.argwhere(
                np.all(moving_matched_points == repeated_point, axis=1)
            )
            duplicate_ind = np.hstack([duplicate_ind, repeated_idx.ravel()])

        if len(duplicate_ind) > 0:
            duplicate_ind = duplicate_ind.astype(int)
            fixed_matched_points = np.delete(
                fixed_matched_points, duplicate_ind, axis=0
            )
            moving_matched_points = np.delete(
                moving_matched_points, duplicate_ind, axis=0
            )
            quality = np.delete(quality, duplicate_ind)

        return fixed_matched_points, moving_matched_points, quality

    def perform_dfbregister(
        self,
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
        fixed_mask: np.ndarray,
        moving_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform DFBR to align a pair of image.

        This function aligns a pair of images using Deep
        Feature based Registration (DFBR) method.

        Args:
            fixed_img (:class:`numpy.ndarray`):
                A fixed image.
            moving_img (:class:`numpy.ndarray`):
                A moving image.
            fixed_mask (:class:`numpy.ndarray`):
                A binary tissue mask for the fixed image.
            moving_mask (:class:`numpy.ndarray`):
                A binary tissue mask for the moving image.

        Returns:
            tuple:
                - :class:`numpy.ndarray` - An affine transformation matrix.
                - :class:`numpy.ndarray` - A transformed moving image.
                - :class:`numpy.ndarray` - A transformed moving mask.

        """
        features = self.extract_features(fixed_img, moving_img)
        fixed_matched_points, moving_matched_points, quality = self.feature_mapping(
            features
        )

        (
            fixed_matched_points,
            moving_matched_points,
            quality,
        ) = self.filtering_matching_points(
            fixed_mask,
            moving_mask,
            fixed_matched_points,
            moving_matched_points,
            quality,
        )

        tissue_transform = DFBRegister.estimate_affine_transform(
            fixed_matched_points, moving_matched_points
        )

        # Apply transformation
        moving_img = cv2.warpAffine(
            moving_img, tissue_transform[0:-1][:], fixed_img.shape[:2][::-1]
        )
        moving_mask = cv2.warpAffine(
            moving_mask, tissue_transform[0:-1][:], fixed_img.shape[:2][::-1]
        )
        return tissue_transform, moving_img, moving_mask

    def perform_dfbregister_block_wise(
        self,
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
        fixed_mask: np.ndarray,
        moving_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform DFBR to align a pair of images in a block wise manner.

        This function divides the images into four equal parts and then
        perform feature matching for each part from the tissue and moving
        images. Matching features of all the parts are then concatenated
        for the estimation of affine transform.

        Args:
            fixed_img (:class:`numpy.ndarray`):
                A fixed image.
            moving_img (:class:`numpy.ndarray`):
                A moving image.
            fixed_mask (:class:`numpy.ndarray`):
                A binary tissue mask for the fixed image.
            moving_mask (:class:`numpy.ndarray`):
                A binary tissue mask for the moving image.

        Returns:
            tuple:
                - :class:`numpy.ndarray` - An affine transformation matrix.
                - :class:`numpy.ndarray` - A transformed moving image.
                - :class:`numpy.ndarray` - A transformed moving mask.

        """
        left_upper_bounding_bbox = [
            0,
            int(np.floor(fixed_img.shape[0] / 2)),
            0,
            int(np.floor(fixed_img.shape[1] / 2)),
        ]
        right_upper_bounding_bbox = [
            0,
            int(np.floor(fixed_img.shape[0] / 2)),
            int(np.ceil(fixed_img.shape[1] / 2)),
            fixed_img.shape[1],
        ]
        left_lower_bounding_bbox = [
            int(np.ceil(fixed_img.shape[0] / 2)),
            fixed_img.shape[0],
            0,
            int(np.floor(fixed_img.shape[1] / 2)),
        ]
        right_lower_bounding_bbox = [
            int(np.ceil(fixed_img.shape[0] / 2)),
            fixed_img.shape[0],
            int(np.ceil(fixed_img.shape[1] / 2)),
            fixed_img.shape[1],
        ]
        blocks_bounding_box = [
            left_upper_bounding_bbox,
            right_upper_bounding_bbox,
            left_lower_bounding_bbox,
            right_lower_bounding_bbox,
        ]

        fixed_matched_points, moving_matched_points, quality = [], [], []
        for _index, bounding_box in enumerate(blocks_bounding_box):
            fixed_block = fixed_img[
                bounding_box[0] : bounding_box[1], bounding_box[2] : bounding_box[3], :
            ]
            moving_block = moving_img[
                bounding_box[0] : bounding_box[1], bounding_box[2] : bounding_box[3], :
            ]
            features = self.extract_features(fixed_block, moving_block)
            (
                fixed_block_matched_points,
                moving_block_matched_points,
                block_quality,
            ) = self.feature_mapping(features)
            fixed_matched_points.append(
                fixed_block_matched_points + [bounding_box[2], bounding_box[0]]
            )
            moving_matched_points.append(
                moving_block_matched_points + [bounding_box[2], bounding_box[0]]
            )
            quality.append(block_quality)
        fixed_matched_points, moving_matched_points, quality = (
            np.concatenate(fixed_matched_points),
            np.concatenate(moving_matched_points),
            np.concatenate(quality),
        )
        (
            fixed_matched_points,
            moving_matched_points,
            _,
        ) = self.filtering_matching_points(
            fixed_mask,
            moving_mask,
            fixed_matched_points,
            moving_matched_points,
            quality,
        )

        block_transform = DFBRegister.estimate_affine_transform(
            fixed_matched_points, moving_matched_points
        )

        # Apply transformation
        moving_img = cv2.warpAffine(
            moving_img, block_transform[0:-1][:], fixed_img.shape[:2][::-1]
        )
        moving_mask = cv2.warpAffine(
            moving_mask, block_transform[0:-1][:], fixed_img.shape[:2][::-1]
        )

        return block_transform, moving_img, moving_mask

    def register(
        self,
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
        fixed_mask: np.ndarray,
        moving_mask: np.ndarray,
        transform_initializer: np.ndarray = None,
    ) -> np.ndarray:
        """Perform whole slide image registration.

        This function aligns a pair of images using Deep
        Feature based Registration (DFBR) method.

        Args:
            fixed_img (:class:`numpy.ndarray`):
                A fixed image.
            moving_img (:class:`numpy.ndarray`):
                A moving image.
            fixed_mask (:class:`numpy.ndarray`):
                A binary tissue mask for the fixed image.
            moving_mask (:class:`numpy.ndarray`):
                A binary tissue mask for the moving image.
            transform_initializer (:class:`numpy.ndarray`):
                A rigid transformation matrix.

        Returns:
            :class:`numpy.ndarray`:
                An affine transformation matrix.

        """
        if len(fixed_img.shape) != 3 or len(moving_img.shape) != 3:
            raise ValueError(
                "The required shape for fixed and moving images is n x m x 3."
            )

        if fixed_img.shape[2] != 3 or moving_img.shape[2] != 3:
            raise ValueError("The input images are expected to have 3 channels.")

        if len(fixed_mask.shape) > 2:
            fixed_mask = fixed_mask[:, :, 0]
        if len(moving_mask.shape) > 2:
            moving_mask = moving_mask[:, :, 0]

        fixed_mask = np.uint8(fixed_mask > 0)
        moving_mask = np.uint8(moving_mask > 0)

        # Perform Pre-alignment
        if transform_initializer is None:
            transform_initializer, moving_img, moving_mask, before_dice = prealignment(
                fixed_img, moving_img, fixed_mask, moving_mask
            )
        else:
            # Apply transformation to both image and mask
            moving_img = cv2.warpAffine(
                moving_img, transform_initializer[0:-1][:], fixed_img.shape[:2][::-1]
            )
            moving_mask = cv2.warpAffine(
                moving_mask, transform_initializer[0:-1][:], fixed_img.shape[:2][::-1]
            )
            before_dice = dice(fixed_mask, moving_mask)

        # Estimate transform using tissue regions
        (
            fixed_tissue_img,
            fixed_tissue_mask,
            moving_tissue_img,
            moving_tissue_mask,
            tissue_top_left_coord,
        ) = self.get_tissue_regions(fixed_img, fixed_mask, moving_img, moving_mask)
        (
            tissue_transform,
            transform_tissue_img,
            transform_tissue_mask,
        ) = self.perform_dfbregister(
            fixed_tissue_img, moving_tissue_img, fixed_tissue_mask, moving_tissue_mask
        )

        # Use the estimated transform only if it improves DICE overlap
        after_dice = dice(fixed_tissue_mask, transform_tissue_mask)
        if after_dice > before_dice:
            moving_tissue_img, moving_tissue_mask = (
                transform_tissue_img,
                transform_tissue_mask,
            )
            before_dice = after_dice
        else:
            tissue_transform = np.eye(3, 3)

        # Perform transform using tissue regions in a block-wise manner
        (
            block_transform,
            transform_tissue_img,
            transform_tissue_mask,
        ) = self.perform_dfbregister_block_wise(
            fixed_tissue_img, moving_tissue_img, fixed_tissue_mask, moving_tissue_mask
        )

        # Use the estimated tissue transform only if it improves DICE overlap
        after_dice = dice(fixed_tissue_mask, transform_tissue_mask)
        if after_dice > before_dice:
            moving_tissue_img, moving_tissue_mask = (
                transform_tissue_img,
                transform_tissue_mask,
            )
            before_dice = after_dice
        else:
            block_transform = np.eye(3, 3)

        # Fix translation offset
        shift, _error, _diff_phase = phase_cross_correlation(
            fixed_tissue_img, moving_tissue_img
        )
        translation_offset = np.array([[1, 0, shift[1]], [0, 1, shift[0]], [0, 0, 1]])

        # Combining tissue and block transform
        tissue_transform = translation_offset @ block_transform @ tissue_transform

        # tissue_transform is computed for cropped images (tissue region only).
        # It is converted using the tissue crop coordinates, so that it can be
        # applied to the full image.
        forward_translation = np.array(
            [
                [1, 0, -tissue_top_left_coord[1]],
                [0, 1, -tissue_top_left_coord[0]],
                [0, 0, 1],
            ]
        )
        inverse_translation = np.array(
            [
                [1, 0, tissue_top_left_coord[1]],
                [0, 1, tissue_top_left_coord[0]],
                [0, 0, 1],
            ]
        )
        image_transform = inverse_translation @ tissue_transform @ forward_translation

        return image_transform @ transform_initializer


def estimate_bspline_transform(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    **kwargs,
) -> sitk.BSplineTransform:
    """Estimate B-spline transformation.

    This function performs registration using the `SimpleITK toolkit
    <https://simpleitk.readthedocs.io/_/downloads/en/v1.2.4/pdf/>`_. We employed
     a deformable registration using a multi-resolution B-spline approach. B-spline
     registration uses B-spline curves to compute the deformation field mapping pixels
     in a moving image to corresponding pixels in a fixed image.

    Args:
        fixed_image (:class:`numpy.ndarray`):
            A fixed image.
        moving_image (:class:`numpy.ndarray`):
            A moving image.
        fixed_mask (:class:`numpy.ndarray`):
            A binary tissue mask for the fixed image.
        moving_mask (:class:`numpy.ndarray`):
            A binary tissue mask for the moving image.
        **kwargs (dict):
            Key-word arguments for B-spline parameters.
                grid_space (float):
                    Grid_space (mm) to decide control points.
                scale_factors (list):
                    Scaling factor of each B-spline per level in a multi-level setting.
                shrink_factor (list):
                    Shrink factor per level to change the size and
                    complexity of the image.
                smooth_sigmas (list):
                    Standard deviation for gaussian smoothing per level.
                num_iterations (int):
                    Maximal number of iterations.
                sampling_percent (float):
                    Fraction of image used for metric evaluation.
                learning_rate (float):
                    Step size along traversal direction in parameter space.
                convergence_min_value (float):
                    Value for checking convergence together with energy
                    profile of the similarity metric.
                convergence_window_size (int):
                    Number of similarity metric values for estimating the
                    energy profile.

    Returns:
        2D deformation transformation represented by a grid of control points.

    Examples:
        >>> from tiatoolbox.tools.registration.wsi_registration import (
        ...     estimate_bspline_transform, apply_bspline_transform
        ... )
        >>> bspline_transform = estimate_bspline_transform(
        ...     fixed_gray_thumbnail, moving_gray_thumbnail, fixed_mask, moving_mask,
        ...     grid_space=50.0, sampling_percent=0.1,
        ... )
        >>> bspline_registered_image = apply_bspline_transform(
        ...     fixed_thumbnail, moving_thumbnail, bspline_transform
        ... )

    """
    bspline_params = {
        "grid_space": 50.0,
        "scale_factors": [1, 2, 5],
        "shrink_factor": [4, 2, 1],
        "smooth_sigmas": [4, 2, 1],
        "num_iterations": 100,
        "sampling_percent": 0.2,
        "learning_rate": 0.5,
        "convergence_min_value": 1e-4,
        "convergence_window_size": 5,
    }
    bspline_params.update(kwargs)

    fixed_image, moving_image = np.squeeze(fixed_image), np.squeeze(moving_image)
    if len(fixed_image.shape) > 3 or len(moving_image.shape) > 3:
        raise ValueError("The input images can only be grayscale or RGB images.")

    if (len(fixed_image.shape) == 3 and fixed_image.shape[2] != 3) or (
        len(moving_image.shape) == 3 and moving_image.shape[2] != 3
    ):
        raise ValueError("The input images can only have 3 channels.")

    # Inverting intensity values
    fixed_image_inv = np.invert(fixed_image)
    moving_image_inv = np.invert(moving_image)

    if len(fixed_mask.shape) > 2:
        fixed_mask = fixed_mask[:, :, 0]
    if len(moving_mask.shape) > 2:
        moving_mask = moving_mask[:, :, 0]
    fixed_mask = np.array(fixed_mask != 0, dtype=np.uint8)
    moving_mask = np.array(moving_mask != 0, dtype=np.uint8)

    # Background Removal
    fixed_image_inv = cv2.bitwise_and(fixed_image_inv, fixed_image_inv, mask=fixed_mask)
    moving_image_inv = cv2.bitwise_and(
        moving_image_inv, moving_image_inv, mask=moving_mask
    )

    # Getting SimpleITK Images from numpy arrays
    fixed_image_inv_sitk = sitk.GetImageFromArray(fixed_image_inv, isVector=True)
    moving_image_inv_sitk = sitk.GetImageFromArray(moving_image_inv, isVector=True)

    cast_filter = sitk.VectorIndexSelectionCastImageFilter()
    cast_filter.SetOutputPixelType(sitk.sitkFloat32)
    fixed_image_inv_sitk = cast_filter.Execute(fixed_image_inv_sitk)
    moving_image_inv_sitk = cast_filter.Execute(moving_image_inv_sitk)

    # Determine the number of B-spline control points using physical spacing
    grid_physical_spacing = 2 * [
        bspline_params["grid_space"]
    ]  # A control point every grid_space (mm)
    image_physical_size = [
        size * spacing
        for size, spacing in zip(
            fixed_image_inv_sitk.GetSize(), fixed_image_inv_sitk.GetSpacing()
        )
    ]
    mesh_size = [
        int(image_size / grid_spacing + 0.5)
        for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)
    ]
    mesh_size = [int(sz / 4 + 0.5) for sz in mesh_size]

    tx = sitk.BSplineTransformInitializer(
        image1=fixed_image_inv_sitk, transformDomainMeshSize=mesh_size
    )
    print("Initial Number of B-spline Parameters:", tx.GetNumberOfParameters)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetInitialTransformAsBSpline(
        tx, inPlace=True, scaleFactors=bspline_params["scale_factors"]
    )
    registration_method.SetMetricAsMattesMutualInformation(50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(
        bspline_params["sampling_percent"], sitk.sitkWallClock
    )

    registration_method.SetShrinkFactorsPerLevel(bspline_params["shrink_factor"])
    registration_method.SetSmoothingSigmasPerLevel(bspline_params["smooth_sigmas"])
    registration_method.SetOptimizerAsGradientDescentLineSearch(
        learningRate=bspline_params["learning_rate"],
        numberOfIterations=bspline_params["num_iterations"],
        convergenceMinimumValue=bspline_params["convergence_min_value"],
        convergenceWindowSize=bspline_params["convergence_window_size"],
    )
    registration_method.SetInterpolator(sitk.sitkLinear)
    return registration_method.Execute(fixed_image_inv_sitk, moving_image_inv_sitk)


def apply_bspline_transform(
    fixed_image: np.ndarray, moving_image: np.ndarray, transform: sitk.BSplineTransform
) -> np.ndarray:
    """Apply the given B-spline transform to a moving image.

    Args:
        fixed_image (:class:`numpy.ndarray`):
            A fixed image.
        moving_image (:class:`numpy.ndarray`):
            A moving image.
        transform (sitk.BSplineTransform):
            A B-spline transform.

    Returns:
        :class:`numpy.ndarray`:
            A transformed moving image.

    """
    fixed_image_sitk = sitk.GetImageFromArray(fixed_image, isVector=True)
    moving_image_sitk = sitk.GetImageFromArray(moving_image, isVector=True)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(transform)

    sitk_registered_image_sitk = resampler.Execute(moving_image_sitk)
    return sitk.GetArrayFromImage(sitk_registered_image_sitk)


class AffineWSITransformer:
    """Resampling regions from a whole slide image.

    This class is used to resample tiles/patches from a whole slide image
    using transformation.

    Example:
        >>> from tiatoolbox.tools.registration.wsi_registration import (
        ... AffineWSITransformer
        ... )
        >>> from tiatoolbox.wsicore.wsireader import WSIReader
        >>> wsi_reader = WSIReader.open(input_img=sample_ome_tiff)
        >>> transform_level0 = np.eye(3)
        >>> tfm = AffineWSITransformer(wsi_reader, transform_level0)
        >>> output = tfm.read_rect(location, size, resolution=resolution, units="level")

    """

    def __init__(self, reader: WSIReader, transform: np.ndarray) -> None:
        """Initialize object.

        Args:
            reader (WSIReader):
                An object with base WSIReader as base class.
            transform (:class:`numpy.ndarray`):
                A 3x3 transformation matrix. The inverse transformation will be applied.

        """
        self.wsi_reader = reader
        self.transform_level0 = transform

    @staticmethod
    def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Transform points using the given transformation matrix.

        Args:
            points (:class:`numpy.ndarray`):
                A set of points of shape (N, 2).
            transform (:class:`numpy.ndarray`):
                Transformation matrix of shape (3, 3).

        Returns:
            :class:`numpy.ndarray`:
                Warped points  of shape (N, 2).

        """
        points = np.array(points)
        # Pad the data with ones, so that our transformation can do translations
        points_pad = np.hstack([points, np.ones((points.shape[0], 1))])
        points_warp = np.dot(points_pad, transform.T)
        return points_warp[:, :-1]

    def get_patch_dimensions(
        self, size: Tuple[int, int], transform: np.ndarray
    ) -> Tuple[int, int]:
        """Compute patch size needed for transformation.

        Args:
            size (tuple(int)):
                (width, height) tuple giving the desired output image size.
            transform (:class:`numpy.ndarray`):
                Transformation matrix of shape (3, 3).

        Returns:
            :py:obj:`tuple` - Maximum size of the patch needed for transformation.
                - :py:obj:`int` - Width
                - :py:obj:`int` - Height

        """
        width, height = size[0], size[1]

        x = [
            np.linspace(1, width, width, endpoint=True),
            np.ones(height) * width,
            np.linspace(1, width, width, endpoint=True),
            np.ones(height),
        ]
        x = np.array(list(itertools.chain.from_iterable(x)))

        y = [
            np.ones(width),
            np.linspace(1, height, height, endpoint=True),
            np.ones(width) * height,
            np.linspace(1, height, height, endpoint=True),
        ]
        y = np.array(list(itertools.chain.from_iterable(y)))

        points = np.array([x, y]).transpose()
        transform_points = self.transform_points(points, transform)

        width = np.max(transform_points[:, 0]) - np.min(transform_points[:, 0]) + 1
        height = np.max(transform_points[:, 1]) - np.min(transform_points[:, 1]) + 1
        width, height = np.ceil(width).astype(int), np.ceil(height).astype(int)

        return (width, height)

    def get_transformed_location(
        self, location: Tuple[int, int], size: Tuple[int, int], level: int
    ) -> Tuple[int, int]:
        """Get corresponding location on unregistered image and the required patch size.

        This function applies inverse transformation to the centre point of the region.
        The transformed centre point is used to obtain the transformed top left pixel
        of the region.

        Args:
            location (tuple(int)):
                (x, y) tuple giving the top left pixel in the baseline (level 0)
                reference frame.
            size (tuple(int)):
                (width, height) tuple giving the desired output image size.
            level (int):
                Pyramid level/resolution layer.

        Returns:
            tuple:
                - :py:obj:`tuple` - Transformed location (top left pixel).
                    - :py:obj:`int` - X coordinate
                    - :py:obj:`int` - Y coordinate
                - :py:obj:`tuple` - Maximum size suitable for transformation.
                    - :py:obj:`int` - Width
                    - :py:obj:`int` - Height

        """
        inv_transform = inv(self.transform_level0)
        size_level0 = [x * (2**level) for x in size]
        center_level0 = [x + size_level0[i] / 2 for i, x in enumerate(location)]
        center_level0 = np.expand_dims(np.array(center_level0), axis=0)
        center_level0 = self.transform_points(center_level0, inv_transform)[0]

        transformed_size = self.get_patch_dimensions(size, inv_transform)
        transformed_location = [
            center_level0[0] - (transformed_size[0] * (2**level)) / 2,
            center_level0[1] - (transformed_size[1] * (2**level)) / 2,
        ]
        transformed_location = tuple(
            np.round(x).astype(int) for x in transformed_location
        )
        return transformed_location, transformed_size

    def transform_patch(self, patch: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Apply transformation to the given patch.

        This function applies the transformation matrix after removing the translation.

        Args:
            patch (:class:`numpy.ndarray`):
                A region of whole slide image.
            size (tuple(int)):
                (width, height) tuple giving the desired output image size.

        Returns:
            :class:`numpy.ndarray`:
                A transformed region/patch.

        """
        transform = self.transform_level0 * [[1, 1, 0], [1, 1, 0], [1, 1, 1]]
        translation = (-size[0] / 2 + 0.5, -size[1] / 2 + 0.5)
        forward_translation = np.array(
            [[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]]
        )
        inverse_translation = np.linalg.inv(forward_translation)
        transform = inverse_translation @ transform @ forward_translation
        return cv2.warpAffine(patch, transform[0:-1][:], patch.shape[:2][::-1])

    def read_rect(
        self,
        location: Tuple[int, int],
        size: Tuple[int, int],
        resolution: Resolution,
        units: str,
    ) -> np.ndarray:
        """Read a transformed region of the transformed whole slide image.

        Location is in terms of the baseline image (level 0 / maximum resolution),
        and size is the output image size.

        Args:
            location (tuple(int)):
                (x, y) tuple giving the top left pixel in the baseline (level 0)
                reference frame.
            size (tuple(int)):
                (width, height) tuple giving the desired output image size.
            resolution (float or tuple(float)):
                Pyramid level/resolution layer.
            units (str):
                Units of the scale.

        Returns:
            :class:`numpy.ndarray`:
                A transformed region/patch.

        """
        (
            read_level,
            _,
            _,
            _post_read_scale,
            _baseline_read_size,
        ) = self.wsi_reader.find_read_rect_params(
            location=location,
            size=size,
            resolution=resolution,
            units=units,
        )
        transformed_location, max_size = self.get_transformed_location(
            location, size, read_level
        )
        patch = self.wsi_reader.read_rect(
            transformed_location, max_size, resolution=resolution, units=units
        )
        transformed_patch = self.transform_patch(patch, max_size)

        start_row = int(max_size[1] / 2) - int(size[1] / 2)
        end_row = int(max_size[1] / 2) + int(size[1] / 2)
        start_col = int(max_size[0] / 2) - int(size[0] / 2)
        end_col = int(max_size[0] / 2) + int(size[0] / 2)
        return transformed_patch[start_row:end_row, start_col:end_col, :]
