import warnings
from typing import Dict, Tuple

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
import torchvision
from matplotlib import path
from skimage import exposure, filters
from skimage.measure import regionprops
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
            A rigid transform matrix.

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
        "Not able to find the best transformation for pre-alignment. "
        "Try changing the values for 'dice_overlap' and 'rotation_step'."
    )
    return np.eye(3)


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

        return matrix[:2, :]

    @staticmethod
    def get_tissue_regions(
        fixed_image: np.ndarray,
        fixed_mask: np.ndarray,
        moving_image: np.ndarray,
        moving_mask: np.ndarray,
    ) -> Tuple[np.array, np.array, np.array, np.array, tuple]:
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
                - np.ndarray - A cropped image containing tissue region.
                - np.ndarray - A cropped image containing tissue mask.
                - np.ndarray - A cropped image containing tissue region.
                - np.ndarray - A cropped image containing tissue mask.
                - tuple - Bounding box (min_row, min_col, max_row, max_col).

        """
        if len(fixed_mask.shape) != 2:
            fixed_mask = fixed_mask[:, :, 0]
        if len(moving_mask.shape) != 2:
            moving_mask = moving_mask[:, :, 0]

        fixed_mask = np.uint8(fixed_mask > 0)
        moving_mask = np.uint8(moving_mask > 0)

        regions = regionprops(fixed_mask)
        fixed_minr, fixed_minc, fixed_maxr, fixed_maxc = regions[0].bbox
        regions = regionprops(moving_mask)
        moving_minr, moving_minc, moving_maxr, moving_maxc = regions[0].bbox
        minc, maxc, minr, maxr = (
            np.min([fixed_minc, moving_minc]),
            np.max([fixed_maxc, moving_maxc]),
            np.min([fixed_minr, moving_minr]),
            np.max([fixed_maxr, moving_maxr]),
        )

        fixed_tissue_image = fixed_image[minr:maxr, minc:maxc]
        fixed_tissue_mask = fixed_mask[minr:maxr, minc:maxc]
        moving_tissue_image = moving_image[minr:maxr, minc:maxc]
        moving_tissue_mask = moving_mask[minr:maxr, minc:maxc]
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
            (minr, minc, maxr, maxc),
        )

    @staticmethod
    def find_points_inside_boundary(mask: np.ndarray, points: np.ndarray):
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
        bound_points, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        included_points_ind = []
        for bound_p in bound_points:
            bound_p = path.Path(np.squeeze(bound_p))
            ind = bound_p.contains_points(points).nonzero()
            included_points_ind = np.hstack([included_points_ind, ind[0]])
        return included_points_ind.astype(int)

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
        included_indx = self.find_points_inside_boundary(
            fixed_mask, fixed_matched_points
        )
        fixed_matched_points, moving_matched_points, quality = (
            fixed_matched_points[included_indx, :],
            moving_matched_points[included_indx, :],
            quality[included_indx],
        )
        included_indx = self.find_points_inside_boundary(
            moving_mask, moving_matched_points
        )
        fixed_matched_points, moving_matched_points, quality = (
            fixed_matched_points[included_indx, :],
            moving_matched_points[included_indx, :],
            quality[included_indx],
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

    def register(
        self,
        fixed_img: np.ndarray,
        moving_img: np.ndarray,
        fixed_mask: np.ndarray,
        moving_mask: np.ndarray,
        transform_initializer: np.ndarray = None,
    ) -> np.ndarray:
        """Image Registration.

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

        if not transform_initializer:
            transform_initializer = prealignment(
                fixed_img[:, :, 0], moving_img[:, :, 0], fixed_mask, moving_mask
            )

        moving_img = cv2.warpAffine(
            moving_img, transform_initializer[0:-1][:], fixed_img.shape[:2][::-1]
        )
        moving_mask = cv2.warpAffine(
            moving_mask, transform_initializer[0:-1][:], fixed_img.shape[:2][::-1]
        )

        (
            fixed_tissue,
            fixed_mask,
            moving_tissue,
            moving_mask,
            tissue_top_left_coord,
        ) = self.get_tissue_regions(fixed_img, fixed_mask, moving_img, moving_mask)
        features = self.extract_features(fixed_tissue, moving_tissue)
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

        moving_tissue = cv2.warpAffine(
            moving_tissue, tissue_transform, fixed_tissue.shape[:2][::-1]
        )
        moving_mask = cv2.warpAffine(
            moving_mask, tissue_transform, fixed_tissue.shape[:2][::-1]
        )

        blocks_bounding_box = [
            [
                0,
                int(np.floor(fixed_tissue.shape[0] / 2)),
                0,
                int(np.floor(fixed_tissue.shape[1] / 2)),
            ],
            [
                0,
                int(np.floor(fixed_tissue.shape[0] / 2)),
                int(np.ceil(fixed_tissue.shape[1] / 2)),
                fixed_tissue.shape[1],
            ],
            [
                int(np.ceil(fixed_tissue.shape[0] / 2)),
                fixed_tissue.shape[0],
                0,
                int(np.floor(fixed_tissue.shape[1] / 2)),
            ],
            [
                int(np.ceil(fixed_tissue.shape[0] / 2)),
                fixed_tissue.shape[0],
                int(np.ceil(fixed_tissue.shape[1] / 2)),
                fixed_tissue.shape[1],
            ],
        ]
        fixed_matched_points, moving_matched_points, quality = [], [], []
        for _index, bounding_box in enumerate(blocks_bounding_box):
            fixed_block = fixed_tissue[
                bounding_box[0] : bounding_box[1], bounding_box[2] : bounding_box[3], :
            ]
            moving_block = moving_tissue[
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
            quality,
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

        translation_ = np.array(
            [
                [1, 0, -tissue_top_left_coord[1]],
                [0, 1, -tissue_top_left_coord[0]],
                [0, 0, 1],
            ]
        )
        translation = np.array(
            [
                [1, 0, tissue_top_left_coord[1]],
                [0, 1, tissue_top_left_coord[0]],
                [0, 0, 1],
            ]
        )
        tissue_transform = np.vstack([tissue_transform, np.array([0, 0, 1])])
        block_transform = np.vstack([block_transform, np.array([0, 0, 1])])
        return np.matmul(
            np.matmul(
                np.matmul(np.matmul(translation, block_transform), tissue_transform),
                translation_,
            ),
            transform_initializer,
        )
