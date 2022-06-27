import math
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import torch
import torchvision
from matplotlib import path
from skimage import color, filters, measure, morphology
from skimage.measure import label, regionprops
from torchvision.models._utils import IntermediateLayerGetter

from tiatoolbox.wsicore.wsireader import WSIReader


def get_tissue_regions(target_I, target_M, source_I, source_M, isDisplay):
    # get bounding box for tissue region in target image
    label_img = label(target_M)
    regions = regionprops(label_img)
    target_minr, target_minc, target_maxr, target_maxc = regions[0].bbox
    target_maxr = (
        target_maxr
        + math.ceil((target_maxr - target_minr) / 2) * 2
        - (target_maxr - target_minr)
    )
    target_maxc = (
        target_maxc
        + math.ceil((target_maxc - target_minc) / 2) * 2
        - (target_maxc - target_minc)
    )

    # get bounding box for tissue region in source image
    label_img = label(source_M)
    regions = regionprops(label_img)
    source_minr, source_minc, source_maxr, source_maxc = regions[0].bbox
    source_maxr = (
        source_maxr
        + math.ceil((source_maxr - source_minr) / 2) * 2
        - (source_maxr - source_minr)
    )
    source_maxc = (
        source_maxc
        + math.ceil((source_maxc - source_minc) / 2) * 2
        - (source_maxc - source_minc)
    )

    minc, maxc, minr, maxr = (
        np.min([target_minc, source_minc]),
        np.min([target_maxc, source_maxc]),
        np.min([target_minr, source_minr]),
        np.min([target_maxr, source_maxr]),
    )

    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    target_tissue_I = target_I[minr:maxr, minc:maxc]
    target_tissue_M = target_M[minr:maxr, minc:maxc]
    source_tissue_I = source_I[minr:maxr, minc:maxc]
    source_tissue_M = source_M[minr:maxr, minc:maxc]

    if isDisplay:
        plt.subplot(121)
        plt.imshow(target_I)
        plt.plot(bx, by, "-b", linewidth=2.5)
        plt.subplot(122)
        plt.imshow(source_I)
        plt.plot(bx, by, "-b", linewidth=2.5)
        plt.show()

    source_tissue_I[np.all(source_tissue_I == 0, axis=-1)] = 243
    return (
        target_tissue_I,
        target_tissue_M,
        source_tissue_I,
        source_tissue_M,
        (minr, minc, maxr, maxc),
    )


class RegistrationParameters:
    resolution: float = 0.3125
    number_of_rotations: int = 10  # for prealignment
    patch_size: Tuple[int] = (224, 224)
    model = torchvision.models.vgg16(True)
    feature_extractor = IntermediateLayerGetter(
        model.features,
        return_layers={"16": "block3_pool", "23": "block4_pool", "30": "block5_pool"},
    )


class Data:
    @classmethod
    def preprocess(cls, target, source, echo=True):
        def image_entropy(image):
            return filters.rank.entropy(image, morphology.disk(3))

        def normalize(image):
            scaled = (image - np.min(image)) / (np.max(image) - np.min(image))
            return scaled

        def histogram_correction(source, target):
            oldshape = source.shape
            source = source.ravel()
            target = target.ravel()

            s_values, bin_idx, s_counts = np.unique(
                source, return_inverse=True, return_counts=True
            )
            t_values, t_counts = np.unique(target, return_counts=True)

            sq = np.cumsum(s_counts).astype(np.float64)
            sq /= sq[-1]
            tq = np.cumsum(t_counts).astype(np.float64)
            tq /= tq[-1]
            interp_t_values = np.interp(sq, tq, t_values)
            return interp_t_values[bin_idx].reshape(oldshape)

        source, target = normalize(source), normalize(target)
        source_entropy, target_entropy = image_entropy(source), image_entropy(target)

        if echo:
            print("Source entropy: ", np.mean(source_entropy))
            print("Target entropy: ", np.mean(target_entropy))

        if np.mean(target_entropy) > np.mean(source_entropy):
            source = histogram_correction(source, target)
        else:
            target = histogram_correction(target, source)

        source, target = normalize(source), normalize(target)
        return target, source

    @classmethod
    def read_image(cls, wsi_path, resolution):
        wsi_reader = WSIReader.open(input_img=wsi_path)
        wsi_thumb = wsi_reader.slide_thumbnail(resolution=resolution, units="power")
        wsi_thumb = color.rgb2gray(wsi_thumb)
        return wsi_thumb

    @classmethod
    def get_mask(cls, grayscale):
        ret, mask = cv2.threshold(grayscale, np.mean(grayscale), 255, cv2.THRESH_BINARY)
        mask = morphology.remove_small_objects(mask == 0, min_size=1000, connectivity=2)
        mask = morphology.binary_opening(mask, morphology.disk(3))
        mask = morphology.remove_small_objects(mask == 1, min_size=1000, connectivity=2)
        mask = morphology.remove_small_holes(mask, area_threshold=100)

        # remove all the objects while keep the biggest object only
        label_img = measure.label(mask)
        regions = measure.regionprops(label_img)
        mask = mask.astype(bool)
        all_area = [i.area for i in regions]
        second_max = max([i for i in all_area if i != max(all_area)])
        mask = morphology.remove_small_objects(mask, min_size=second_max + 1)
        mask = mask.astype(np.uint8)
        return mask


class Registration:
    @classmethod
    def extract_features(cls, target, source, param):
        # resize image
        x_scale = 1.0 * np.array(target.shape[:2]) / param.patch_size
        y_scale = 1.0 * np.array(source.shape[:2]) / param.patch_size
        target_cnn = cv2.resize(target, param.patch_size)
        source_cnn = cv2.resize(source, param.patch_size)

        # input image noramlization
        target_cnn = target_cnn / 255
        source_cnn = source_cnn / 255

        # changing the channel ordering
        target_cnn = np.moveaxis(target_cnn, -1, 0)
        source_cnn = np.moveaxis(source_cnn, -1, 0)

        target_cnn = np.expand_dims(target_cnn, axis=0)
        source_cnn = np.expand_dims(source_cnn, axis=0)
        cnn_input = np.concatenate((target_cnn, source_cnn), axis=0)

        x = torch.from_numpy(cnn_input).type(torch.float32)
        # x = torch.rand(2, 3, 224, 224)
        features = param.feature_extractor(x)
        return features, x_scale, y_scale

    @classmethod
    def feature_mapping(cls, features, x_scale, y_scale):
        def pairwise_distance(
            X, Y
        ):  # for this function shape of M and N should be same
            assert len(X.shape) == len(Y.shape)
            N = X.shape[0]
            M = Y.shape[0]
            D = len(X.shape)

            return np.linalg.norm(
                np.repeat(np.expand_dims(X, axis=0), M, axis=0)
                - np.repeat(np.expand_dims(Y, axis=1), N, axis=1),
                axis=D,
            )

        def pd_expand(PD, k):
            N0 = np.int(np.sqrt(PD.shape[0]))
            N1 = k * N0
            L0, L1 = N0**2, N1**2
            Cmat = np.kron(
                np.arange(L0).reshape([N0, N0]), np.ones([k, k], dtype="int32")
            )
            i = np.repeat(Cmat.reshape([L1, 1]), L1, axis=1)
            j = np.repeat(Cmat.reshape([1, L1]), L1, axis=0)
            return PD[i, j]

        def finding_match(PD):
            seq = np.arange(PD.shape[0])
            amin1 = np.argmin(PD, axis=1)
            C = np.array([seq, amin1]).transpose()
            min1 = PD[seq, amin1]
            mask = np.zeros_like(PD)
            mask[seq, amin1] = 1
            masked = np.ma.masked_array(PD, mask)
            min2 = np.amin(masked, axis=1)
            return C, np.array(min2 / min1)

        pool3 = features["block3_pool"].detach().numpy()
        pool4 = features["block4_pool"].detach().numpy()
        pool5 = features["block5_pool"].detach().numpy()

        # flatten
        DX1, DY1 = np.reshape(pool3[0, :, :, :], [-1, 256]), np.reshape(
            pool3[1, :, :, :], [-1, 256]
        )
        DX2, DY2 = np.reshape(pool4[0, :, :, :], [-1, 512]), np.reshape(
            pool4[1, :, :, :], [-1, 512]
        )
        DX3, DY3 = np.reshape(pool5[0, :, :, :], [-1, 512]), np.reshape(
            pool5[1, :, :, :], [-1, 512]
        )

        # normalization
        DX1, DY1 = DX1 / np.std(DX1), DY1 / np.std(DY1)
        DX2, DY2 = DX2 / np.std(DX2), DY2 / np.std(DY2)
        DX3, DY3 = DX3 / np.std(DX3), DY3 / np.std(DY3)

        # compute feature space distance
        PD1 = pairwise_distance(DX1, DY1)
        PD2 = pd_expand(pairwise_distance(DX2, DY2), 2)
        PD3 = pd_expand(pairwise_distance(DX3, DY3), 4)
        PD = 1.414 * PD1 + PD2 + PD3

        del DX1, DY1, DX2, DY2, DX3, DY3, PD1, PD2, PD3

        seq = np.array(
            [[i, j] for i in range(pool3.shape[2]) for j in range(pool3.shape[2])],
            dtype="int32",
        )
        X = np.array(seq, dtype="float32") * 8.0 + 4.0
        Y = np.array(seq, dtype="float32") * 8.0 + 4.0

        # normalize
        X = (X - 112.0) / 224.0
        Y = (Y - 112.0) / 224.0

        # prematch and select points
        C_all, quality = finding_match(PD)
        tau_max = np.max(quality)
        while np.where(quality >= tau_max)[0].shape[0] <= 128:  # 128
            tau_max -= 0.01

        C = C_all[np.where(quality >= tau_max)]
        cnt = C.shape[0]

        # select prematched feature points
        X, Y = X[C[:, 1]], Y[C[:, 0]]
        PD = PD[
            np.repeat(np.reshape(C[:, 1], [cnt, 1]), cnt, axis=1),
            np.repeat(np.reshape(C[:, 0], [1, cnt]), cnt, axis=0),
        ]

        X, Y = ((X * 224.0) + 112.0) * x_scale, ((Y * 224.0) + 112.0) * y_scale
        return X, Y, np.amin(PD, axis=1)

    @classmethod
    def estimate_affine_transform(cls, points_0, points_1):
        nb = min(len(points_0), len(points_1))
        # Pad the data with ones, so that our transformation can do translations
        x = np.hstack([points_0[:nb], np.ones((nb, 1))])
        y = np.hstack([points_1[:nb], np.ones((nb, 1))])

        # Solve the least squares problem X * A = Y to find our transform. matrix A
        matrix = np.linalg.lstsq(x, y, rcond=-1)[0].T
        matrix[-1, :] = [0, 0, 1]
        return matrix

    @classmethod
    def perform_prealignment(cls, target_I, source_I, target_M, source_M, param):
        def compute_center_of_mass(grayI, mask):
            complement = (1 - grayI) * mask
            cy, cx = ndi.center_of_mass(complement)
            return [cx, cy]

        def dice(mask_1, mask_2):
            if mask_1.shape != mask_2.shape:
                x_size = max(mask_1.shape[1], mask_2.shape[1])
                y_size = max(mask_1.shape[0], mask_2.shape[0])

                right_y, right_x = y_size - mask_1.shape[0], x_size - mask_1.shape[1]
                mask_1 = np.pad(mask_1, [(0, right_y), (0, right_x)], mode="constant")

                right_y, right_x = y_size - mask_2.shape[0], x_size - mask_2.shape[1]
                mask_2 = np.pad(mask_2, [(0, right_y), (0, right_x)], mode="constant")

            mask_1 = mask_1.astype(np.bool)
            mask_2 = mask_2.astype(np.bool)
            return (
                2 * np.logical_and(mask_1, mask_2).sum() / (mask_1.sum() + mask_2.sum())
            )

        print("== Performing pre-alignment ==")
        targetCOM = compute_center_of_mass(target_I, target_M)
        sourceCOM = compute_center_of_mass(source_I, source_M)
        # plt.subplot(121)
        # plt.imshow(target_I, cmap='gray')
        # plt.plot(targetCOM[0], targetCOM[1], '*b')
        # plt.subplot(122)
        # plt.imshow(source_I, cmap='gray')
        # plt.plot(sourceCOM[0], sourceCOM[1], 'r*')
        # plt.show()

        # find translation difference between target and source using COM
        comT = np.array(
            [
                [1, 0, targetCOM[0] - sourceCOM[0]],
                [0, 1, targetCOM[1] - sourceCOM[1]],
                [0, 0, 1],
            ]
        )

        # search for the best rotation angle
        list_angles = np.around(np.linspace(0, 350, param.number_of_rotations))
        originT_COM_ = [[1, 0, -targetCOM[0]], [0, 1, -targetCOM[1]], [0, 0, 1]]
        originT_COM = [[1, 0, targetCOM[0]], [0, 1, targetCOM[1]], [0, 0, 1]]

        all_dice = []
        all_transform = []
        for i in range(len(list_angles)):
            theta = np.radians(list_angles[i])
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

            combinedT_COM = np.matmul(
                np.matmul(np.matmul(originT_COM, R), originT_COM_), comT
            )
            warped_source_M = cv2.warpAffine(
                source_M, combinedT_COM[0:-1][:], target_I.shape[:2][::-1]
            )
            dice_COM = dice(target_M, warped_source_M)

            all_dice.append(dice_COM)
            all_transform.append(combinedT_COM)

        return all_transform[all_dice.index(max(all_dice))]

    @classmethod
    def perform_refinement(
        cls, target_I, source_I, target_M, source_M, param, initial_transform
    ):
        def remove_points_using_mask(mask, X, Y, dist, indx):
            kernel = np.ones((50, 50), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask_X = np.array([], dtype=np.int64).reshape(0, 2)
            mask_Y = np.array([], dtype=np.int64).reshape(0, 2)
            mask_dist = np.array([], dtype=np.int64)
            bound_points, _ = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            for bound_p in bound_points:
                bound_p = np.squeeze(bound_p)
                bound_p = path.Path(bound_p)

                if indx == 1:
                    included_points_index = bound_p.contains_points(X)
                else:
                    included_points_index = bound_p.contains_points(Y)
                tempX = X[included_points_index, :]
                tempY = Y[included_points_index, :]
                temp_dist = dist[included_points_index]

                mask_X = np.vstack([mask_X, tempX])
                mask_Y = np.vstack([mask_Y, tempY])
                mask_dist = np.hstack([mask_dist, temp_dist])
            return mask_X, mask_Y, mask_dist

        def show_matching_points(target, source, X, Y, dist):
            keypoints1 = []
            keypoints2 = []
            matchingPoints = []
            for i in range(len(dist)):
                matchingPoints.append(
                    cv2.DMatch(_distance=dist[i], _imgIdx=0, _queryIdx=i, _trainIdx=i)
                )
                keypoints1.append(cv2.KeyPoint(X[i, 0], X[i, 1], 5, class_id=0))
                keypoints2.append(cv2.KeyPoint(Y[i, 0], Y[i, 1], 5, class_id=0))

            matchingPoints = sorted(
                matchingPoints, key=lambda x: x.distance
            )  # Sort them in the order of their distance.
            if np.max(target) <= 1:
                target = 255 * target
            target = target.astype(np.uint8)

            if np.max(source) <= 1:
                source = 255 * source
            source = source.astype(np.uint8)
            img3 = cv2.drawMatches(
                target, keypoints1, source, keypoints2, matchingPoints, None, flags=2
            )
            return img3

        print("== Performing refinement using DFBR approach ==")
        target_I = np.repeat(target_I[:, :, np.newaxis], 3, axis=2)
        source_I = np.repeat(source_I[:, :, np.newaxis], 3, axis=2)

        multiscale_features, x_scale, y_scale = Registration.extract_features(
            target_I, source_I, param
        )
        target_points, source_points, dist = Registration.feature_mapping(
            multiscale_features, x_scale, y_scale
        )
        target_points, source_points = (
            target_points[:, [1, 0]],
            source_points[:, [1, 0]],
        )

        # remove the points which are outside the dilated mask of the target image
        (
            filtered_target_points,
            filtered_source_points,
            filtered_dist,
        ) = remove_points_using_mask(target_M, target_points, source_points, dist, 1)
        (
            filtered_target_points,
            filtered_source_points,
            filtered_dist,
        ) = remove_points_using_mask(
            source_M, filtered_target_points, filtered_source_points, filtered_dist, 2
        )

        # matchingI_before = show_matching_points(target_I, source_I, target_points, source_points, dist)
        # matchingI_after = show_matching_points(target_I, source_I, filtered_target_points, filtered_source_points,
        # 									   filtered_dist)
        #
        # fig = plt.figure(figsize=(25, 10))
        # plt.subplot(121)
        # plt.imshow(matchingI_before)
        # plt.subplot(122)
        # plt.imshow(matchingI_after)
        # plt.show()

        return Registration.estimate_affine_transform(
            filtered_target_points, filtered_source_points
        )  # compute transformation params, if removeScale == 1, remove the scale params

    @staticmethod
    def run_registration(target_wsi_path, source_wsi_path, param):
        target_I = Data.read_image(target_wsi_path, param.resolution)
        source_I = Data.read_image(source_wsi_path, param.resolution)
        [target_I, source_I] = Data.preprocess(target_I, source_I, echo=True)
        target_M = Data.get_mask(target_I)
        source_M = Data.get_mask(source_I)

        # Pre-alignment
        pre_transform = Registration.perform_prealignment(
            target_I, source_I, target_M, source_M, param
        )
        warped_source_I = cv2.warpAffine(
            source_I, pre_transform[0:-1][:], target_I.shape[:2][::-1]
        )
        warped_source_M = cv2.warpAffine(
            source_M, pre_transform[0:-1][:], target_I.shape[:2][::-1]
        )
        [
            target_tissue_I,
            target_tissue_M,
            source_tissue_I,
            source_tissue_M,
            tissue_top_left_coor,
        ] = get_tissue_regions(target_I, target_M, warped_source_I, warped_source_M, 0)

        # DFBR refinement using tissue region
        tissue_transform = Registration.perform_refinement(
            target_tissue_I,
            source_tissue_I,
            target_tissue_M,
            source_tissue_M,
            param,
            pre_transform,
        )

        # perform block-wise tissue transform
        # fix translation offset
        return pre_transform, tissue_transform
