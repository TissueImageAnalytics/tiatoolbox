# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIALab, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""Visualisation and overlay functions used in tiatoolbox."""


from tiatoolbox.utils.misc import imread
from tiatoolbox.models.classification.patch_predictor import CNNPatchPredictor
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, get_wsireader
from tiatoolbox.wsicore.wsimeta import WSIMeta
import numpy as np
import cv2
import warnings


def _merge_patch_predictions(model_output, output_shape, scale=1):
    """Merge patch level predictions.

    Args:
        model_output (dict): Output produced by CNNPatchPredictor, containing
            predictions, patch coordinates and class probabilities (if requested).
        output_shape (tuple): Size of merged 2-dimensional output array.
        scale (float): How many times smaller the output array is to the original image.

    Returns:
        output (ndarray): 2D map of merged predictions.

    """
    coordinates = model_output["coordinates"]
    predictions = model_output["predictions"]

    patch_shape = (coordinates[0][2], coordinates[0][3])
    coordinates_x1 = np.unique([v[0] for v in coordinates]).tolist()
    coordinates_y1 = np.unique([v[1] for v in coordinates]).tolist()
    stride_shape = (sorted(coordinates_x1)[1], sorted(coordinates_y1)[1])

    if (
        stride_shape[0] < patch_shape[0] or stride_shape[1] < patch_shape[1]
    ) and "probabilities" not in model_output.keys():
        warnings.warn(
            "For a better result when using stride size < patch size, consider returning "
            "the probabilities. This will result in a smoother output."
        )

    if stride_shape == patch_shape or "probabilities" not in model_output.keys():
        output = np.zeros(np.array(output_shape))
        for idx, coords in enumerate(coordinates):
            prediction = predictions[idx]
            coords = np.round(np.array(coords) / scale).astype("int")
            output[coords[1] : coords[3], coords[0] : coords[2]] = prediction
    else:
        probabilities = model_output["probabilities"]
        num_classes = len(model_output["probabilities"][0])
        output = np.zeros([output_shape[0], output_shape[1], num_classes])
        # used to merge overlapping patches
        denominator = np.ones(np.array(output_shape))
        for idx, coords in enumerate(coordinates):
            probabilties_ = probabilities[idx]
            coords = np.round(np.array(coords) / scale).astype("int")
            output[coords[1] : coords[3], coords[0] : coords[2]] += probabilties_
            denominator[coords[1] : coords[3], coords[0] : coords[2]] += 1
        # deal with overlapping regions
        output = output / np.expand_dims(denominator, -1)
        # convert raw probabilities to preditions
        output = CNNPatchPredictor._postprocess(output)

    return output


def _get_patch_prediction_overlay(img, prediction, alpha, seed=456):
    """Generate an overlay, given a 2D prediction map.

    Args:
        img (ndarray): Image to overlay the results on top of.
        prediction (ndarray): 2D prediction map. Multi-class prediction should have
            values ranging from 0 to N-1, where N is the number of classes.
        alpha (float): Opacity value used for the overlay.
        seed (int): Random number seed used to determine random colours for overlay.

    Returns:
        overlay (ndarray): Overlaid result on top of the original image.

    """
    np.random.seed(seed)

    img = img.astype("uint8")
    overlay = img.copy()
    predicted_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # predicted_classes = sorted(np.unique(prediction).tolist())

    rgb_prediction = np.zeros(
        [prediction.shape[0], prediction.shape[1], 3], dtype=np.uint8
    )
    for predicted_class in predicted_classes:
        prediction_ = prediction.copy()
        prediction_single_class = prediction_ == predicted_class
        prediction_single_class = np.dstack(
            [prediction_single_class, prediction_single_class, prediction_single_class]
        )
        prediction_single_class = prediction_single_class.astype("uint8")
        random_colour = (np.random.choice(range(256), size=3)).astype("uint8")
        prediction_single_class *= random_colour
        rgb_prediction += prediction_single_class

    cv2.addWeighted(rgb_prediction, alpha, overlay, 1 - alpha, 0, overlay)

    return overlay.astype("uint8"), rgb_prediction


def visualise_patch_prediction(
    img_list,
    predictions_list,
    mode="tile",
    tile_resolution=None,
    resolution=1.25,
    units="power",
    alpha=0.5,
):
    """Generate patch-level overlay.

    Args:
        img: input image
        predictions: output of the model

    Returns:
        overlay: overlaid output.

    """
    if mode not in ["tile", "wsi"]:
        raise ValueError("`mode` must be either `tile` or `wsi`.")

    if mode == "tile" and tile_resolution is None:
        raise ValueError("If using `tile` mode, `tile_resolution` must be provided.")

    if len(img_list) > 1:
        warnings.warn(
            "When providing multiple whole-slide images / tiles, "
            "we save the overlays and return the locations "
            "to the corresponding files."
        )

    for idx, img_file in enumerate(img_list):
        if mode == "wsi":
            reader = get_wsireader(img_file)
        else:
            img = imread(img_file)
            slide_dims = np.array(img.shape[:2][::-1])
            metadata = WSIMeta(
                mpp=np.array([1.0, 1.0]),
                objective_power=tile_resolution,
                slide_dimensions=slide_dims,
                level_downsamples=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
                level_dimensions=[
                    slide_dims,
                    slide_dims / 2.0,
                    slide_dims / 4.0,
                    slide_dims / 8.0,
                    slide_dims / 16.0,
                    slide_dims / 32.0,
                ],
            )
            reader = VirtualWSIReader(
                img,
                metadata,
            )

        read_img = reader.slide_thumbnail(resolution=resolution, units=units)
        scale = reader.info.objective_power / resolution

        predictions = predictions_list[idx]
        merged_predictions = _merge_patch_predictions(
            predictions, read_img.shape[:2], scale
        )

        overlay, rgb_pred = _get_patch_prediction_overlay(
            read_img, merged_predictions, alpha=alpha
        )

    return overlay, rgb_pred
