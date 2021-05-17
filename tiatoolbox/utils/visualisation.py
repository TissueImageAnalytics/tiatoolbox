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
        model_output (dict):
        output_shape (tuple):
        scale (float):

    Returns:
        output (ndarray): 2D map of merged predictions.

    """
    coordinates = model_output["coordinates"]
    predictions = model_output["predictions"]
    probabilities = model_output["probabilities"]

    patch_shape = (coordinates[0][2], coordinates[0][3])
    coordinates_x1 = np.unique([v[0] for v in coordinates]).tolist()
    coordinates_y1 = np.unique([v[1] for v in coordinates]).tolist()
    stride_shape = (sorted(coordinates_x1)[1], sorted(coordinates_y1)[1])

    if stride_shape == patch_shape:
        output = np.zeros(np.array(output_shape))
        for idx, coords in enumerate(coordinates):
            prediction = predictions[idx]
            coords = np.round(np.array(coords) / scale).astype("int")
            output[coords[1] : coords[3], coords[0] : coords[2]] = prediction
    else:
        num_classes = len(model_output["probabilities"][0])
        output = np.zeros([output_shape[0], output_shape[1], num_classes])
        denominator = np.ones(np.array(output_shape))
        for idx, coords in enumerate(coordinates):
            probabilties_ = probabilities[idx]
            coords = np.round(np.array(coords) / scale).astype("int")
            output[coords[1] : coords[3], coords[0] : coords[2]] += probabilties_
            denominator[coords[1] : coords[3], coords[0] : coords[2]] += 1
        output = output / np.expand_dims(denominator, -1)
        output = CNNPatchPredictor.__postprocess(output)

    return output


def _get_patch_prediction_overlay(img, prediction, alpha=0.25, seed=456):
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
    predicted_classes = np.unique(prediction)
    for idx in range(len(predicted_classes)):
        prediction_ = prediction.copy()
        prediction_single_class = prediction_ == predicted_classes[idx]
        prediction_single_class = np.dstack(
            [prediction_single_class, prediction_single_class, prediction_single_class]
        )
        prediction_single_class = prediction_single_class.astype("uint8")
        random_colour = (np.random.choice(range(256), size=3)).astype("uint8")
        prediction_single_class *= random_colour

        cv2.addWeighted(prediction_single_class, alpha, overlay, 1 - alpha, 0, overlay)

    return overlay.astype("uint8")


def overlay_patch_prediction(
    img_list,
    predictions_list,
    mode="tile",
    tile_resolution=None,
    resolution=1.25,
    units="power",
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

        overlay = _get_patch_prediction_overlay(read_img, merged_predictions)

    return overlay
