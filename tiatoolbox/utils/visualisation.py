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

import numpy as np
import os
import pathlib
import json
import cv2
import warnings
import matplotlib as mpl

from tiatoolbox.utils.misc import imread, imwrite, get_pretrained_model_info
from tiatoolbox.models.classification.patch_predictor import CNNPatchPredictor
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, get_wsireader
from tiatoolbox.wsicore.wsimeta import WSIMeta


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
            "For a better result when using stride size < patch size, consider "
            "returning the probabilities. This will result in a smoother output."
        )

    if stride_shape == patch_shape or "probabilities" not in model_output.keys():
        output = np.full(output_shape, -1)
        for idx, coords in enumerate(coordinates):
            prediction = predictions[idx]
            coords = np.round(np.array(coords) / scale).astype("int")
            output[coords[1] : coords[3], coords[0] : coords[2]] = prediction
    else:
        probabilities = model_output["probabilities"]
        num_classes = len(model_output["probabilities"][0])
        output = np.full([output_shape[0], output_shape[1], num_classes], -1.0)
        # used to merge overlapping patches
        denominator = np.ones(np.array(output_shape))
        for idx, coords in enumerate(coordinates):
            probabilties_ = probabilities[idx]
            coords = np.round(np.array(coords) / scale).astype("int")
            output[coords[1] : coords[3], coords[0] : coords[2]] += probabilties_
            denominator[coords[1] : coords[3], coords[0] : coords[2]] += 1
        # deal with overlapping regions
        output = output / np.expand_dims(denominator, -1)
        selection = denominator >= 2
        # convert raw probabilities to preditions
        output = CNNPatchPredictor.postprocess(output)
        # set the background to -1
        output[~selection] = -1

    return output


def _get_patch_prediction_overlay(
    img, prediction, alpha, pretrained_model, random_colours, random_seed
):
    """Generate an overlay, given a 2D prediction map.

    Args:
        img (ndarray): Image to overlay the results on top of.
        prediction (ndarray): 2D prediction map. Multi-class prediction should have
            values ranging from 0 to N-1, where N is the number of classes.
        alpha (float): Opacity value used for the overlay.
        pretrained_model (str): Pretrained model used for generating the predictions.
            This is used to determine predefined RGB overlay colours.
        random_colours (bool): Whether to use random colours for the overlay. If set
            to False, then the predefined colours will be used.
        random_seed (int): Random number seed used to determine random colours
            for overlay.

    Returns:
        overlay (ndarray): Overlaid result on top of the original image.
        rgb_prediction (ndarray): RGB prediction map.

    """
    img = img.astype("uint8")
    overlay = img.copy()

    if random_colours:
        np.random.seed(random_seed)
        # if pretrained_model is not provided, generate random colours
        predicted_classes = sorted(np.unique(prediction).tolist())
        label_dict = {}
        for label in predicted_classes:
            label_dict[label] = (np.random.choice(range(256), size=3)).astype("uint8")
    else:
        pretrained_model = pretrained_model.lower()
        _, dataset = pretrained_model.split("-")
        # get label/model information
        pretrained_yml = get_pretrained_model_info()
        # focus on specific dataset
        pretrained_info = pretrained_yml[dataset]
        # get a dictionary of label ID and overlay colour
        label_dict = pretrained_info["overlay_info"]

    rgb_prediction = np.zeros(
        [prediction.shape[0], prediction.shape[1], 3], dtype=np.uint8
    )
    for label, overlay_rgb in label_dict.items():
        prediction_ = prediction.copy()
        prediction_single_class = prediction_ == label
        prediction_single_class = np.dstack(
            [prediction_single_class, prediction_single_class, prediction_single_class]
        )
        prediction_single_class = prediction_single_class.astype("uint8")
        prediction_single_class *= np.array(overlay_rgb).astype("uint8")
        rgb_prediction += prediction_single_class

    # add the overlay
    cv2.addWeighted(rgb_prediction, alpha, overlay, 1 - alpha, 0, overlay)
    overlay = overlay.astype("uint8")

    # create colorbar parameters
    colors_list = np.array(list(label_dict.values()), dtype=np.float) / 255
    bounds = list(label_dict.keys())
    cmap = mpl.colors.ListedColormap(colors_list)
    colorbar_params = {
        "mappable": mpl.cm.ScalarMappable(cmap=cmap),
        "boundaries": bounds + [bounds[-1] + 1],
        "ticks": [b + 0.5 for b in bounds],
        "spacing": "proportional",
        "orientation": "vertical",
    }

    return overlay, rgb_prediction, colorbar_params


def visualise_patch_prediction(
    img_list,
    model_output_list,
    mode="tile",
    resolution=1.25,
    units="power",
    alpha=0.5,
    random_colours=False,
    random_seed=123,
    save_dir=None,
    return_colorbar=False,
):
    """Generate patch-level overlay.

    Args:
        img_list (list): List of input image paths.
        model_output_list (list): List of dictionaries output by the model.
        mode (str): Determines the format of the input images. Choose either `patch`,
            `tile` or `wsi`.
        resolution (float): Resolution of generated output.
        units (str): Units that the resolution argument corresponds to. Choose from
            either `level`, `power` or `mpp`.
        alpha (float): Used to determine how the transparent the overlay is. Must be
            between 0 and 1.
        random_colours (bool): Whether to use random colours for the overlay. If set
            to False, then the predefined colours will be used.
        save_dir (str): Output directory when processing multiple tiles and
                whole-slide images.
        return_colorbar (bool): whether to return the arguments to be used for colorbar.

    Returns:
        overlay (ndarray): overlaid output.
        rgb_array (ndarray): segmentation prediction map, where different colours
            denote different class predictions.
        colorbar_params (dictionary): [optional] The dictionary defining the parameters
            related to the colobar of the prediction map.

    """
    if len(img_list) != len(model_output_list):
        raise ValueError(
            "The lengths of `img_list` and `model_output_list` must be the same."
        )
    if mode not in ["tile", "wsi"]:
        raise ValueError("`mode` must be either `tile` or `wsi`.")

    if len(img_list) > 1:
        output_files = []  # generate a list of output file paths
        warnings.warn(
            "When providing multiple whole-slide images / tiles "
            "we save the overlays and return the locations "
            "to the corresponding files."
        )

    for idx, img_file in enumerate(img_list):
        img_file = pathlib.Path(img_file)
        basename = img_file.stem

        if len(img_list) > 1:
            with open(model_output_list[idx]) as json_file:
                model_output = json.load(json_file)
        else:
            model_output = model_output_list[idx]

        # get the resolution and pretrained model used during duing training
        process_resolution_mpp = model_output["resolution_mpp"]
        process_resolution_power = model_output["resolution_power"]
        pretrained_model = model_output["pretrained_model"]

        if mode == "wsi":
            reader = get_wsireader(img_file)
        else:
            img = imread(img_file)
            slide_dims = np.array(img.shape[:2][::-1])
            metadata = WSIMeta(
                mpp=np.array([process_resolution_mpp, process_resolution_mpp]),
                objective_power=process_resolution_power,
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
        if units == "power":
            scale = reader.info.objective_power / resolution
        elif units == "mpp":
            scale = (reader.info.mpp * resolution)[0]

        merged_predictions = _merge_patch_predictions(
            model_output, read_img.shape[:2], scale
        )

        overlay, rgb_array, colorbar_params = _get_patch_prediction_overlay(
            read_img,
            merged_predictions,
            alpha,
            pretrained_model,
            random_colours,
            random_seed,
        )

        if len(img_list) > 1:
            save_dir_ = os.path.join(save_dir, basename)
            if not os.path.exists(save_dir_):
                os.makedirs(save_dir_)

            # save overlay and prediction map
            imwrite(save_dir_ + "overlay.png", overlay)
            imwrite(save_dir_ + "rgb_prediction.png", rgb_array)
            output_files.append(
                [save_dir_ + "overlay.png", save_dir_ + "rgb_prediction.png"]
            )

            # set output to return locations of saved files
            output = output_files
        else:
            if return_colorbar:
                output = [overlay, rgb_array, colorbar_params]
            else:
                output = [overlay, rgb_array]

    return output
