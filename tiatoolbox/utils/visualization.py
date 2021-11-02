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
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl


def overlay_patch_prediction(
    img: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.35,
    label_info: dict = None,
    min_val: float = 0.0,
    ax=None,
    return_ax: bool = True,
):
    """Generate an overlay, given a 2D prediction map.

    Args:
        img (ndarray): Input image to overlay the results on top of.
        prediction (ndarray): 2D prediction map. Multi-class prediction should have
            values ranging from 0 to N-1, where N is the number of classes.
        label_info (dict): A dictionary contains the mapping for each integer value
            within `prediction` to its string and color. [int] : (str, (int, int, int)).
            By default, integer will be taken as label and color will be random.
        min_val (float):
        alpha (float): Opacity value used for the overlay.
        ax (ax): Matplotlib ax object.
        return_ax:
    
    Returns:

    """
    if img.shape[:2] != prediction.shape[:2]:
        raise ValueError(
            "Mismatch shape `img` {0} vs `prediction` {1}.".format(
                img.shape[:2], prediction.shape[:2]
            )
        )

    if np.issubdtype(img.dtype, np.floating):
        if not (img.max() <= 1.0 and img.min() >= 0):
            raise ValueError("Not support float `img` outside [0, 1].")
        img = np.array(img * 255, dtype=np.uint8)

    # if `min_val` is defined, only display the overlay for areas with pred > min_val
    if min_val > 0:
        prediction_sel = prediction >= min_val

    overlay = img.copy()

    # generate random colours
    predicted_classes = sorted(np.unique(prediction).tolist())
    if label_info is None:
        np.random.seed(123)
        label_info = {}
        for label_uid in predicted_classes:
            random_colour = np.random.choice(range(256), size=3)
            label_info[label_uid] = (str(label_uid), random_colour)
    else:
        # may need better error message
        check_uid_list = predicted_classes.copy()
        for label_uid, (label_name, label_colour) in label_info.items():
            if label_uid in check_uid_list:
                check_uid_list.remove(label_uid)
            if not isinstance(label_uid, int):
                raise ValueError(
                    "Wrong `label_info` format: label_uid {0}".format(
                        [label_uid, (label_name, label_colour)]
                    )
                )
            if not isinstance(label_name, str):
                raise ValueError(
                    "Wrong `label_info` format: label_name {0}".format(
                        [label_uid, (label_name, label_colour)]
                    )
                )
            if not isinstance(label_colour, (tuple, list, np.ndarray)):
                raise ValueError(
                    "Wrong `label_info` format: label_colour {0}".format(
                        [label_uid, (label_name, label_colour)]
                    )
                )
            if len(label_colour) != 3:
                raise ValueError(
                    "Wrong `label_info` format: label_colour {0}".format(
                        [label_uid, (label_name, label_colour)]
                    )
                )
        #
        if len(check_uid_list) != 0:
            raise ValueError("Missing label for: {0}".format(check_uid_list))

    rgb_prediction = np.zeros(
        [prediction.shape[0], prediction.shape[1], 3], dtype=np.uint8
    )
    for label_uid, (_, overlay_rgb) in label_info.items():
        sel = prediction == label_uid
        rgb_prediction[sel] = overlay_rgb

    # add the overlay
    cv2.addWeighted(rgb_prediction, alpha, overlay, 1 - alpha, 0, overlay)
    overlay = overlay.astype(np.uint8)

    if min_val > 0.0:
        overlay[~prediction_sel] = img[~prediction_sel]

    if ax is None and not return_ax:
        return overlay

    # create colorbar parameters
    name_list = [v[0] for v in label_info.values()]
    color_list = [v[1] for v in label_info.values()]
    color_list = np.array(color_list) / 255
    uid_list = list(label_info.keys())
    cmap = mpl.colors.ListedColormap(color_list)
    colorbar_params = {
        "mappable": mpl.cm.ScalarMappable(cmap=cmap),
        "boundaries": uid_list + [uid_list[-1] + 1],
        "ticks": [b + 0.5 for b in uid_list],
        "spacing": "proportional",
        "orientation": "vertical",
    }

    # generate another ax, else using the provided
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(overlay)
    ax.axis("off")
    # generate colour bar
    cbar = plt.colorbar(**colorbar_params)
    cbar.ax.set_yticklabels(name_list)
    cbar.ax.tick_params(labelsize=12)

    return ax


def overlay_patch_probmap(
    img: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.35,
    colour_map: str = "jet",
    min_val: float = 0.0,
    ax=None,
    return_ax: bool = True,
):
    """Generate an overlay, given a 2D prediction map.

    Args:
        img (ndarray): Input image to overlay the results on top of. Assumed to be HW.
        prediction (ndarray): 2D prediction map. Multi-class prediction should have
            values ranging from 0 to N-1, where N is the number of classes.
        alpha (float): Opacity value used for the overlay.
        colour_map (string): 
        min_val (float):
        ax (ax): Matplotlib ax object.
        return_ax (bool):

    Returns:

    """
    if img.shape[:2] != prediction.shape[:2]:
        raise ValueError(
            "Mismatch shape `img` {0} vs `prediction` {1}.".format(
                img.shape[:2], prediction.shape[:2]
            )
        )

    assert np.issubdtype(
        prediction.dtype, np.floating
    ), "`prediction` must be of type float"
    if not (prediction.max() <= 1.0 and prediction.min() >= 0):
        raise ValueError("Not support float `prediction` outside [0, 1].")

    if np.issubdtype(img.dtype, np.floating):
        if not (img.max() <= 1.0 and img.min() >= 0):
            raise ValueError("Not support float `img` outside [0, 1].")
        img = np.array(img * 255, dtype=np.uint8)

    # if `min_val` is defined, only display the overlay for areas with prob > min_val
    if min_val > 0.0:
        assert (
            min_val >= 0 and min_val <= 1
        ), "`min_val` must be a probability between 0 and 1"
        prediction_sel = prediction >= min_val

    overlay = img.copy()

    cmap = plt.get_cmap(colour_map)
    prediction = np.squeeze(prediction.astype("float32"))
    # take RGB from RGBA heat map
    rgb_prediction = (cmap(prediction)[..., :3] * 255).astype("uint8")

    # add the overlay
    # cv2.addWeighted(rgb_prediction, alpha, overlay, 1 - alpha, 0, overlay)
    overlay = (1 - alpha) * rgb_prediction + alpha * overlay
    overlay[overlay > 255.0] = 255.0
    overlay = overlay.astype(np.uint8)

    if min_val > 0.0:
        overlay[~prediction_sel] = img[~prediction_sel]

    if ax is None and not return_ax:
        return overlay

    colorbar_params = {
        "mappable": mpl.cm.ScalarMappable(cmap="jet"),
        "spacing": "proportional",
        "orientation": "vertical",
    }

    # generate another ax, else using the provided
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(overlay)
    ax.axis("off")
    # generate colour bar
    cbar = plt.colorbar(**colorbar_params)
    cbar.ax.tick_params(labelsize=12)

    return ax
