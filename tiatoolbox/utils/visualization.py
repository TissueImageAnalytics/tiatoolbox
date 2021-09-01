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
import colorsys
import random

import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def overlay_patch_prediction(
    img: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.35,
    label_info: dict = None,
    ax=None,
):
    """Generate an overlay, given a 2D prediction map.

    Args:
        img (ndarray): Input image to overlay the results on top of.
        prediction (ndarray): 2D prediction map. Multi-class prediction should have
            values ranging from 0 to N-1, where N is the number of classes.
        label_info (dict): A dictionary contains the mapping for each integer value
            within `prediction` to its string and color. [int] : (str, (int, int, int)).
            By default, integer will be taken as label and color will be random.
        alpha (float): Opacity value used for the overlay.
        ax (ax): Matplotlib ax object.
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


def random_colors(num_colors, bright=True):
    """Generate a number of random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.

    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / num_colors, 1, brightness) for i in range(num_colors)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def overlay_instance_prediction(
    canvas, inst_dict, draw_dot=False, type_colour=None, line_thickness=2
):
    """Overlaying instance contours on image.

    Args:
        canvas (ndarray): image to be drawn upon.
        inst_dict (dict): dict of instances.
        draw_dot: to draw a dot for each centroid
        type_colour: a dict of {type_id : (type_name, colour)},
            `type_id` is from 0-N and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours

    Return:

    """
    overlay = np.copy((canvas))

    inst_rng_colors = random_colors(len(inst_dict))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for idx, [_, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colour is not None:
            inst_colour = type_colour[inst_info["type"]][1]
        else:
            inst_colour = (inst_rng_colors[idx]).tolist()
        cv2.drawContours(overlay, [inst_contour], -1, inst_colour, line_thickness)

        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple([int(v) for v in inst_centroid])
            overlay = cv2.circle(overlay, inst_centroid, 3, (255, 0, 0), -1)
    return overlay
