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

from tiatoolbox.utils.misc import get_pretrained_model_info


def overlay_patch_prediction(img, prediction, pretrained_model, alpha=0.35):
    """Generate an overlay, given a 2D prediction map.

    Args:
        img (ndarray): Image to overlay the results on top of.
        prediction (ndarray): 2D prediction map. Multi-class prediction should have
            values ranging from 0 to N-1, where N is the number of classes.
        pretrained_model (str): Pretrained model used for generating the predictions.
            This is used to determine predefined RGB overlay colours.
        alpha (float): Opacity value used for the overlay.

    """
    img = img.astype("uint8")
    overlay = img.copy()

    pretrained_model = pretrained_model.lower()
    _, dataset = pretrained_model.split("-")
    # get label/model information
    pretrained_yml = get_pretrained_model_info()
    # focus on specific dataset
    if dataset in pretrained_yml:
        pretrained_info = pretrained_yml[dataset]
        # get a dictionary of label ID and overlay colour
        label_dict = pretrained_info["overlay_info"]
        class_names = list(pretrained_info["label_info"].values())
    else:
        np.random.seed(123)
        # if pretrained_model is not provided, generate random colours
        predicted_classes = sorted(np.unique(prediction).tolist())
        label_dict = {}
        for label in predicted_classes:
            label_dict[label] = (np.random.choice(range(256), size=3)).astype("uint8")
        class_names = None

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

    # generate overlay with colourbar
    fig, ax = plt.subplots()
    ax.imshow(overlay)
    ax.axis("off")
    # only save colourbar if class names are available
    if class_names is not None:
        cbar = plt.colorbar(**colorbar_params)
        cbar.ax.set_yticklabels(class_names)
        cbar.ax.tick_params(labelsize=12)

    return fig
