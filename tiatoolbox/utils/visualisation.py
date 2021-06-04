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
import matplotlib as plt
import numpy as np


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

####
def colorize(ch, vmin, vmax):
    """Will clamp value value outside the provided range to vmax and vmin."""
    cmap = plt.get_cmap("jet")
    ch = np.squeeze(ch.astype("float32"))
    vmin = vmin if vmin is not None else ch.min()
    vmax = vmax if vmax is not None else ch.max()
    ch[ch > vmax] = vmax  # clamp value
    ch[ch < vmin] = vmin
    ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
    # take RGB from RGBA heat map
    ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
    return ch_cmap


####
def random_colors(N, bright=True):
    """Generate random colors.
    
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


####
def visualize_instances_map(
    input_image, inst_map, type_map=None, type_colour=None, line_thickness=2
):
    """Overlays segmentation results on image as contours.
    Args:
        input_image: input image
        inst_map: instance mask with unique value for every object
        type_map: type mask with unique value for every class
        type_colour: a dict of {type : colour} , `type` is from 0-N
                     and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours
    Returns:
        overlay: output image with segmentation overlay as contours
    """
    overlay = np.copy((input_image).astype(np.uint8))

    inst_list = list(np.unique(inst_map))  # get list of instances
    inst_list.remove(0)  # remove background

    inst_rng_colors = random_colors(len(inst_list))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for inst_idx, inst_id in enumerate(inst_list):
        inst_map_mask = np.array(inst_map == inst_id, np.uint8)  # get single object
        y1, y2, x1, x2 = get_bounding_box(inst_map_mask)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        inst_map_crop = inst_map_mask[y1:y2, x1:x2]
        contours_crop = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # only has 1 instance per map, no need to check #contour detected by opencv
        contours_crop = np.squeeze(
            contours_crop[0][0].astype("int32")
        )  # * opencv protocol format may break
        contours_crop += np.asarray([[x1, y1]])  # index correction
        if type_map is not None:
            type_map_crop = type_map[y1:y2, x1:x2]
            type_id = np.unique(type_map_crop).max()  # non-zero
            inst_colour = type_colour[type_id]
        else:
            inst_colour = (inst_rng_colors[inst_idx]).tolist()
        cv2.drawContours(overlay, [contours_crop], -1, inst_colour, line_thickness)
    return overlay


####
def visualize_instances_dict(
    input_image, inst_dict, draw_dot=False, type_colour=None, line_thickness=2
):
    """Overlays segmentation results (dictionary) on image as contours.
    Args:
        input_image: input image
        inst_dict: dict of output prediction, defined as in this library
        draw_dot: to draw a dot for each centroid
        type_colour: a dict of {type_id : (type_name, colour)} , 
                     `type_id` is from 0-N and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours
    """
    overlay = np.copy((input_image))

    inst_rng_colors = random_colors(len(inst_dict))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for idx, [inst_id, inst_info] in enumerate(inst_dict.items()):
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
