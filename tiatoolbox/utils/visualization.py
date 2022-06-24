"""Visualisation and overlay functions used in tiatoolbox."""
import colorsys
import random
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def random_colors(num_colors, bright=True):
    """Generate a number of random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.

    Args:
        num_colors(int):
            Number of perceptively different colors to generate.
        bright(bool):
            To use bright color or not.

    Returns:
        list:
            List of (r, g, b) colors.

    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / num_colors, 1, brightness) for i in range(num_colors)]
    colors = [colorsys.hsv_to_rgb(*c) for c in hsv]
    random.shuffle(colors)
    return colors


def colourise_image(img, cmap="viridis"):
    """If input img is single channel, colourise it.

    Args:
        img(ndarray):
            Single channel or RGB image as ndarray.
        cmap(str):
            Colormap to use, must be a valid matplotlib cmap string.

    Returns:
        img(ndarray): An RGB image.
    """
    if len(img.shape) == 2:
        # Single channel, make into rgb with colormap.
        c_map = cm.get_cmap(cmap)
        im_rgb = (c_map(img) * 255).astype(np.uint8)
        return im_rgb[:, :, :3]
    # Already rgb, return unaltered
    return img


def overlay_prediction_mask(
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
        img (ndarray):
            Input image to overlay the results on top of.
        prediction (ndarray):
            2D prediction map. Multi-class prediction should have values
            ranging from 0 to N-1, where N is the number of classes.
        label_info (dict):
            A dictionary containing the mapping for each integer value
            within `prediction` to its string and color. [int] : (str,
            (int, int, int)). By default, integer will be taken as label
            and color will be random.
        min_val (float):
            Only consider predictions greater than or equal to
            `min_val`. Otherwise, the original WSI in those regions will
            be displayed.
        alpha (float):
            Opacity value used for the overlay.
        ax (ax):
            Matplotlib ax object.
        return_ax (bool):
            Whether to return the matplotlib ax object. If not, then the
            overlay array will be returned.

    Returns:
        If return_ax is True, return the matplotlib ax object. Else,
        return the overlay array.

    """
    # Validate inputs
    if img.shape[:2] != prediction.shape[:2]:
        raise ValueError(
            f"Mismatch shape "
            f"`img` {img.shape[:2]} vs `prediction` {prediction.shape[:2]}."
        )
    if np.issubdtype(img.dtype, np.floating):
        if not (img.max() <= 1.0 and img.min() >= 0):
            raise ValueError("Not support float `img` outside [0, 1].")
        img = np.array(img * 255, dtype=np.uint8)
    # If `min_val` is defined, only display the overlay for areas with pred > min_val
    if min_val > 0:
        prediction_sel = prediction >= min_val

    overlay = img.copy()

    predicted_classes = sorted(np.unique(prediction).tolist())
    # Generate random colours if None are given
    rand_state = np.random.get_state()
    np.random.seed(123)
    label_info = label_info or {  # Use label_info if provided OR generate
        label_uid: (str(label_uid), np.random.randint(0, 255, 3))
        for label_uid in predicted_classes
    }
    np.random.set_state(rand_state)

    # Validate label_info
    missing_label_uids = _validate_label_info(label_info, predicted_classes)
    if len(missing_label_uids) != 0:
        raise ValueError(f"Missing label for: {missing_label_uids}.")

    rgb_prediction = np.zeros(
        [prediction.shape[0], prediction.shape[1], 3], dtype=np.uint8
    )
    for label_uid, (_, overlay_rgb) in label_info.items():
        sel = prediction == label_uid
        rgb_prediction[sel] = overlay_rgb

    # Add the overlay
    cv2.addWeighted(rgb_prediction, alpha, overlay, 1 - alpha, 0, overlay)
    overlay = overlay.astype(np.uint8)

    if min_val > 0.0:
        overlay[~prediction_sel] = img[~prediction_sel]

    if ax is None and not return_ax:
        return overlay

    # Create colorbar parameters
    name_list, color_list = zip(*label_info.values())  # Unzip values
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

    # Generate another ax, else using the provided
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(overlay)
    ax.axis("off")
    # Generate colour bar
    cbar = plt.colorbar(**colorbar_params)
    cbar.ax.set_yticklabels(name_list)
    cbar.ax.tick_params(labelsize=12)

    return ax


def _validate_label_info(
    label_info: Dict[int, Tuple[str, ArrayLike]], predicted_classes
) -> List[int]:
    """Validate the label_info dictionary.

    Args:
        label_info (dict):
            A dictionary containing the mapping for each integer value
            within `prediction` to its string and color. [int] : (str,
            (int, int, int)).
        predicted_classes (list):
            List of predicted classes.

    Raises:
        ValueError:
            If the label_info dictionary is not valid.

    Returns:
        list:
            List of missing label UIDs.

    """
    # May need better error messages
    check_uid_list = predicted_classes.copy()
    for label_uid, (label_name, label_colour) in label_info.items():
        if label_uid in check_uid_list:
            check_uid_list.remove(label_uid)
        if not isinstance(label_uid, int):
            raise ValueError(
                "Wrong `label_info` format: label_uid "
                f"{[label_uid, (label_name, label_colour)]}"
            )
        if not isinstance(label_name, str):
            raise ValueError(
                "Wrong `label_info` format: label_name "
                f"{[label_uid, (label_name, label_colour)]}"
            )
        if not isinstance(label_colour, (tuple, list, np.ndarray)):
            raise ValueError(
                "Wrong `label_info` format: label_colour "
                f"{[label_uid, (label_name, label_colour)]}"
            )
        if len(label_colour) != 3:
            raise ValueError(
                "Wrong `label_info` format: label_colour "
                f"{[label_uid, (label_name, label_colour)]}"
            )

    return check_uid_list


def overlay_probability_map(
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
        img (ndarray):
            Input image to overlay the results on top of. Assumed to be
            HW.
        prediction (ndarray):
            2D prediction map. Values are expected to be between 0-1.
        alpha (float):
            Opacity value used for the overlay.
        colour_map (string):
            The colour map to use for the heatmap. `jet` is used as the
            default.
        min_val (float):
            Only consider pixels that are greater than or equal to
            `min_val`. Otherwise, the original WSI in those regions will
            be displayed.
        alpha (float):
            Opacity value used for the overlay.
        ax (ax):
            Matplotlib axis object.
        return_ax (bool):
            Whether to return the matplotlib ax object. If not, then the
            overlay array will be returned.

    Returns:
        If return_ax is True, return the matplotlib ax object. Else,
        return the overlay array.

    """
    prediction = prediction.astype(np.float32)
    img = _validate_overlay_probability_map(img, prediction, min_val)
    prediction_sel = prediction >= min_val
    overlay = img.copy()

    cmap = plt.get_cmap(colour_map)
    prediction = np.squeeze(prediction.astype("float32"))
    # Take RGB from RGBA heat map
    rgb_prediction = (cmap(prediction)[..., :3] * 255).astype("uint8")

    # Add the overlay
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


def _validate_overlay_probability_map(img, prediction, min_val) -> np.ndarray:
    """Validate the input for the overlay_probability_map function.

    Args:
        img (:class:`numpy.ndarray`):
            Input image to overlay the results on top of. Assumed to be
            HW.
        prediction (:class:`numpy.ndarray`):
            2D prediction map. Values are expected to be between 0-1.
        min_val (float):
            Only consider pixels that are greater than or equal to
            `min_val`. Otherwise, the original WSI in those regions will
            be displayed.

    Raises:
        ValueError:
            If the input is not valid.

    Returns:
        :class:`numpy.ndarray`:
            Input image. May be modified if `min_val` has dtype float.

    """
    if prediction.ndim != 2:
        raise ValueError("The input prediction must be 2-dimensional of the form HW.")

    if img.shape[:2] != prediction.shape[:2]:
        raise ValueError(
            f"Mismatch shape `img` {img.shape[:2]}"
            f" vs `prediction` {prediction.shape[:2]}."
        )

    if prediction.max() > 1.0:
        raise ValueError("Not support float `prediction` outside [0, 1].")
    if prediction.min() < 0:
        raise ValueError("Not support float `prediction` outside [0, 1].")

    # if `min_val` is defined, only display the overlay for areas with prob > min_val
    if min_val < 0.0:
        raise ValueError(f"`min_val={min_val}` is not between [0, 1].")
    if min_val > 1.0:
        raise ValueError(f"`min_val={min_val}` is not between [0, 1].")

    if np.issubdtype(img.dtype, np.floating):
        if img.max() > 1.0:
            raise ValueError("Not support float `img` outside [0, 1].")
        if img.min() < 0:
            raise ValueError("Not support float `img` outside [0, 1].")
        return np.array(img * 255, dtype=np.uint8)
    return img


def overlay_prediction_contours(
    canvas: np.ndarray,
    inst_dict: dict,
    draw_dot: bool = False,
    type_colours: dict = None,
    inst_colours: Union[np.ndarray, Tuple[int]] = (255, 255, 0),
    line_thickness: int = 2,
):
    """Overlaying instance contours on image.

    Internally, colours from `type_colours` are prioritized over
    `inst_colours`. However, if `inst_colours` is `None` and
    `type_colours` is not provided, random colour is generated for each
    instance.

    Args:
        canvas (:class:`numpy.ndarray`):
            Image to draw predictions on.
        inst_dict (dict):
            Dictionary of instances. It is expected to be in the
            following format: `{instance_id: {type: int, contour:
            List[List[int]], centroid:List[float]}`.
        draw_dot (bool):
            To draw a dot for each centroid or not.
        type_colours (dict):
            A dict of {type_id : (type_name, colour)}, `type_id` is from
            0 to N and `colour` is a tuple of `(r, g, b)`.
        inst_colours (tuple, np.ndarray):
            A colour to assign for all instances, or a list of colours
            to assigned for each instance in `inst_dict`. By default,
            all instances will have RGB colour `(255, 255, 0).
        line_thickness:
            Line thickness of contours.

    Returns:
        :class:`numpy.ndarray`:
            The overlaid image.

    """
    overlay = np.copy((canvas))

    if inst_colours is None:
        inst_colours = random_colors(len(inst_dict))
        inst_colours = np.array(inst_colours) * 255
        inst_colours = inst_colours.astype(np.uint8)
    elif isinstance(inst_colours, tuple):
        inst_colours = np.array([inst_colours] * len(inst_dict))
    elif not isinstance(inst_colours, np.ndarray):
        raise ValueError(
            f"`inst_colours` must be np.ndarray or tuple: {type(inst_colours)}"
        )
    inst_colours = inst_colours.astype(np.uint8)

    for idx, [_, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colours is not None:
            inst_colour = type_colours[inst_info["type"]][1]
        else:
            inst_colour = (inst_colours[idx]).tolist()
        cv2.drawContours(
            overlay, [np.array(inst_contour)], -1, inst_colour, line_thickness
        )

        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple([int(v) for v in inst_centroid])
            overlay = cv2.circle(overlay, inst_centroid, 3, (255, 0, 0), -1)
    return overlay


def plot_graph(
    canvas: np.ndarray,
    nodes: np.ndarray,
    edges: np.ndarray,
    node_colors: Union[Tuple[int], np.ndarray] = (255, 0, 0),
    node_size: int = 5,
    edge_colors: Union[Tuple[int], np.ndarray] = (0, 0, 0),
    edge_size: int = 5,
):
    """Drawing a graph onto a canvas.

    Drawing a graph onto a canvas.

    Args:
        canvas (np.ndarray):
            Canvas to be drawn upon.
        nodes (np.ndarray):
            List of nodes, expected to be Nx2 where N is the number of
            nodes. Each node is expected to be of `(x, y)` and should be
            within the height and width of the canvas.
        edges (np.ndarray):
            List of edges, expected to be Mx2 where M is the number of
            edges. Each edge is defined as a pair of indexes `(from,
            to)`, where each corresponds to a node of within `nodes`.
        node_colors (tuple or np.ndarray):
            A color or list of node colors. Each color is expected to be
            `(r, g, b)` and is between 0 and 255.
        edge_colors (tuple or np.ndarray):
            A color or list of node colors. Each color is expected to be
            `(r, g, b)` and is between 0 and 255.
        node_size (int):
            Radius of each node.
        edge_size (int):
            Linewidth of the edge.

    """
    if isinstance(node_colors, tuple):
        node_colors = [node_colors] * len(nodes)
    if isinstance(edge_colors, tuple):
        edge_colors = [edge_colors] * len(edges)

    # draw the edges
    def to_int_tuple(x):
        """Helper to convert to tuple of int."""
        return tuple([int(v) for v in x])

    for idx, (src, dst) in enumerate(edges):
        src = to_int_tuple(nodes[src])
        dst = to_int_tuple(nodes[dst])
        color = to_int_tuple(edge_colors[idx])
        cv2.line(canvas, src, dst, color, thickness=edge_size)

    # draw the nodes
    for idx, node in enumerate(nodes):
        node = to_int_tuple(node)
        color = to_int_tuple(node_colors[idx])
        cv2.circle(canvas, node, node_size, color, thickness=-1)
    return canvas
