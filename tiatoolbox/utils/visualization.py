"""Visualisation and overlay functions used in tiatoolbox."""
import colorsys
import random
import warnings
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy.typing import ArrayLike
from PIL import Image, ImageFilter, ImageOps
from shapely import speedups
from shapely.geometry import Polygon

from tiatoolbox.annotation.storage import Annotation, AnnotationStore

if speedups.available:  # pragma: no branch
    speedups.enable()


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
        "values": uid_list,
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
    cbar = plt.colorbar(**colorbar_params, ax=ax)
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
    cbar = plt.colorbar(**colorbar_params, ax=ax)
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
            all instances will have RGB colour `(255, 255, 0)`.
        line_thickness:
            Line thickness of contours.

    Returns:
        :class:`numpy.ndarray`:
            The overlaid image.

    """
    overlay = np.copy(canvas)

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
            inst_centroid = tuple(int(v) for v in inst_centroid)
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
            Line width of the edge.

    """
    if isinstance(node_colors, tuple):
        node_colors = [node_colors] * len(nodes)
    if isinstance(edge_colors, tuple):
        edge_colors = [edge_colors] * len(edges)

    # draw the edges
    def to_int_tuple(x):
        """Helper to convert to tuple of int."""
        return tuple(int(v) for v in x)

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


class AnnotationRenderer:
    """Renderer containing information and methods to render annotations
    from an AnnotationStore to a tile.

    Args:
        score_prop (str):
            A key that is present in the properties of annotations
            to be rendered that will be used to color rendered annotations.
        mapper (str, Dict or List):
            A dictionary or colormap used to color annotations according
            to the value of properties[score_prop] of an annotation.  Should
            be either a matplotlib colormap, a string which is a name of a
            matplotlib colormap, a dict of possible property {value: color}
            pairs, or a list of categorical property values (in which case a
            dict will be created with a random color generated for each
            category)
        where (str or Callable):
            a callable or predicate which will be passed on to
            AnnotationStore.query() when fetching annotations to be rendered
            (see AnnotationStore for more details)
        score_fn (Callable):
            an optional callable which will be called on the value of
            the property that will be used to generate the color before giving
            it to colormap. Use it for example to normalise property
            values if they do not fall into the range [0,1], as matplotlib
            colormap expects values in this range. i.e roughly speaking
            annotation_color=mapper(score_fn(ann.properties[score_prop]))
        max_scale (int):
            downsample level above which Polygon geometries on crowded
            tiles will be rendered as a bounding box instead
        zoomed_out_strat (int, str):
            strategy to use when rendering zoomed out tiles at
            a level above max_scale.  Can be one of 'decimate', 'scale', or a number
            which defines the minimum area an abject has to cover to be rendered
            while zoomed out above max_scale.
        thickness (int):
            line thickness of rendered contours. -1 will render filled
            contours.
        edge_thickness (int):
            line thickness of rendered edges.
        secondary_cmap (dict [str, str, cmap])):
            a dictionary of the form {"type": some_type,
            "score_prop": a property name, "mapper": a matplotlib cmap object}.
            For annotations of the specified type, the given secondary colormap
            will override the primary colormap.
        blur_radius (int):
            radius of gaussian blur to apply to rendered annotations.

    """

    def __init__(
        self,
        score_prop=None,
        mapper=None,
        where=None,
        score_fn=lambda x: x,
        max_scale=8,
        zoomed_out_strat=10000,
        thickness=-1,
        edge_thickness=1,
        secondary_cmap=None,
        blur_radius=0,
        score_prop_edge=None,
    ):
        if mapper is None:
            mapper = cm.get_cmap("jet")
        if isinstance(mapper, str) and mapper != "categorical":
            mapper = cm.get_cmap(mapper)
        if isinstance(mapper, list):
            colors = random_colors(len(mapper))
            mapper = {key: (*color, 1) for key, color in zip(mapper, colors)}
        if isinstance(mapper, dict):
            self.mapper = lambda x: mapper[x]
        else:
            self.mapper = mapper
        self.score_prop = score_prop
        self.score_prop_edge = score_prop_edge
        self.where = where
        self.score_fn = score_fn
        self.max_scale = max_scale
        self.info = {"mpp": None}
        self.thickness = thickness
        self.edge_thickness = edge_thickness
        self.zoomed_out_strat = zoomed_out_strat
        self.secondary_cmap = secondary_cmap
        self.blur_radius = blur_radius
        if blur_radius > 0:
            self.blur = ImageFilter.GaussianBlur(blur_radius)
            self.edge_thickness = 0
        else:
            self.blur = None

    @staticmethod
    def to_tile_coords(coords: List, top_left: Tuple[float, float], scale: float):
        """Return coords relative to top left of tile, as array suitable for cv2.
        Args:
            coords (List):
                List of coordinates in the form [x, y].
            top_left (tuple):
                The top left corner of the tile in wsi.
            scale (float):
                The zoom scale at which we are rendering.
        Returns:
            np.array:
                Array of coordinates in tile space in the form [x, y].

        """
        return np.squeeze(((np.array(coords) - top_left) / scale).astype(np.int32))

    def get_color(self, annotation: Annotation, edge=False):
        """Get the color for an annotation.
        Args:
            annotation (Annotation):
                Annotation to get color for.
        Returns:
            tuple:
                A color tuple (rgba).

        """
        if edge:
            score_prop = self.score_prop_edge
        else:
            score_prop = self.score_prop
        try:
            if (
                self.secondary_cmap is not None
                and "type" in annotation.properties
                and annotation.properties["type"] == self.secondary_cmap["type"]
            ):
                # use secondary colormap to color annotations of specific type
                return tuple(
                    int(c * 255)
                    for c in self.secondary_cmap["mapper"](
                        self.score_fn(
                            annotation.properties[self.secondary_cmap["score_prop"]]
                        )
                    )
                )
            if score_prop == "color":
                # use colors directly specified in annotation properties
                return (*[int(255 * c) for c in annotation.properties["color"]], 255)
            if score_prop is not None:
                return tuple(
                    int(c * 255)
                    for c in self.mapper(
                        self.score_fn(annotation.properties[score_prop])
                    )
                )
        except KeyError:
            warnings.warn("score_prop not found in properties. Using default color.")
        if edge:
            return (0, 0, 0, 255)  # default to black for edge
        return 0, 255, 0, 255  # default color if no score_prop given

    def render_poly(
        self,
        tile: np.ndarray,
        annotation: Annotation,
        top_left: Tuple[float, float],
        scale: float,
    ):
        """Render a polygon annotation onto a tile using cv2.
        Args:
            tile (ndarray):
                The rgb(a) tile image to render onto.
            annotation (Annotation):
                The annotation to render.
            top_left (tuple):
                The top left corner of the tile in wsi.
            scale (float):
                The zoom scale at which we are rendering.

        """
        col = self.get_color(annotation)

        cnt = self.to_tile_coords(annotation.geometry.exterior.coords, top_left, scale)
        if self.thickness > -1:
            cv2.drawContours(
                tile, [cnt], 0, col, self.edge_thickness, lineType=cv2.LINE_8
            )
        else:
            cv2.drawContours(tile, [cnt], 0, col, self.thickness, lineType=cv2.LINE_8)
        if self.thickness == -1 and self.edge_thickness > 0:
            edge_col = self.get_color(annotation, True)
            cv2.drawContours(tile, [cnt], 0, edge_col, 1, lineType=cv2.LINE_8)

    def render_multipoly(self, tile, annotation, top_left, scale):
        """render a multipolygon annotation onto a tile using cv2"""
        col = self.get_color(annotation)

        for poly in annotation.geometry.geoms:
            cnt = self.to_tile_coords(poly.exterior.coords, top_left, scale)
            cv2.drawContours(tile, [cnt], 0, col, self.thickness, lineType=cv2.LINE_8)

    def render_pt(
        self,
        tile: np.ndarray,
        annotation: Annotation,
        top_left: Tuple[float, float],
        scale: float,
    ):
        """Render a point annotation onto a tile using cv2.
        Args:
            tile (ndarray):
                The rgb(a) tile image to render onto.
            annotation (Annotation):
                The annotation to render.
            top_left (tuple):
                The top left corner of the tile in wsi.
            scale (float):
                The zoom scale at which we are rendering.

        """
        col = self.get_color(annotation)
        cv2.circle(
            tile,
            self.to_tile_coords(list(annotation.geometry.coords), top_left, scale),
            np.maximum(self.edge_thickness, 1),
            col,
            thickness=self.thickness,
        )

    def render_line(
        self,
        tile: np.ndarray,
        annotation: Annotation,
        top_left: Tuple[float, float],
        scale: float,
    ):
        """Render a line annotation onto a tile using cv2.
        Args:
            tile (ndarray):
                The rgb(a) tile image to render onto.
            annotation (Annotation):
                The annotation to render.
            top_left (tuple):
                The top left corner of the tile in wsi.
            scale (float):
                The zoom scale at which we are rendering.

        """
        col = self.get_color(annotation)
        cv2.polylines(
            tile,
            [self.to_tile_coords(list(annotation.geometry.coords), top_left, scale)],
            False,
            col,
            thickness=3,
        )

    def __setattr__(self, __name: str, __value) -> None:
        if __name == "blur_radius":
            # need to change additional settings
            if __value > 0:
                self.__dict__["blur"] = ImageFilter.GaussianBlur(__value)
                self.__dict__["edge_thickness"] = 0
            else:
                self.__dict__["blur"] = None
                self.__dict__["edge_thickness"] = self.__dict__["edge_thickness_old"]
        elif __name == "edge_thickness":
            self.__dict__["edge_thickness_old"] = __value

        self.__dict__[__name] = __value

    def render_annotations(
        self,
        store: AnnotationStore,
        bounds: Tuple[float, float, float, float],
        scale: float,
        res: int = 1,
        border: int = 0,
    ):
        """Render annotations within given bounds.

        This gets annotations as bounding boxes or geometries according to
        zoom level, and renders them. Large collections of small
        annotation geometries are decimated if appropriate.

        Args:
            rgb (np.ndarray):
                The image to render the annotation on.
            bound_geom (Polygon):
                A polygon representing the bounding box of the tile.
            scale (float):
                The scale at which we are rendering the tile.
        Returns:
            np.ndarray:
                The tile with the annotations rendered.

        """
        bound_geom = Polygon.from_bounds(*bounds)
        top_left = np.array(bounds[:2])
        output_size = [
            int((bounds[3] - bounds[1]) / scale),
            int((bounds[2] - bounds[0]) / scale),
        ]

        mpp_sf = (
            np.minimum(self.info["mpp"][0] / 0.25, 1)
            if self.info["mpp"] is not None
            else 1
        )
        min_area = 0.0005 * (output_size[0] * output_size[1]) * (scale * mpp_sf) ** 2

        tile = np.zeros((output_size[0] * res, output_size[1] * res, 4), dtype=np.uint8)

        if scale <= self.max_scale:
            # get all annotations
            anns = store.query(
                bound_geom,
                self.where,
                geometry_predicate="bbox_intersects",
            )

            for ann in anns.values():
                self.render_by_type(tile, ann, top_left, scale / res)

        elif self.zoomed_out_strat == "decimate":
            # do decimation on small annotations
            decimate = int(scale / self.max_scale) + 1

            bounding_boxes = store.bquery(
                bound_geom,
                self.where,
            )

            for i, (key, box) in enumerate(bounding_boxes.items()):
                area = (box[0] - box[2]) * (box[1] - box[3])
                if area > min_area or i % decimate == 0:
                    ann = store[key]
                    self.render_by_type(tile, ann, top_left, scale / res)
        else:
            # Get only annotations > min_area. Plot them all
            anns = store.query(
                bound_geom,
                self.where,
                min_area=min_area,
                geometry_predicate="bbox_intersects",
            )

            for ann in anns.values():
                self.render_by_type(tile, ann, top_left, scale / res)

        if self.blur is None:
            return tile
        return np.array(
            ImageOps.crop(Image.fromarray(tile).filter(self.blur), border * res)
        )

    def render_by_type(
        self,
        tile: np.ndarray,
        annotation: Annotation,
        top_left: Tuple[float, float],
        scale: float,
    ):
        """Render annotation appropriately to its geometry type.

        Args:
            tile (np.ndarray):
                The rgb(a) tile image to render the annotation on.
            annotation (Annotation):
                The annotation to render.
            top_left (Tuple[int, int]):
                The top left coordinate of the tile.
            scale (float):
                The scale at which we are rendering the tile.

        """
        geom_type = annotation.geometry.geom_type
        if geom_type == "Point":
            self.render_pt(tile, annotation, top_left, scale)
        elif geom_type == "Polygon":
            self.render_poly(tile, annotation, top_left, scale)
        elif geom_type == "MultiPolygon":
            self.render_multipoly(tile, annotation, top_left, scale)
        elif geom_type == "LineString":
            self.render_line(tile, annotation, top_left, scale)
        else:
            warnings.warn(f"Unknown geometry: {geom_type}")
