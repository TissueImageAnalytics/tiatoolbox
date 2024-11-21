"""Visualisation and overlay functions used in tiatoolbox."""

from __future__ import annotations

import colorsys
import random
from typing import TYPE_CHECKING, Callable, TypedDict, cast

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageFilter, ImageOps
from shapely.geometry import Polygon

from tiatoolbox import DuplicateFilter, logger
from tiatoolbox.enums import GeometryType

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from numpy.typing import ArrayLike

    from tiatoolbox.annotation import Annotation, AnnotationStore


class ColorbarParamsDict(TypedDict, total=False):
    """A subclass of TypedDict.

    Defines the types of the keyword arguments for 'colorbar_params'.

    """

    mappable: ScalarMappable
    boundaries: list[float]
    values: list[float]
    ticks: list[float]
    spacing: str
    orientation: str


def random_colors(num_colors: int, *, bright: bool) -> np.ndarray:
    """Generate a number of random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.

    Args:
        num_colors(int):
            Number of perceptively different colors to generate.
        bright(bool):
            To use bright color or not.

    Returns:
        np.ndarray:
            Array of (r, g, b) colors.

    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / num_colors, 1, brightness) for i in range(num_colors)]
    colors = [colorsys.hsv_to_rgb(*c) for c in hsv]
    random.shuffle(colors)
    return np.array(colors)


def colourise_image(img: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    """If input img is single channel, colourise it.

    Args:
        img(ndarray):
            Single channel or RGB image as ndarray.
        cmap(str):
            Colormap to use, must be a valid matplotlib cmap string.

    Returns:
        img(ndarray): An RGB image.
    """
    if len(img.shape) == 2:  # noqa: PLR2004
        # Single channel, make into rgb with colormap.
        c_map = colormaps[cmap]
        im_rgb = (c_map(img) * 255).astype(np.uint8)
        return im_rgb[:, :, :3]
    # Already rgb, return unaltered
    return img


def overlay_prediction_mask(
    img: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.35,
    label_info: dict | None = None,
    min_val: float = 0.0,
    ax: Axes | None = None,
    *,
    return_ax: bool,
) -> np.ndarray | Axes:
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
        msg = (
            f"Mismatch shape `img` {img.shape[:2]} "
            f"vs `prediction` {prediction.shape[:2]}."
        )
        raise ValueError(
            msg,
        )
    if np.issubdtype(img.dtype, np.floating):
        if not (img.max() <= 1.0 and img.min() >= 0):
            msg = "Not support float `img` outside [0, 1]."
            raise ValueError(msg)
        img = np.array(img * 255, dtype=np.uint8)
    # If `min_val` is defined, only display the overlay for areas with pred > min_val
    if min_val > 0:
        prediction_sel = prediction >= min_val

    overlay = img.copy()

    predicted_classes = sorted(np.unique(prediction).tolist())
    # Generate random colours if None are given
    rand_state = np.random.default_rng().__getstate__()
    rng = np.random.default_rng(123)
    label_info = label_info or {  # Use label_info if provided OR generate
        label_uid: (str(label_uid), rng.integers(0, 255, 3))
        for label_uid in predicted_classes
    }
    np.random.default_rng().__setstate__(rand_state)

    # Validate label_info
    missing_label_uids = _validate_label_info(label_info, predicted_classes)
    if len(missing_label_uids) != 0:
        msg = f"Missing label for: {missing_label_uids}."
        raise ValueError(msg)

    rgb_prediction = np.zeros(
        [prediction.shape[0], prediction.shape[1], 3],
        dtype=np.uint8,
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
    color_list_arr = np.array(color_list) / 255
    uid_list = list(label_info.keys())
    cmap = mpl.colors.ListedColormap(color_list_arr)

    colorbar_params: ColorbarParamsDict = {
        "mappable": mpl.cm.ScalarMappable(cmap=cmap),
        "boundaries": [*uid_list, uid_list[-1] + 1],
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
    label_info: dict[int, tuple[str, ArrayLike]],
    predicted_classes: list,
) -> list[int]:
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
            msg = (
                f"Wrong `label_info` format: label_uid "
                f"{[label_uid, (label_name, label_colour)]}"
            )
            raise TypeError(
                msg,
            )
        if not isinstance(label_name, str):
            msg = (
                f"Wrong `label_info` format: label_name "
                f"{[label_uid, (label_name, label_colour)]}"
            )
            raise TypeError(
                msg,
            )
        if not isinstance(label_colour, (tuple, list, np.ndarray)):
            msg = (
                f"Wrong `label_info` format: label_colour "
                f"{[label_uid, (label_name, label_colour)]}"
            )
            raise TypeError(
                msg,
            )
        if len(label_colour) != 3:  # noqa: PLR2004
            msg = (
                f"Wrong `label_info` format: label_colour "
                f"{[label_uid, (label_name, label_colour)]}"
            )
            raise ValueError(
                msg,
            )

    return check_uid_list


def overlay_probability_map(
    img: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.35,
    colour_map: str = "jet",
    min_val: float = 0.0,
    ax: Axes | None = None,
    *,
    return_ax: bool,
) -> np.ndarray | Axes:
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
    overlay[overlay > 255.0] = 255.0  # noqa: PLR2004
    overlay = overlay.astype(np.uint8)

    if min_val > 0.0:
        overlay[~prediction_sel] = img[~prediction_sel]

    if ax is None and not return_ax:
        return overlay

    colorbar_params: ColorbarParamsDict = {
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


def _validate_overlay_probability_map(
    img: np.ndarray,
    prediction: np.ndarray,
    min_val: float,
) -> np.ndarray:
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
    if prediction.ndim != 2:  # noqa: PLR2004
        msg = "The input prediction must be 2-dimensional of the form HW."
        raise ValueError(msg)

    if img.shape[:2] != prediction.shape[:2]:
        msg = (
            f"Mismatch shape `img` {img.shape[:2]} "
            f"vs `prediction` {prediction.shape[:2]}."
        )
        raise ValueError(
            msg,
        )

    if prediction.max() > 1.0:
        msg = "Not support float `prediction` outside [0, 1]."
        raise ValueError(msg)
    if prediction.min() < 0:
        msg = "Not support float `prediction` outside [0, 1]."
        raise ValueError(msg)

    # if `min_val` is defined, only display the overlay for areas with prob > min_val
    if min_val < 0.0:
        msg = f"`min_val={min_val}` is not between [0, 1]."
        raise ValueError(msg)
    if min_val > 1.0:
        msg = f"`min_val={min_val}` is not between [0, 1]."
        raise ValueError(msg)

    if np.issubdtype(img.dtype, np.floating):
        if img.max() > 1.0:
            msg = "Not support float `img` outside [0, 1]."
            raise ValueError(msg)
        if img.min() < 0:
            msg = "Not support float `img` outside [0, 1]."
            raise ValueError(msg)
        return np.array(img * 255, dtype=np.uint8)
    return img


def overlay_prediction_contours(
    canvas: np.ndarray,
    inst_dict: dict,
    type_colours: dict | None = None,
    inst_colours: np.ndarray | tuple[int, int, int] = (255, 255, 0),
    line_thickness: int = 2,
    *,
    draw_dot: bool,
) -> np.ndarray:
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
        inst_colours = random_colors(len(inst_dict), bright=True)

    if not isinstance(inst_colours, (tuple, np.ndarray)):
        msg = f"`inst_colours` must be np.ndarray or tuple: {type(inst_colours)}."
        raise TypeError(
            msg,
        )

    inst_colours_array = np.array(inst_colours) * 255

    if isinstance(inst_colours, tuple):
        inst_colours_array = np.array([inst_colours] * len(inst_dict))

    inst_colours_array = inst_colours_array.astype(np.uint8)

    for idx, [_, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colours is not None:
            inst_colour = type_colours[inst_info["type"]][1]
        else:
            inst_colour = (inst_colours_array[idx]).tolist()
        cv2.drawContours(
            overlay,
            [np.array(inst_contour)],
            -1,
            inst_colour,
            line_thickness,
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
    node_colors: tuple[int, int, int] | np.ndarray = (255, 0, 0),
    node_size: int = 5,
    edge_colors: tuple[int, int, int] | np.ndarray = (0, 0, 0),
    edge_size: int = 5,
) -> np.ndarray:
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
        node_colors_list = np.array([node_colors] * len(nodes))
    else:
        node_colors_list = node_colors.tolist()
    if isinstance(edge_colors, tuple):
        edge_colors_list = [edge_colors] * len(edges)
    else:
        edge_colors_list = edge_colors.tolist()

    # draw the edges
    def to_int_tuple(x: tuple[int, ...] | np.ndarray) -> tuple[int, ...]:
        """Helper to convert to tuple of int."""
        return tuple(int(v) for v in x)

    for idx, (src, dst) in enumerate(edges):
        src_ = to_int_tuple(nodes[src])
        dst_ = to_int_tuple(nodes[dst])
        color = to_int_tuple(edge_colors_list[idx])
        cv2.line(canvas, src_, dst_, color, thickness=edge_size)

    # draw the nodes
    for idx, node in enumerate(nodes):
        node_ = to_int_tuple(node)
        color = to_int_tuple(node_colors_list[idx])
        cv2.circle(canvas, node_, node_size, color, thickness=-1)
    return canvas


def _find_minimum_mpp_sf(mpp: tuple[float, float] | None) -> float:
    """Calculates minimum mpp scale factor."""
    if mpp is not None:
        return np.minimum(mpp[0] / 0.25, 1)
    return 1.0


class AnnotationRenderer:
    """Renders AnnotationStore to a tile.

    Renderer containing information and methods to render annotations
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
        secondary_cmap (dict [str, str, cmap]):
            a dictionary of the form {"type": some_type,
            "score_prop": a property name, "mapper": a matplotlib cmap object}.
            For annotations of the specified type, the given secondary colormap
            will override the primary colormap.
        blur_radius (int):
            radius of gaussian blur to apply to rendered annotations.
        score_prop_edge (str):
            A key that is present in the properties of annotations
            to be rendered that will be used to color rendered edges.
        function_mapper (Callable):
            A callable which will be given the properties of an annotation
            and should return a color for the annotation.  If this is specified,
            mapper and score_prop are ignored.


    """

    def __init__(  # noqa: PLR0913
        self: AnnotationRenderer,
        score_prop: str | None = None,
        mapper: str | dict | list | None = None,
        where: str | Callable | None = None,
        score_fn: Callable = lambda x: x,
        max_scale: int = 8,
        zoomed_out_strat: int | str = 10000,
        thickness: int = -1,
        edge_thickness: int = 1,
        secondary_cmap: dict | None = None,
        blur_radius: int = 0,
        score_prop_edge: str | None = None,
        function_mapper: Callable | None = None,
    ) -> None:
        """Initialize :class:`AnnotationRenderer`."""
        self.raw_mapper: str | dict | list | None = None
        self.mapper = self._set_mapper(value=mapper)
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
        self.function_mapper = function_mapper
        self.blur: ImageFilter.GaussianBlur | None
        if blur_radius > 0:
            self.blur = ImageFilter.GaussianBlur(blur_radius)
            self.edge_thickness = 0
        else:
            self.blur = None

    @staticmethod
    def to_tile_coords(
        coords: list,
        top_left: tuple[float, float],
        scale: float,
    ) -> list[np.ndarray]:
        """Return coords relative to top left of tile, as array suitable for cv2.

        Args:
            coords (List):
                List of coordinates in the form [x, y].
            top_left (tuple):
                The top left corner of the tile in wsi.
            scale (float):
                The zoom scale at which we are rendering.

        Returns:
            list:
                Array of coordinates in tile space in the form [x, y].

        """
        return [
            ((np.reshape(ring, (-1, 2)) - top_left) / scale).astype(np.int32)
            for ring in coords
        ]

    def get_color(
        self: AnnotationRenderer,
        annotation: Annotation,
        *,
        edge: bool,
    ) -> tuple[int, ...]:
        """Get the color for an annotation.

        Args:
            annotation (Annotation):
                Annotation to get color for.
            edge (bool):
                Whether to get the color for the edge of the annotation,
                or the interior.

        Returns:
            tuple:
                A color tuple (rgba).

        """
        score_prop = self.score_prop_edge if edge else self.score_prop

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
                            annotation.properties[self.secondary_cmap["score_prop"]],
                        ),
                    )
                )
            if self.function_mapper:
                return self.function_mapper(annotation.properties)
            if score_prop == "color":
                # use colors directly specified in annotation properties
                rgb = []
                for c in annotation.properties["color"]:  # type: ignore[union-attr]
                    c = cast(int, c)
                    rgb.append(int(255 * c))
                # rgb = [int(255 * c) for cast(int,c) in annotation.properties["color"]]
                return (*rgb, 255)
            if score_prop is not None:
                return tuple(
                    int(c * 255)
                    for c in self.mapper(
                        self.score_fn(annotation.properties[score_prop]),
                    )
                )
        except KeyError:
            logger.warning(
                "property: %s not found in properties. Using default color.",
                score_prop,
                stacklevel=2,
            )
        except TypeError:
            logger.warning(
                "property value type incompatable with cmap. Using default color.",
                stacklevel=2,
            )

        if edge:
            return 0, 0, 0, 255  # default to black for edge
        return 0, 255, 0, 255  # default color if no score_prop given

    def render_poly(
        self: AnnotationRenderer,
        tile: np.ndarray,
        annotation: Annotation,
        top_left: tuple[float, float],
        scale: float,
    ) -> None:
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
        col = self.get_color(annotation, edge=False)

        cnt = self.to_tile_coords(
            list(annotation.coords),
            top_left,
            scale,
        )
        if self.thickness > -1:
            cv2.polylines(
                tile,
                cnt,
                isClosed=True,
                color=col,
                thickness=self.edge_thickness,
                lineType=cv2.LINE_8,
            )
        else:
            cv2.fillPoly(tile, cnt, col)
        if self.thickness == -1 and self.edge_thickness > 0:
            edge_col = self.get_color(annotation, edge=True)
            cv2.polylines(
                tile,
                cnt,
                isClosed=True,
                color=edge_col,
                thickness=1,
                lineType=cv2.LINE_8,
            )

    def render_multipoly(
        self: AnnotationRenderer,
        tile: np.ndarray,
        annotation: Annotation,
        top_left: tuple[float, float],
        scale: float,
    ) -> None:
        """Render a multipolygon annotation onto a tile using cv2."""
        col = self.get_color(annotation, edge=False)
        geoms = annotation.coords
        for poly in geoms:
            cnt = self.to_tile_coords(list(poly), top_left, scale)
            cv2.fillPoly(tile, cnt, col)

    def render_pt(
        self: AnnotationRenderer,
        tile: np.ndarray,
        annotation: Annotation,
        top_left: tuple[float, float],
        scale: float,
    ) -> None:
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
        col = self.get_color(annotation, edge=False)
        cv2.circle(
            tile,
            self.to_tile_coords(
                list(annotation.coords),
                top_left,
                scale,
            )[0][0],
            np.maximum(self.edge_thickness, 1),
            col,
            thickness=self.thickness,
        )

    def render_line(
        self: AnnotationRenderer,
        tile: np.ndarray,
        annotation: Annotation,
        top_left: tuple[float, float],
        scale: float,
    ) -> None:
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
        col = self.get_color(annotation, edge=False)
        cnt = self.to_tile_coords(
            list(annotation.coords),
            top_left,
            scale,
        )
        cv2.polylines(
            tile,
            [np.array(cnt)],
            isClosed=False,
            color=col,
            thickness=3,
        )

    def _set_mapper(
        self: AnnotationRenderer,
        value: str | list | dict | None,
    ) -> Callable:
        """Set the mapper."""
        self.__dict__["mapper"] = value
        if value is None:
            self.raw_mapper = "jet"
            self.__dict__["mapper"] = colormaps["jet"]
        if isinstance(value, str) and value != "categorical":
            self.raw_mapper = value
            self.__dict__["mapper"] = colormaps[value]
        if isinstance(value, list):
            colors = random_colors(len(value), bright=True)
            self.__dict__["mapper"] = {
                key: (*color, 1) for key, color in zip(value, colors)
            }
        if isinstance(value, dict):
            self.raw_mapper = value
            self.__dict__["mapper"] = lambda x: value[x]

        return self.__dict__["mapper"]

    def __setattr__(
        self: AnnotationRenderer,
        __name: str,
        __value: str | list | dict | None,
    ) -> None:
        """Set attribute each time an attribute is set."""
        if __name == "mapper":
            # save a more readable version of the mapper too
            _ = self._set_mapper(__value)
            return
        if __name == "blur_radius" and isinstance(__value, int):
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
        self: AnnotationRenderer,
        store: AnnotationStore,
        bounds: tuple[float, float, float, float],
        scale: float,
        res: int = 1,
        border: int = 0,
    ) -> np.ndarray:
        """Render annotations within given bounds.

        This gets annotations as bounding boxes or geometries according to
        zoom level, and renders them. Large collections of small
        annotation geometries are decimated if appropriate.

        Args:
            store (AnnotationStore):
                The annotation store to render from.
            bounds (Polygon):
                The bounding box of the tile to render.
            scale (float):
                The scale at which we are rendering the tile.
            res (int):
                The resolution of the tile. Defaults to 1. Can be set to 2 for
                higher resolution rendering.
            border (int):
                The border to add around the tile. Defaults to 0. Used for blurred
                rendering to avoid edge effects.

        Returns:
            np.ndarray:
                The tile with the annotations rendered.

        """
        # Don't print out multiple warnings.
        duplicate_filter = DuplicateFilter()
        logger.addFilter(duplicate_filter)
        bound_geom = Polygon.from_bounds(*bounds)
        top_left = np.array(bounds[:2])
        output_size = [
            int((bounds[3] - bounds[1]) / scale),
            int((bounds[2] - bounds[0]) / scale),
        ]

        mpp_sf = _find_minimum_mpp_sf(self.info["mpp"])

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
                self.render_by_type(tile, ann, (top_left[0], top_left[1]), scale / res)

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
                    self.render_by_type(
                        tile, ann, (top_left[0], top_left[1]), scale / res
                    )
        else:
            # Get only annotations > min_area. Plot them all
            anns = store.query(
                bound_geom,
                self.where,
                min_area=min_area,
                geometry_predicate="bbox_intersects",
            )

            for ann in anns.values():
                self.render_by_type(tile, ann, (top_left[0], top_left[1]), scale / res)

        logger.removeFilter(duplicate_filter)
        if self.blur is None:
            return tile
        return np.array(
            ImageOps.crop(Image.fromarray(tile).filter(self.blur), border * res),
        )

    def render_by_type(
        self: AnnotationRenderer,
        tile: np.ndarray,
        annotation: Annotation,
        top_left: tuple[float, float],
        scale: float,
    ) -> None:
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
        geom_type = annotation.geometry_type
        if geom_type == GeometryType.POINT:
            self.render_pt(tile, annotation, top_left, scale)
        elif geom_type == GeometryType.POLYGON:
            self.render_poly(tile, annotation, top_left, scale)
        elif geom_type == GeometryType.LINE_STRING:
            self.render_line(tile, annotation, top_left, scale)
        else:
            logger.warning("Unknown geometry: %s", geom_type, stacklevel=3)
