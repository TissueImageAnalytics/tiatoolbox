"""Test for visualization."""

import copy
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from tiatoolbox.annotation.storage import Annotation
from tiatoolbox.enums import GeometryType
from tiatoolbox.utils.visualization import (
    overlay_prediction_contours,
    overlay_prediction_mask,
    overlay_probability_map,
    plot_graph,
)
from tiatoolbox.wsicore.wsireader import WSIReader


def test_overlay_prediction_mask(sample_wsi_dict: dict) -> None:
    """Test for overlaying merged patch prediction of wsi."""
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    mini_wsi_pred = Path(sample_wsi_dict["wsi2_4k_4k_pred"])
    reader = WSIReader.open(mini_wsi_svs)

    raw, merged = joblib.load(mini_wsi_pred)

    thumb = reader.slide_thumbnail(resolution=2.77, units="mpp")
    with pytest.raises(ValueError, match=r".*Mismatch shape.*"):
        _ = overlay_prediction_mask(thumb, merged, return_ax=True)

    label_info_full = {
        0: ("BACKGROUND", (0, 0, 0)),
        1: ("01_TUMOR", (255, 0, 0)),
        2: ("02_STROMA", (0, 255, 0)),
        3: ("03_COMPLEX", (0, 0, 255)),
        4: ("04_LYMPHO", (0, 255, 255)),
        5: ("05_DEBRIS", (255, 0, 255)),
        6: ("06_MUCOSA", (255, 255, 0)),
        7: ("07_ADIPOSE", (125, 255, 255)),
        8: ("08_EMPTY", (255, 125, 255)),
    }

    thumb = reader.slide_thumbnail(resolution=raw["resolution"], units=raw["units"])
    with pytest.raises(ValueError, match=r".*float `img` outside.*"):
        _ = overlay_prediction_mask(thumb.astype(np.float32), merged, return_ax=True)

    label_info_fail = copy.deepcopy(label_info_full)
    del label_info_fail[1]
    with pytest.raises(ValueError, match=r".*Missing label.*"):
        _ = overlay_prediction_mask(
            thumb,
            merged,
            label_info=label_info_fail,
            return_ax=True,
        )

    label_info_fail = copy.deepcopy(label_info_full)
    label_info_fail[1] = (1, (255, 255, 255))
    with pytest.raises(TypeError, match=r".*Wrong `label_info` format.*"):
        _ = overlay_prediction_mask(
            thumb,
            merged,
            label_info=label_info_fail,
            return_ax=True,
        )

    label_info_fail = copy.deepcopy(label_info_full)
    label_info_fail["ABC"] = ("ABC", (255, 255, 255))
    with pytest.raises(TypeError, match=r".*Wrong `label_info` format.*"):
        _ = overlay_prediction_mask(
            thumb,
            merged,
            label_info=label_info_fail,
            return_ax=True,
        )

    label_info_fail = copy.deepcopy(label_info_full)
    label_info_fail[1] = ("ABC", "ABC")
    with pytest.raises(TypeError, match=r".*Wrong `label_info` format.*"):
        _ = overlay_prediction_mask(
            thumb,
            merged,
            label_info=label_info_fail,
            return_ax=True,
        )

    label_info_fail = copy.deepcopy(label_info_full)
    label_info_fail[1] = ("ABC", (255, 255))
    with pytest.raises(ValueError, match=r".*Wrong `label_info` format.*"):
        _ = overlay_prediction_mask(
            thumb,
            merged,
            label_info=label_info_fail,
            return_ax=True,
        )

    # Test normal run, should not crash.
    thumb_float = thumb / 255.0
    ax = overlay_prediction_mask(
        thumb_float,
        merged,
        label_info=label_info_full,
        return_ax=True,
    )
    ax.remove()
    ax = overlay_prediction_mask(
        thumb,
        merged,
        label_info=label_info_full,
        return_ax=True,
    )
    ax.remove()

    ax = plt.subplot(1, 2, 1)
    _ = overlay_prediction_mask(thumb, merged, ax=ax, return_ax=True)
    _ = overlay_prediction_mask(thumb_float, merged, min_val=0.5, return_ax=False)


def test_overlay_probability_map(sample_wsi_dict: dict) -> None:
    """Test functional run for overlaying merged patch prediction of wsi."""
    mini_wsi_svs = Path(sample_wsi_dict["wsi2_4k_4k_svs"])
    reader = WSIReader.open(mini_wsi_svs)

    thumb = reader.slide_thumbnail(resolution=2.77, units="mpp")

    # * Test normal run, should not crash.
    thumb_float = np.mean(thumb, axis=-1) / 255.0
    output = overlay_probability_map(thumb, thumb_float, min_val=0.5, return_ax=True)
    output = overlay_probability_map(
        thumb / 256.0,
        thumb_float,
        min_val=0.5,
        return_ax=True,
    )
    output = overlay_probability_map(thumb, thumb_float, return_ax=False)
    assert isinstance(output, np.ndarray)
    output = overlay_probability_map(thumb, thumb_float, return_ax=True)
    assert isinstance(output, mpl.axes.Axes)
    output = overlay_probability_map(thumb, thumb_float, ax=output, return_ax=True)
    assert isinstance(output, mpl.axes.Axes)

    # * Test crash mode
    with pytest.raises(ValueError, match=r".*min_val.*0, 1*"):
        overlay_probability_map(thumb, thumb_float, min_val=-0.5, return_ax=True)
    with pytest.raises(ValueError, match=r".*min_val.*0, 1*"):
        overlay_probability_map(thumb, thumb_float, min_val=1.5, return_ax=True)
    with pytest.raises(ValueError, match=r".*float `img`.*0, 1*"):
        overlay_probability_map(
            np.full_like(thumb, 1.5, dtype=float),
            thumb_float,
            return_ax=True,
        )
    with pytest.raises(ValueError, match=r".*float `img`.*0, 1*"):
        overlay_probability_map(
            np.full_like(thumb, -0.5, dtype=float),
            thumb_float,
            return_ax=True,
        )
    with pytest.raises(ValueError, match=r".*prediction.*0, 1*"):
        overlay_probability_map(thumb, thumb_float + 1.05, thumb_float, return_ax=True)
    with pytest.raises(ValueError, match=r".*prediction.*0, 1*"):
        overlay_probability_map(thumb, thumb_float - 1.05, thumb_float, return_ax=True)
    with pytest.raises(ValueError, match=r".*Mismatch shape*"):
        overlay_probability_map(np.zeros([2, 2, 3]), thumb_float, return_ax=True)
    with pytest.raises(ValueError, match=r".*2-dimensional*"):
        overlay_probability_map(thumb, thumb_float[..., None], return_ax=True)


def test_overlay_instance_prediction() -> None:
    """Test for overlaying instance predictions on canvas."""
    inst_map = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 2, 2, 0],
            [0, 0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    # dummy instance dict
    type_colours = {
        0: ("A", (1, 0, 1)),
        1: ("B", (2, 0, 2)),
    }
    inst_dict = {
        0: {
            "centroid": [1, 1],
            "type": 0,
            "contour": [[1, 1], [1, 2], [2, 2], [2, 1]],
        },
        1: {
            "centroid": [3, 3],
            "type": 1,
            "contour": [[3, 3], [3, 4], [4, 4], [4, 3]],
        },
    }
    canvas = np.zeros((*inst_map.shape, 3), dtype=np.uint8)
    canvas = overlay_prediction_contours(
        canvas,
        inst_dict,
        draw_dot=False,
        type_colours=type_colours,
        line_thickness=1,
    )
    ref_value = -12
    assert np.sum(canvas[..., 0].astype(np.int32) - inst_map) == 0
    assert np.sum(canvas[..., 1].astype(np.int32) - inst_map) == ref_value
    assert np.sum(canvas[..., 2].astype(np.int32) - inst_map) == 0
    canvas = overlay_prediction_contours(
        canvas,
        inst_dict,
        draw_dot=True,
        type_colours=None,
        line_thickness=1,
    )

    # test run with randomized colours
    canvas = overlay_prediction_contours(
        canvas,
        inst_dict,
        inst_colours=None,
        draw_dot=False,
    )
    # test run with custom colour
    canvas = overlay_prediction_contours(
        canvas,
        inst_dict,
        inst_colours=(0, 0, 1),
        draw_dot=False,
    )
    # test run with custom colour for each instance
    inst_colours = [[0, 155, 155] for v in range(len(inst_dict))]
    canvas = overlay_prediction_contours(
        canvas,
        inst_dict,
        inst_colours=np.array(inst_colours),
        draw_dot=False,
    )
    # test crash
    with pytest.raises(TypeError, match=r"`.*inst_colours`.*tuple.*"):
        overlay_prediction_contours(
            canvas,
            inst_dict,
            inst_colours=inst_colours,
            draw_dot=False,
        )


def test_plot_graph() -> None:
    """Test plotting graph."""
    canvas = np.zeros([10, 10])
    nodes = np.array([[1, 1], [2, 2], [2, 5]])
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    node_colors = np.array([[0, 0, 0]] * 3)
    edge_colors = np.array([[1, 1, 1]] * 3)
    plot_graph(
        canvas,
        nodes,
        edges,
    )
    plot_graph(canvas, nodes, edges, node_colors=node_colors, edge_colors=edge_colors)


def test_decode_wkb() -> None:
    """Test decoding of WKB geometries."""
    # Create some Shapely geometries of supported types
    point = Point(0, 0)
    line = LineString([(0, 0), (1, 1), (2, 0)])
    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

    # Convert the geometries to WKB format
    point_wkb = point.wkb
    line_wkb = line.wkb
    polygon_wkb = polygon.wkb

    # Decode the WKB geometries
    point_contours = Annotation.decode_wkb(
        point_wkb,
        GeometryType.POINT,
    )
    line_contours = Annotation.decode_wkb(
        line_wkb,
        GeometryType.LINE_STRING,
    )
    polygon_contours = Annotation.decode_wkb(
        polygon_wkb,
        GeometryType.POLYGON,
    )

    # Check that the decoded contours are as expected
    assert np.all(point_contours == np.array([[0, 0]]))
    assert np.all(line_contours == np.array([[0, 0], [1, 1], [2, 0]]))
    assert np.all(
        polygon_contours == np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]),
    )

    # do the same for multi-point, multi-line and multi-polygon
    multipoint = MultiPoint([(0, 0), (1, 1), (2, 0)])
    multiline = MultiLineString([((0, 0), (1, 1), (2, 0)), ((0, 0), (1, 1), (2, 0))])
    multipolygon = MultiPolygon(
        [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
        ],
    )

    multipoint_wkb = multipoint.wkb
    multiline_wkb = multiline.wkb
    multipolygon_wkb = multipolygon.wkb

    multipoint_contours = Annotation.decode_wkb(multipoint_wkb, 4)
    multiline_contours = Annotation.decode_wkb(multiline_wkb, 5)
    multipolygon_contours = Annotation.decode_wkb(multipolygon_wkb, 6)

    assert np.all(multipoint_contours == np.array([[[0, 0]], [[1, 1]], [[2, 0]]]))
    assert np.all(
        multiline_contours
        == np.array([[[0, 0], [1, 1], [2, 0]], [[0, 0], [1, 1], [2, 0]]]),
    )
    assert np.all(
        multipolygon_contours
        == np.array(
            [
                [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]],
                [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]],
            ],
        ),
    )

    # test unknown geometry type
    with pytest.raises(ValueError, match=r"Unknown geometry type"):
        Annotation.decode_wkb(multipolygon_wkb, 7)
