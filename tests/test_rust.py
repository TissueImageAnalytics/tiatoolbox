"""Test for rust functionality."""

import json
import tempfile
from pathlib import Path
from typing import cast

import cv2
import dask.array as da
import numpy as np
import pytest

from tiatoolbox import rmisc, rmultitask, utils
from tiatoolbox.type_hints import JSON


def test_add() -> None:
    """Temp test to test function add."""
    assert rmultitask.add(5, 4) == 9


def test_contrast_enhancer() -> None:
    """Test contrast enhancement functionality."""
    input_array = np.array(
        [
            [
                [37, 244, 193, 106, 235, 128, 71, 140, 47],
                [103, 184, 72, 20, 188, 238, 126, 7, 0],
                [137, 195, 204, 32, 203, 170, 101, 77, 133],
            ]
        ],
        dtype=np.uint8,
    )

    # expected output of the contrast_enhancer
    result_array = np.array(
        [
            [
                [35, 255, 203, 110, 248, 133, 72, 146, 46],
                [106, 193, 73, 17, 198, 251, 131, 3, 0],
                [143, 205, 215, 30, 214, 178, 104, 78, 139],
            ]
        ],
        dtype=np.uint8,
    )

    # Calculating the contrast enhanced version of input_array
    output_array = utils.misc.contrast_enhancer(input_array, low_p=2, high_p=98)
    # The out_put array should be equal to expected result_array
    assert np.all(result_array == output_array)
    input_array = np.array(
        [
            [
                [0],
                [0],
                [0],
            ]
        ],
        dtype=np.uint8,
    )

    # expected output of the contrast_enhancer
    result_array = np.array(
        [
            [
                [0],
                [0],
                [0],
            ]
        ],
        dtype=np.uint8,
    )

    # Calculating the contrast enhanced version of input_array
    output_array = utils.misc.contrast_enhancer(input_array, low_p=2, high_p=98)
    # The out_put array should be equal to expected result_array
    assert np.all(result_array == output_array)


def test_patch_predictions_as_qupath_json() -> None:
    """Tests that the rust code for patch_predictions_as_qupath_json works correctly."""
    class_colours = {
        0.0: [255, 0, 0],
        1.0: [0, 255, 0],
    }

    class_dict = {
        0.0: "Tumour",
        1.0: "Normal",
    }

    preds = [0.0, 1.0]

    patch_coords = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0],
        ],
        dtype=np.float64,
    )

    result = rmisc.patch_predictions_as_qupath_json(
        class_colours,
        preds,
        class_dict,
        patch_coords,
    )

    expected = [
        {
            "type": "Feature",
            "id": "patch_0",
            "geometry": {
                "type": "Polygon",
                "coordinates": (
                    (
                        (10.0, 20.0),
                        (10.0, 40.0),
                        (30.0, 40.0),
                        (30.0, 20.0),
                        (10.0, 20.0),
                    ),
                ),
            },
            "properties": {
                "classification": {
                    "name": "Tumour",
                    "color": [255, 0, 0],
                }
            },
            "objectType": "annotation",
            "name": "Tumour",
            "class_value": 0.0,
        },
        {
            "type": "Feature",
            "id": "patch_1",
            "geometry": {
                "type": "Polygon",
                "coordinates": (
                    (
                        (50.0, 60.0),
                        (50.0, 80.0),
                        (70.0, 80.0),
                        (70.0, 60.0),
                        (50.0, 60.0),
                    ),
                ),
            },
            "properties": {
                "classification": {
                    "name": "Normal",
                    "color": [0, 255, 0],
                }
            },
            "objectType": "annotation",
            "name": "Normal",
            "class_value": 1.0,
        },
    ]

    assert result == expected


class DummyPolygon:
    """Minimal polygon class used to test the method from bounds."""

    @classmethod
    def from_bounds(
        cls,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> tuple[float, float, float, float]:
        """Return polygon bounds as a tuple."""
        return (xmin, ymin, xmax, ymax)


class DummyAnnotation:
    """Minimal annotation class storing polygon geometry and properties."""

    def __init__(
        self,
        polygon: tuple[float, float, float, float],
        properties: dict[str, str | float],
    ) -> None:
        """Initialize an annotation with geometry and associated properties."""
        self.polygon: tuple[float, float, float, float] = polygon
        self.properties: dict[str, str | float] = properties


def test_patch_predictions_as_annotations() -> None:
    """Tests that the rust code for patch_predictions_as_annotations works correctly."""
    preds = [0.0, 1.0]

    class_dict = {
        0.0: "Tumour",
        1.0: "Normal",
    }

    class_probs = np.array(
        [
            [0.8, 0.2],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )

    patch_coords = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0],
        ],
        dtype=np.float64,
    )

    keys_contains_labels = True
    keys_contains_probabilities = True

    annotations = rmisc.patch_predictions_as_annotations(
        DummyAnnotation,
        DummyPolygon,
        preds,
        keys_contains_labels,
        keys_contains_probabilities,
        class_dict,
        class_probs,
        patch_coords,
        [0.0, 1.0],  # classes_predicted
        [1.0, 0.0],  # labels
    )

    assert len(annotations) == 2

    assert annotations[0].polygon == (10.0, 20.0, 30.0, 40.0)
    assert annotations[0].properties == {
        "prob_Tumour": 0.8,
        "prob_Normal": 0.2,
        "label": "Normal",
        "type": "Tumour",
    }

    assert annotations[1].polygon == (50.0, 60.0, 70.0, 80.0)
    assert annotations[1].properties == {
        "prob_Tumour": 0.1,
        "prob_Normal": 0.9,
        "label": "Tumour",
        "type": "Normal",
    }
    class_dict = {
        0.0: 1.2,
        1.0: "Normal",
    }

    annotations = rmisc.patch_predictions_as_annotations(
        DummyAnnotation,
        DummyPolygon,
        preds,
        keys_contains_labels,
        keys_contains_probabilities,
        class_dict,
        class_probs,
        patch_coords,
        [0.0, 1.0],  # classes_predicted
        [1.0, 0.0],  # labels
    )

    assert len(annotations) == 2

    assert annotations[0].polygon == (10.0, 20.0, 30.0, 40.0)
    assert annotations[0].properties == {
        "prob_1.2": 0.8,
        "prob_Normal": 0.2,
        "label": "Normal",
        "type": 1.2,
    }

    assert annotations[1].polygon == (50.0, 60.0, 70.0, 80.0)
    assert annotations[1].properties == {
        "prob_1.2": 0.1,
        "prob_Normal": 0.9,
        "label": 1.2,
        "type": "Normal",
    }

    keys_contains_labels = False
    keys_contains_probabilities = False
    class_dict = {
        0.0: "Tumour",
        1.0: "Normal",
    }
    annotations = rmisc.patch_predictions_as_annotations(
        DummyAnnotation,
        DummyPolygon,
        preds,
        keys_contains_labels,
        keys_contains_probabilities,
        class_dict,
        class_probs,
        patch_coords,
        [0.0, 1.0],  # classes_predicted
        [1.0, 0.0],  # labels
    )

    assert len(annotations) == 2

    assert annotations[0].polygon == (10.0, 20.0, 30.0, 40.0)
    assert annotations[0].properties == {
        "type": "Tumour",
    }

    assert annotations[1].polygon == (50.0, 60.0, 70.0, 80.0)
    assert annotations[1].properties == {
        "type": "Normal",
    }
    annotations = rmisc.patch_predictions_as_annotations(
        DummyAnnotation,
        DummyPolygon,
        [],
        keys_contains_labels,
        keys_contains_probabilities,
        class_dict,
        class_probs,
        patch_coords,
        [0.0, 1.0],  # classes_predicted
        [1.0, 0.0],  # labels
    )

    assert len(annotations) == 2

    assert annotations[0].polygon == (10.0, 20.0, 30.0, 40.0)
    assert annotations[0].properties == {}

    assert annotations[1].polygon == (50.0, 60.0, 70.0, 80.0)
    assert annotations[1].properties == {}
    with pytest.raises((TypeError, AttributeError)):
        annotations = rmisc.patch_predictions_as_annotations(
            DummyPolygon,
            DummyPolygon,
            [],
            keys_contains_labels,
            keys_contains_probabilities,
            class_dict,
            class_probs,
            patch_coords,
            [0.0, 1.0],  # classes_predicted
            [1.0, 0.0],  # labels
        )
    with pytest.raises((TypeError, AttributeError)):
        annotations = rmisc.patch_predictions_as_annotations(
            DummyAnnotation,
            DummyAnnotation,
            [],
            keys_contains_labels,
            keys_contains_probabilities,
            class_dict,
            class_probs,
            patch_coords,
            [0.0, 1.0],  # classes_predicted
            [1.0, 0.0],  # labels
        )


def test_json_dump_python_object() -> None:
    """Tests whether json_dump_python_object works."""
    obj = {
        "name": "Alice",
        "age": 30,
        "active": True,
        "numbers": [1, 2, 3],
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        path = Path(tmp.name)

    try:
        rmisc.json_dump_python_object(str(path), obj)

        with path.open() as f:
            result = json.load(f)

        assert result == obj

    finally:
        path.unlink()


def poly_geo_func(_coords: list) -> list:
    """Dummy function for testing semantic_segmentations_as_qupath_json."""
    return []


def test_semantic_segmentations_as_qupath_json() -> None:
    """Test semantic_segmentations_as_qupath_json.

    Ensure the Rust implementation returns the expected result.
    """
    class_colours = {
        0.0: [255, 0, 0],
        1.0: [0, 255, 0],
    }

    class_dict = {
        0.0: "Tumour",
        1.0: "Normal",
    }

    preds = da.from_array(
        np.array(
            [
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )
    )

    scale_factor = (0.5, 0.5)

    layer_list = [0.0, 1.0]

    result = rmisc.semantic_segmentations_as_qupath_json(
        layer_list, preds, scale_factor, class_dict, class_colours, cv2, poly_geo_func
    )

    expected = [
        {
            "type": "Feature",
            "geometry": [],
            "id": "class_0_0",
            "properties": {"classification": {"name": "Tumour", "color": [255, 0, 0]}},
            "objectType": "annotation",
            "name": "Tumour",
            "class_value": 0.0,
        },
        {
            "type": "Feature",
            "geometry": [],
            "id": "class_1_1",
            "properties": {"classification": {"name": "Normal", "color": [0, 255, 0]}},
            "objectType": "annotation",
            "name": "Normal",
            "class_value": 1.0,
        },
    ]

    assert result == expected


def dummy_process_contours(
    contours: list[np.ndarray],
    hierarchy: np.ndarray,
    scale_factor: tuple[float, float] = (1, 1),
    offset: np.ndarray | None = None,
    properties: dict[str, JSON] | None = None,
) -> list[DummyAnnotation]:
    """Used for test_semantic_segmentations_as_annotations()."""
    annotations = []
    _ = offset
    _ = scale_factor
    _ = hierarchy

    for contour in contours:
        points = contour.reshape(-1, 2)
        annotations.append(
            DummyAnnotation(
                polygon=(
                    points[:, 0].min(),
                    points[:, 1].min(),
                    points[:, 0].max(),
                    points[:, 1].max(),
                ),
                properties=cast("dict[str, str | float]", properties.copy()),
            )
        )

    return annotations


def test_semantic_segmentations_as_annotations() -> None:
    """Test semantic_segmentations_as_annotations.

    Ensure the Rust implementation returns the expected result.
    """
    class_dict = {
        0.0: "Tumour",
        1.0: "Normal",
    }

    preds = da.from_array(
        np.array(
            [
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )
    )

    scale_factor = (0.5, 0.5)
    layer_list = [0.0, 1.0]

    result = rmisc.semantic_segmentations_as_annotations(
        layer_list,
        preds,
        scale_factor,
        class_dict,
        None,
        cv2,
        dummy_process_contours,
    )

    assert len(result) == 2

    assert isinstance(result[0], DummyAnnotation)
    assert result[0].polygon == (0.0, 0.0, 2.0, 2.0)
    assert result[0].properties == {
        "type": "Tumour",
        "class": 0.0,
    }

    assert isinstance(result[1], DummyAnnotation)
    assert result[1].polygon == (0.0, 0.0, 4.0, 3.0)
    assert result[1].properties == {
        "type": "Normal",
        "class": 1.0,
    }
