"""Test for rust functionality."""

import json
import tempfile
import typing
from pathlib import Path
from typing import cast

import cv2
import dask.array as da
import numpy as np
import pytest
from dask import delayed
from matplotlib import pyplot as plt
from shapely.geometry import mapping
from shapely.geometry import shape as feature2geometry

from tiatoolbox import rmisc, rmultitask, utils
from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.annotation.storage import Annotation
from tiatoolbox.type_hints import JSON
from tiatoolbox.utils import misc
from tiatoolbox.utils.misc import make_valid_poly, tqdm_dask_progress_bar


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


class DummyCV2:
    """Dummy cv2 module returning a contour with fewer than 3 points."""

    RETR_CCOMP = 0
    CHAIN_APPROX_NONE = 1

    @staticmethod
    @typing.override
    def findContours(
        _layer: np.ndarray,
        _mode: int,
        _method: int,
    ) -> tuple[list[np.ndarray], None]:
        """Dummy cv2 findContours eturning a contour with fewer than 3 points."""
        contour = np.array(
            [
                [[0, 0]],
                [[1, 1]],
            ],
            dtype=np.int32,
        )
        return [contour], None


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

    result = rmisc.semantic_segmentations_as_qupath_json(
        layer_list,
        preds,
        scale_factor,
        class_dict,
        class_colours,
        DummyCV2,
        poly_geo_func,
    )

    assert result == []


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

    class_dict = {0.0: "Tumour"}

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
        "type": 1.0,
        "class": 1.0,
    }


class DaskDelayedJSONStore:
    """Compute and write TIAToolbox annotations using batched Dask Delayed tasks.

    This class parallelizes annotation construction using Dask Delayed while
    avoiding serialization overhead by storing contours and prediction arrays
    as instance attributes. Annotations are computed in batches and written
    directly to a TIAToolbox `SQLiteStore` via `append_many()`.

    """

    def __init__(
        self,
        contours: np.ndarray,
        processed_predictions: dict,
    ) -> None:
        """Initialize :class:`DaskDelayedAnnotationStore`.

        Args:
            contours (np.ndarray):
                A sequence of polygon contours. Each element is an array-like
                of shape ``(N_i, 2)`` representing the coordinates of a single
                object contour.

            processed_predictions (dict):
                A dictionary of per-object prediction fields. Each key maps to
                an array-like of length ``len(contours)``. Example keys include
                ``"type"``, ``"prob"``, ``"centroid"``, etc. May also contain
                a global field ``"geom_type"``.

        """
        self._contours = contours
        self._processed_predictions = processed_predictions

    def _build_single_annotation(
        self,
        i: int,
        class_dict: dict[int, str] | None,
        origin: tuple[float, float],
        scale_factor: tuple[float, float],
    ) -> Annotation:
        """Build a single annotation for index ``i``.

        This method performs:
        - geometry creation
        - coordinate scaling and translation
        - per-object property extraction
        - optional class label mapping

        Args:
            i (int):
                Index of the object to convert into an annotation.

            class_dict (dict[int, str] | None):
                Optional mapping from integer class IDs to string labels.
                If ``None``, raw integer class IDs are used.

            origin (tuple[float, float]):
                Translation offset ``(x, y)`` applied after scaling.

            scale_factor (tuple[float, float]):
                Scaling factors ``(sx, sy)`` applied to contour coordinates.

        Returns:
            Annotation:
                A fully constructed TIAToolbox `Annotation` instance.

        """
        geom = make_valid_poly(
            feature2geometry(
                {
                    "type": self._processed_predictions.get("geom_type", "Polygon"),
                    "coordinates": scale_factor * np.array([self._contours[i]]),
                }
            ),
            tuple(origin),
        )

        if class_dict is None:
            class_dict = {}

        properties = rmultitask.build_single_annotation(
            np, i, self._processed_predictions, class_dict
        )

        return Annotation(geom, properties)

    def _build_single_qupath_feature(
        self,
        i: int,
        class_dict: dict | None,
        origin: tuple[float, float],
        scale_factor: tuple[float, float],
        class_colors: dict,
    ) -> dict:
        """Build a single feature for index ``i``.

        This method performs:
        - geometry creation
        - coordinate scaling and translation
        - per-object property extraction
        - optional class label mapping

        Args:
            i (int):
                Index of the object to convert into an annotation.

            class_dict (dict[int, str] | None):
                Optional mapping from integer class IDs to string labels.
                If ``None``, raw integer class IDs are used.

            origin (tuple[float, float]):
                Translation offset ``(x, y)`` applied after scaling.

            scale_factor (tuple[float, float]):
                Scaling factors ``(sx, sy)`` applied to contour coordinates.

            class_colors (dict):
                Maps classes to specific colors.

        Returns:
            dict:
                A fully constructed Feature dictionary instance for writing
                to QuPath JSON.

        """
        geom = make_valid_poly(
            feature2geometry(
                {
                    "type": self._processed_predictions.get("geom_type", "Polygon"),
                    "coordinates": scale_factor * np.array([self._contours[i]]),
                }
            ),
            tuple(origin),
        )
        geo_map = mapping(geom)

        return rmultitask.build_single_qupath_feature(
            geo_map, self._processed_predictions, i, class_dict, class_colors
        )

    def compute_annotations(
        self,
        store: SQLiteStore,
        class_dict: dict[int, str] | None,
        origin: tuple[float, float] = (0, 0),
        scale_factor: tuple[float, float] = (1, 1),
        batch_size: int = 100,
        num_workers: int = 0,
        *,
        verbose: bool = True,
    ) -> SQLiteStore:
        """Compute annotations in batches and write them to a SQLiteStore.

        This method creates Dask Delayed tasks in batches to reduce scheduler
        overhead. Each batch is computed and written immediately using
        ``store.append_many()``.

        Args:
            store (SQLiteStore):
                A TIAToolbox SQLiteStore instance used to write annotations.

            class_dict (dict[int, str] | None):
                Optional mapping from integer class IDs to string labels.

            origin (tuple[float, float], optional):
                Translation offset ``(x, y)`` applied after scaling.
                Defaults to ``(0, 0)``.

            scale_factor (tuple[float, float], optional):
                Scaling factors ``(sx, sy)`` applied to contour coordinates.
                Defaults to ``(1, 1)``.

            batch_size (int, optional):
                Number of annotations to compute per batch. Larger batches
                reduce Dask scheduler overhead. Defaults to ``100``.

            num_workers (int, optional):
                Number of Dask workers to use. ``0`` means auto-detect.
                Passed through to the progress bar helper. Defaults to ``0``.

            verbose (bool, optional):
                Whether to display progress bars. Defaults to ``True``.

        Returns:
            SQLiteStore:
                The same store instance, after all annotations have been written.

        """
        return rmultitask.compute_annotations(
            store,
            class_dict,
            origin,
            scale_factor,
            batch_size,
            num_workers,
            verbose,
            len(self._contours),
            self._build_single_annotation,
            delayed,
            tqdm_dask_progress_bar,
        )

    def compute_qupath_json(
        self,
        class_dict: dict[int, str] | None,
        origin: tuple[float, float] = (0, 0),
        scale_factor: tuple[float, float] = (1, 1),
        save_path: Path | None = None,
        batch_size: int = 100,
        num_workers: int = 0,
        *,
        verbose: bool = True,
    ) -> Path:
        """Compute annotations in batches and return/save QuPath JSON."""
        if class_dict is None:
            class_dict = {}
        features = rmultitask.compute_qupath_json(
            class_dict,
            origin,
            scale_factor,
            batch_size,
            verbose,
            num_workers,
            len(self._contours),
            self._processed_predictions.get("type").tolist(),
            plt,
            self._build_single_qupath_feature,
            delayed,
            tqdm_dask_progress_bar,
        )
        qupath_json = {"type": "FeatureCollection", "features": features}

        return misc.save_qupath_json(save_path=save_path, qupath_json=qupath_json)


def test_qupath_feature_classification_block_skipped() -> None:
    """Test qupath_feature_classification_block_skipped fails."""
    qupath_json = DaskDelayedJSONStore.__new__(DaskDelayedJSONStore)
    qupath_json._contours = [np.array([[0, 0], [1, 0], [1, 1]])]
    qupath_json._processed_predictions = {"type": np.array([1], dtype=object)}

    class_dict = {1: "Tumor"}
    class_colors = {0: [255, 0, 0]}  # does NOT contain 1

    feat = qupath_json._build_single_qupath_feature(
        i=0,
        class_dict=class_dict,
        origin=(0, 0),
        scale_factor=(1, 1),
        class_colors=class_colors,
    )

    assert feat["properties"]["type"] == "Tumor"
    assert "classification" not in feat["properties"]


def test_compute_qupath_json_valid_ids_not_empty(track_tmp_path: Path) -> None:
    """Test compute_qupath_json valid ids not empty."""
    store = DaskDelayedJSONStore.__new__(DaskDelayedJSONStore)

    # One simple contour
    store._contours = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])]

    # Mixed type array → valid_ids = [1, 2]
    store._processed_predictions = {"type": np.array([1, None, 2], dtype=object)}

    out_path = track_tmp_path / "out.json"
    result_path = store.compute_qupath_json(
        class_dict=None,
        save_path=out_path,
        verbose=False,
    )

    # Load JSON
    data = json.loads(Path(result_path).read_text())
    props = data["features"][0]["properties"]

    # 1. class_dict should have been inferred as {0:0, 1:1, 2:2}
    assert props["type"] in (1, 2)

    # 2. type must NOT be null
    assert props["type"] is not None

    # 3. classification block should exist only if class_value in class_colours
    assert "null" not in json.dumps(data)


def test_compute_qupath_json_with_explicit_class_dict(
    monkeypatch: pytest.MonkeyPatch,
    track_tmp_path: Path,
) -> None:
    """Test compute_qupath_json when class_dict is explicitly provided."""
    store = DaskDelayedJSONStore.__new__(DaskDelayedJSONStore)

    store._contours = [
        np.array(
            [
                [0, 0],
                [10, 0],
                [10, 10],
            ],
            dtype=float,
        ),
    ]

    store._processed_predictions = {
        "type": np.array([1], dtype=object),
    }

    class_dict = {
        0: "background",
        1: "tumour",
    }

    def _fake_build_single_qupath_feature(
        i: int,
        class_dict_in: dict[int, str],
        origin: tuple[float, float],
        scale_factor: tuple[float, float],
        class_colors: dict[int, list[int]],
    ) -> dict[str, object]:
        """Return a simple fake feature."""
        _ = i, origin, scale_factor

        assert class_dict_in is class_dict
        assert 0 in class_colors
        assert 1 in class_colors

        return {
            "type": "Feature",
            "properties": {
                "type": "tumour",
            },
        }

    def _fake_save_qupath_json(
        save_path: Path | None,
        qupath_json: dict[str, object],
    ) -> dict[str, object]:
        """Return JSON rather than touching disk."""
        assert save_path == track_tmp_path / "output.json"
        return qupath_json

    monkeypatch.setattr(
        store,
        "_build_single_qupath_feature",
        _fake_build_single_qupath_feature,
    )

    monkeypatch.setattr(
        misc,
        "save_qupath_json",
        _fake_save_qupath_json,
    )

    result = store.compute_qupath_json(
        class_dict=class_dict,
        save_path=track_tmp_path / "output.json",
        verbose=False,
    )

    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 1


def test_qupath_feature_class_dict_lookup_fails() -> None:
    """Test qupath_feature_class_dict lookup fails."""
    qupath_json = DaskDelayedJSONStore.__new__(DaskDelayedJSONStore)
    qupath_json._contours = [np.array([[0, 0], [1, 0], [1, 1]])]
    qupath_json._processed_predictions = {"type": np.array([5], dtype=object)}

    class_dict = {0: "A", 1: "B"}  # does NOT contain 5
    class_colors = {0: [255, 0, 0], 1: [0, 255, 0]}  # also does NOT contain 5

    feat = qupath_json._build_single_qupath_feature(
        i=0,
        class_dict=class_dict,
        origin=(0, 0),
        scale_factor=(1, 1),
        class_colors=class_colors,
    )

    # type should fall back to raw value (5)
    assert feat["properties"]["type"] == 5
    # classification block should NOT appear
    assert "classification" not in feat["properties"]


def test_compute_qupath_json_string_class_names(track_tmp_path: Path) -> None:
    """Test compute_qupath_json string class names not empty and str."""
    store = DaskDelayedJSONStore.__new__(DaskDelayedJSONStore)

    # One simple contour
    store._contours = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])]

    # String class names → triggers the "already class names" branch
    store._processed_predictions = {
        "type": np.array(["Tumor", None, "Stroma"], dtype=object)
    }

    # Run compute_qupath_json with class_dict=None
    out_path = track_tmp_path / "out.json"
    result_path = store.compute_qupath_json(
        class_dict=None,
        save_path=out_path,
        verbose=False,
    )

    # Load JSON
    data = json.loads(Path(result_path).read_text())
    props = data["features"][0]["properties"]

    # --- Assertions ---

    # 1. type must be one of the string class names
    assert props["type"] in ("Tumor", "Stroma")

    # 2. type must NOT be null
    assert props["type"] is not None

    # 3. class_dict should have been inferred as identity mapping
    #    "Stroma": "Stroma", "Tumor": "Tumor"
    #    So classification block should exist only if class_colours
    #    contains the key, but we don't enforce that here — just
    #    ensure no nulls
    assert "null" not in json.dumps(data)
