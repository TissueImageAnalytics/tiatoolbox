from typing import Any

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from tiatoolbox.annotation.storage import Annotation

def add(a: int, b: int) -> int: ...
def json_dump_python_object(
    save_path: str,
    obj: object,
) -> None: ...
def patch_predictions_as_annotations(
    annotation_class: type[Annotation],
    polygon_class: type[Polygon],
    preds: list[float],
    keys_contains_labels: bool,
    keys_contains_probabilities: bool,
    class_dict: dict[float, str | float],
    py_class_probs: NDArray[np.float64],
    py_patch_coords: NDArray[np.float64],
    classes_predicted: list[float],
    labels: list[float],
) -> list[Any]: ...
def patch_predictions_as_qupath_json(
    class_colours: dict[float, list[int]],
    preds: list[float],
    class_dict: dict[float, str],
    py_patch_coords: NDArray[np.float64],
) -> list[dict[str, Any]]: ...
def contrast_enhancer(
    img: NDArray[np.uint8],
    low_p: int,
    high_p: int,
) -> NDArray[np.uint8]: ...
