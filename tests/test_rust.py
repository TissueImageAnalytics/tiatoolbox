"""Test for rust functionality."""

import numpy as np

from tiatoolbox import rmisc, utils


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
