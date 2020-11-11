from tiatoolbox.tools import patchextraction
from tiatoolbox.utils.exceptions import MethodNotSupported
from tiatoolbox.utils.misc import imread

import pytest
import pathlib


def test_get_patch_extractor():
    """Test get_patch_extractor returns the right object."""
    file_parent_dir = pathlib.Path(__file__).parent
    input_image = imread(file_parent_dir.joinpath("data/source_image.png"))
    labels = file_parent_dir.joinpath("data/sample_patch_extraction.csv")
    points = patchextraction.get_patch_extractor(
        input_image=input_image,
        labels=labels,
        method_name="point",
        img_patch_h=200,
        img_patch_w=200,
    )

    assert isinstance(points, patchextraction.PointsPatchExtractor)

    with pytest.raises(MethodNotSupported):
        points.merge_patches()

    fixed_window = patchextraction.get_patch_extractor(
        input_image=input_image,
        method_name="fixedwindow",
        img_patch_h=200,
        img_patch_w=200,
    )

    assert isinstance(fixed_window, patchextraction.FixedWindowPatchExtractor)

    variable_window = patchextraction.get_patch_extractor(
        input_image=input_image,
        method_name="variablewindow",
        img_patch_h=200,
        img_patch_w=200,
    )

    assert isinstance(variable_window, patchextraction.VariableWindowPatchExtractor)

    with pytest.raises(MethodNotSupported):
        patchextraction.get_patch_extractor("unknown")


def test_points_patch_extractor():
    file_parent_dir = pathlib.Path(__file__).parent
    input_image = imread(file_parent_dir.joinpath("data/source_image.png"))
    labels = file_parent_dir.joinpath("data/sample_patch_extraction.csv")
    points = patchextraction.get_patch_extractor(
        input_image=input_image, labels=labels, method_name="point", img_patch_h=200, img_patch_w=200
    )

    with pytest.raises(MethodNotSupported):
        points.merge_patches(patches=None)
