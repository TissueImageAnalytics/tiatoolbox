from tiatoolbox.tools import patchextraction
from tiatoolbox.utils.exceptions import MethodNotSupported, FileNotSupported
from tiatoolbox.utils.misc import imread
from tiatoolbox.dataloader.wsireader import (
    OpenSlideWSIReader,
    OmnyxJP2WSIReader,
    VirtualWSIReader,
)

import pytest
import pathlib


def test_get_patch_extractor():
    """Test get_patch_extractor returns the right object."""
    file_parent_dir = pathlib.Path(__file__).parent
    input_img = imread(file_parent_dir.joinpath("data/source_image.png"))
    labels = file_parent_dir.joinpath("data/sample_patch_extraction.csv")
    points = patchextraction.get_patch_extractor(
        input_img=input_img,
        labels=labels,
        method_name="point",
        patch_size=(200, 200),
    )

    assert isinstance(points, patchextraction.PointsPatchExtractor)

    with pytest.raises(MethodNotSupported):
        points.merge_patches()

    fixed_window = patchextraction.get_patch_extractor(
        input_img=input_img,
        method_name="fixedwindow",
        patch_size=(200, 200),
    )

    assert isinstance(fixed_window, patchextraction.FixedWindowPatchExtractor)

    variable_window = patchextraction.get_patch_extractor(
        input_img=input_img,
        method_name="variablewindow",
        patch_size=(200, 200),
    )

    assert isinstance(variable_window, patchextraction.VariableWindowPatchExtractor)

    with pytest.raises(MethodNotSupported):
        patchextraction.get_patch_extractor("unknown")


def test_points_patch_extractor(_sample_svs, _sample_jp2):
    """Test PointsPatchExtractor returns the right object."""
    file_parent_dir = pathlib.Path(__file__).parent
    input_img = pathlib.Path(file_parent_dir.joinpath("data/source_image.png"))
    labels = file_parent_dir.joinpath("data/sample_patch_extraction.csv")
    points = patchextraction.get_patch_extractor(
        input_img=input_img,
        labels=labels,
        method_name="point",
        patch_size=(200, 200),
    )

    assert isinstance(points.wsi, VirtualWSIReader)

    with pytest.raises(MethodNotSupported):
        points.merge_patches(patches=None)

    points = patchextraction.get_patch_extractor(
        input_img=pathlib.Path(_sample_svs),
        labels=labels,
        method_name="point",
        patch_size=(200, 200),
    )

    assert isinstance(points.wsi, OpenSlideWSIReader)

    points = patchextraction.get_patch_extractor(
        input_img=pathlib.Path(_sample_jp2),
        labels=labels,
        method_name="point",
        patch_size=(200, 200),
    )

    assert isinstance(points.wsi, OmnyxJP2WSIReader)

    with pytest.raises(FileNotSupported):
        false_image = pathlib.Path(file_parent_dir.joinpath("data/source_image.test"))
        _ = patchextraction.get_patch_extractor(
            input_img=false_image,
            labels=labels,
            method_name="point",
            patch_size=(200, 200),
        )
