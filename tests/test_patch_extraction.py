from tiatoolbox.tools import patchextraction
from tiatoolbox.utils.exceptions import MethodNotSupported

import pytest


def test_get_patch_extractor():
    """Test get_patch_extractor returns the right object."""
    points = patchextraction.get_patch_extractor(
        "point", img_patch_h=200, img_patch_w=200
    )

    assert isinstance(points, patchextraction.PointsPatchExtractor)

    fixed_window = patchextraction.get_patch_extractor(
        "fixedwindow", img_patch_h=200, img_patch_w=200
    )

    assert isinstance(fixed_window, patchextraction.FixedWindowPatchExtractor)

    variable_window = patchextraction.get_patch_extractor(
        "variablewindow", img_patch_h=200, img_patch_w=200
    )

    assert isinstance(variable_window, patchextraction.VariableWindowPatchExtractor)

    with pytest.raises(MethodNotSupported):
        patchextraction.get_patch_extractor("unknown")
