from tiatoolbox.tools import patchextraction


def test_get_patch_extractor():
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
