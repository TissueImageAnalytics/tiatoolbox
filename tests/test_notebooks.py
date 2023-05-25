import pytest
from testbook import testbook

are_notebooks_skipped = pytest.mark.skipif("not config.getoption('notebooks')")


@are_notebooks_skipped
@testbook("examples/01-wsi-reading.ipynb", execute=True)
def test_wsi_reading(tb):
    assert "mpp" in tb["info_dict"]
    assert tb["thumbnail"].shape == (185, 139, 3)
    assert tb["img_rect"].shape == (256, 256, 3)

    bounds = tb["bounds"]
    bounds_size = (bounds[2] - bounds[0], bounds[3] - bounds[1])
    assert tb["img_bounds"].shape == (*bounds_size, 3)
