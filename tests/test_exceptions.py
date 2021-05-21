"""Tests for exceptions used in the toolbox."""

from tiatoolbox.utils.exceptions import FileNotSupported, MethodNotSupported
from tiatoolbox.wsicore.save_tiles import save_tiles
from tiatoolbox.wsicore.slide_info import slide_info
from tiatoolbox.tools.stainnorm import get_normaliser
from tiatoolbox import utils


import pytest
import pathlib


def test_exception_tests():
    """Test for Exceptions."""

    with pytest.raises(FileNotSupported):
        utils.misc.save_yaml(
            slide_info(input_path="/mnt/test/sample.txt", verbose=True).as_dict(),
            "test.yaml",
        )

    with pytest.raises(FileNotSupported):
        save_tiles(
            input_path="/mnt/test/sample.txt",
            tile_objective_value=5,
            output_dir=str(pathlib.Path(__file__).parent.joinpath("tiles_save_tiles")),
            verbose=True,
        )

    with pytest.raises(MethodNotSupported):
        get_normaliser(method_name="invalid_normaliser")

    with pytest.raises(Exception) as e:
        get_normaliser(method_name="reinhard", stain_matrix="[1, 2]")
    assert str(e.value) == "stain_matrix is only defined when using custom"
