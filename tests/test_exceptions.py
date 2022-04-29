"""Tests for exceptions used in the toolbox."""

import pathlib

import pytest

from tiatoolbox import utils
from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.utils.exceptions import FileNotSupported, MethodNotSupported
from tiatoolbox.wsicore.save_tiles import save_tiles
from tiatoolbox.wsicore.slide_info import slide_info


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
        get_normalizer(method_name="invalid_normalizer")

    with pytest.raises(
        Exception, match="`stain_matrix` is only defined when using.*custom"
    ):
        get_normalizer(method_name="reinhard", stain_matrix="[1, 2]")
