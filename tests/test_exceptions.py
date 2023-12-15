"""Test for exceptions used in the toolbox."""

import pytest

from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.utils.exceptions import MethodNotSupportedError


def test_exception_tests() -> None:
    """Test for Exceptions."""
    with pytest.raises(MethodNotSupportedError):
        get_normalizer(method_name="invalid_normalizer")

    with pytest.raises(
        ValueError,
        match="`stain_matrix` is only defined when using.*custom",
    ):
        get_normalizer(method_name="reinhard", stain_matrix="[1, 2]")
