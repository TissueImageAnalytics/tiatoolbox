"""Tests for tiatoolbox.type_hints module."""

from collections.abc import Callable
from typing import Literal

import pytest

from tiatoolbox import type_hints


def test_aliases_exist() -> None:
    """Ensure all expected type aliases are defined in type_hints."""
    expected_aliases = [
        "JSON",
        "NumPair",
        "IntPair",
        "Resolution",
        "Units",
        "Bounds",
        "IntBounds",
        "Geometry",
        "Properties",
        "QueryGeometry",
        "CallablePredicate",
        "CallableSelect",
        "Predicate",
        "Select",
        "NumpyPadLiteral",
    ]
    for alias in expected_aliases:
        assert hasattr(type_hints, alias), f"Missing alias: {alias}"


def test_units_is_literal() -> None:
    """Check that Units alias is a Literal type."""
    assert isinstance(type_hints.Units, type(Literal["mpp"]))


def test_callable_predicate_signature() -> None:
    """Verify CallablePredicate expects Properties and returns bool."""
    alias = type_hints.CallablePredicate
    # Check that it's a typing Callable
    assert getattr(alias, "__origin__", None) is Callable
    # Check argument and return types
    args = alias.__args__
    assert len(args) == 2
    assert args[1] is bool


@pytest.mark.parametrize("alias", ["Bounds", "IntBounds"])
def test_bounds_alias_is_tuple(alias: str) -> None:
    """Check that Bounds and IntBounds are tuple type hints."""
    assert "tuple" in str(getattr(type_hints, alias))


def test_numpy_pad_literal_contains_expected_values() -> None:
    """Ensure NumpyPadLiteral includes common numpy pad modes."""
    modes = ["constant", "reflect", "wrap"]
    for mode in modes:
        assert mode in type_hints.NumpyPadLiteral.__args__
