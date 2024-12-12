"""Test for predicate module."""

from __future__ import annotations

import json
import sqlite3
from numbers import Number
from typing import Callable, ClassVar

import pytest

from tiatoolbox.annotation.dsl import (
    PY_GLOBALS,
    SQL_GLOBALS,
    SQLJSONDictionary,
    SQLTriplet,
    json_contains,
    json_list_sum,
    py_regexp,
)

BINARY_OP_STRINGS = [
    "+",
    "-",
    "/",
    "//",
    "*",
    "<",
    ">",
    "<=",
    ">=",
    "==",
    "!=",
    "**",
    "&",
    "|",
    "%",
]
PREFIX_OP_STRINGS = ["-", "not "]
FUNCTION_NAMES = ["abs", "is_none", "is_not_none", "has_key"]
SAMPLE_PROPERTIES = {
    "int": 2,
    "string": "Hello world!",
    "null": None,
    "dict": {"a": 1},
    "list": [0, 1, 2, 3],
    "neg": -1,
    "bool": True,
    "nesting": {"fib": [1, 1, 2, 3, 5], "foo": {"bar": "baz"}},
    "dot.key": 3.14,
}


def test_invalid_sqltriplet() -> None:
    """Test invalid SQLTriplet."""
    with pytest.raises(ValueError, match="Invalid SQLTriplet"):
        str(SQLTriplet(SQLJSONDictionary()))


def test_json_contains() -> None:
    """Test json_contains function."""
    properties = json.dumps(SAMPLE_PROPERTIES)
    assert json_contains(properties, "int")
    assert json_contains(json.dumps([1]), 1)
    assert not json_contains(properties, "foo")


def sqlite_eval(query: str | Number) -> bool:
    """Evaluate an SQL predicate on dummy data and return the result.

    Args:
        query (Union[str, Number]): SQL predicate to evaluate.

    Returns:
        bool: Result of the evaluation.

    """
    with sqlite3.connect(":memory:") as con:
        con.create_function("REGEXP", 2, py_regexp)
        con.create_function("REGEXP", 3, py_regexp)
        con.create_function("LISTSUM", 1, json_list_sum)
        con.create_function("CONTAINS", 1, json_contains)
        cur = con.cursor()
        cur.execute("CREATE TABLE test(properties TEXT)")
        cur.execute(
            "INSERT INTO test VALUES(:properties)",
            {"properties": json.dumps(SAMPLE_PROPERTIES)},
        )
        con.commit()
        if isinstance(query, str):
            assert query.count("(") == query.count(")")
            assert query.count("[") == query.count("]")
        cur.execute(f"SELECT {query} FROM test")  # noqa: S608
        (result,) = cur.fetchone()
    return result


class TestSQLite:
    """Test converting from our DSL to an SQLite backend."""

    @staticmethod
    def test_prop_or_prop() -> None:
        """Test OR operator between two prop accesses."""
        query = eval(  # skipcq: PYL-W0123
            "(props['int'] == 2) | (props['int'] == 3)",
            SQL_GLOBALS,
            {},
        )
        assert str(query) == (
            """((json_extract(properties, '$."int"') == 2) OR """
            """(json_extract(properties, '$."int"') == 3))"""
        )


py_variables: dict = {
    "eval_globals": PY_GLOBALS,
    "eval_locals": {"props": SAMPLE_PROPERTIES},
    "check": lambda x: x,
}

sqlite_variables: dict = {
    "eval_globals": SQL_GLOBALS,
    "eval_locals": {"props": SQLJSONDictionary()},
    "check": sqlite_eval,
}

scenario_python: tuple = (
    "Python",
    {"scenario_variables": py_variables},
)
scenario_sqlite = (
    "SQLite",
    {"scenario_variables": sqlite_variables},
)


def extract_variables(scenario_variables: dict) -> tuple[dict, dict, Callable]:
    """Extract variables from scenario variables."""
    eval_globals = scenario_variables["eval_globals"]
    eval_locals = scenario_variables["eval_locals"]
    check = scenario_variables["check"]
    return eval_globals, eval_locals, check


class TestPredicate:
    """Test predicate statements with various backends."""

    scenarios: ClassVar[list[str, dict]] = [scenario_python, scenario_sqlite]

    @staticmethod
    def test_number_binary_operations(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Check that binary operations between ints does not error."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        for op in BINARY_OP_STRINGS:
            query = f"2 {op} 2"
            result = eval(  # skipcq: PYL-W0123
                query,
                eval_globals,
                eval_locals,
            )
            assert isinstance(check(result), Number)

    @staticmethod
    def test_property_binary_operations(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Check that binary operations between properties does not error."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        for op in BINARY_OP_STRINGS:
            query = f"props['int'] {op} props['int']"
            result = eval(  # skipcq: PYL-W0123
                query,
                eval_globals,
                eval_locals,
            )
            assert isinstance(check(result), Number)

    @staticmethod
    def test_r_binary_operations(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test right hand binary operations between numbers and properties."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        for op in BINARY_OP_STRINGS:
            query = f"2 {op} props['int']"
            result = eval(  # skipcq: PYL-W0123
                query,
                eval_globals,
                eval_locals,
            )
            assert isinstance(check(result), Number)

    @staticmethod
    def test_number_prefix_operations(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test prefix operations on numbers."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        for op in PREFIX_OP_STRINGS:
            query = f"{op}1"
            result = eval(  # skipcq: PYL-W0123
                query,
                eval_globals,
                eval_locals,
            )
            assert isinstance(check(result), Number)

    @staticmethod
    def test_property_prefix_operations(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test prefix operations on properties."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        for op in PREFIX_OP_STRINGS:
            query = f"{op}props['int']"
            result = eval(  # skipcq: PYL-W0123
                query,
                eval_globals,
                eval_locals,
            )
            assert isinstance(check(result), Number)

    @staticmethod
    def test_regex_nested_props(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test regex on nested properties."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "props['nesting']['fib'][4]"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert check(result) == 5

    @staticmethod
    def test_regex_str_props(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test regex on string properties."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "regexp('Hello', props['string'])"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert check(result) == "Hello"

    @staticmethod
    def test_regex_str_str(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test regex on string and string."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "regexp('Hello', 'Hello world!')"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert check(result) == "Hello"

    @staticmethod
    def test_regex_props_str(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test regex on property and string."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "regexp(props['string'], 'Hello world!')"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert check(result) == "Hello world!"

    @staticmethod
    def test_regex_ignore_case(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test regex with ignorecase flag."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "regexp('hello', props['string'], re.IGNORECASE)"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert check(result) == "Hello"

    @staticmethod
    def test_regex_no_match(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test regex with no match."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "regexp('Yello', props['string'])"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert check(result) is None

    @staticmethod
    def test_has_key(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test has_key function."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "has_key(props, 'foo')"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert bool(check(result)) is False

    @staticmethod
    def test_is_none(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test is_none function."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "is_none(props['null'])"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert bool(check(result)) is True

    @staticmethod
    def test_is_not_none(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test is_not_none function."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "is_not_none(props['int'])"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert bool(check(result)) is True

    @staticmethod
    def test_nested_has_key(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test nested has_key function."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "has_key(props['dict'], 'a')"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert bool(check(result)) is True

    @staticmethod
    def test_list_sum(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test sum function on a list."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "sum(props['list'])"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert check(result) == sum(SAMPLE_PROPERTIES["list"])

    @staticmethod
    def test_abs(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test abs function."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "abs(props['neg'])"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert check(result) == 1

    @staticmethod
    def test_not(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test not operator."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "not props['bool']"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert bool(check(result)) is False

    @staticmethod
    def test_props_int_keys(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test props with int keys."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "props['list'][1]"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert check(result) == 1

    @staticmethod
    def test_props_get(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test props.get function."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "is_none(props.get('foo'))"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert bool(check(result)) is True

    @staticmethod
    def test_props_get_default(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test props.get function with default."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "props.get('foo', 42)"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert check(result) == 42

    @staticmethod
    def test_in_list(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test in operator for list."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "1 in props.get('list')"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert bool(check(result)) is True

    @staticmethod
    def test_has_key_exception(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test has_key function with exception."""
        eval_globals, eval_locals, _ = extract_variables(scenario_variables)
        query = "has_key(1, 'a')"
        with pytest.raises(TypeError, match="(not iterable)|(Unsupported type)"):
            _ = eval(  # skipcq: PYL-W0123
                query,
                eval_globals,
                eval_locals,
            )

    @staticmethod
    def test_logical_and(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test logical and operator."""
        query = "props['bool'] & is_none(props['null'])"
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert bool(check(result)) is True

    @staticmethod
    def test_logical_or(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test logical or operator."""
        query = "props['bool'] | (props['int'] < 2)"
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert bool(check(result)) is True

    @staticmethod
    def test_nested_logic(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test nested logical operators."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "(props['bool'] | (props['int'] < 2)) & abs(props['neg'])"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert bool(check(result)) is True

    @staticmethod
    def test_contains_list(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test contains operator for list."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "1 in props['list']"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert bool(check(result)) is True

    @staticmethod
    def test_contains_dict(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test contains operator for dict."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "'a' in props['dict']"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert bool(check(result)) is True

    @staticmethod
    def test_contains_str(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test contains operator for str."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "'Hello' in props['string']"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )
        assert bool(check(result)) is True

    @staticmethod
    def test_key_with_period(
        scenario_variables: dict[str, dict],
    ) -> None:
        """Test key with period."""
        eval_globals, eval_locals, check = extract_variables(scenario_variables)
        query = "props['dot.key']"
        result = eval(  # skipcq: PYL-W0123
            query,
            eval_globals,
            eval_locals,
        )

        assert check(result) == 3.14
