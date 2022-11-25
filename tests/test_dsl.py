"""Tests for predicate module."""
import json
import sqlite3
from numbers import Number
from typing import Union

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
}


def test_invalid_sqltriplet():
    """Test invalid SQLTriplet."""
    with pytest.raises(ValueError, match="Invalid SQLTriplet"):
        str(SQLTriplet(SQLJSONDictionary()))


def test_json_contains():
    """Test json_contains function."""
    properties = json.dumps(SAMPLE_PROPERTIES)
    assert json_contains(properties, "int")
    assert json_contains(json.dumps([1]), 1)
    assert not json_contains(properties, "foo")


def sqlite_eval(query: Union[str, Number]):
    """Evaluate an SQL predicate on dummpy data and return the result.

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
        cur.execute(f"SELECT {query} FROM test")
        (result,) = cur.fetchone()
    return result


class TestPredicate:
    """Test predicate statments with various backends."""

    scenarios = [
        (
            "Python",
            {
                "eval_globals": PY_GLOBALS,
                "eval_locals": {"props": SAMPLE_PROPERTIES},
                "check": lambda x: x,
            },
        ),
        (
            "SQLite",
            {
                "eval_globals": SQL_GLOBALS,
                "eval_locals": {"props": SQLJSONDictionary()},
                "check": sqlite_eval,
            },
        ),
    ]

    @staticmethod
    def test_number_binary_operations(eval_globals, eval_locals, check):
        """Check that binary operations between ints does not error."""
        for op in BINARY_OP_STRINGS:
            query = f"2 {op} 2"
            result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
            assert isinstance(check(result), Number)

    @staticmethod
    def test_property_binary_operations(eval_globals, eval_locals, check):
        """Check that binary operations between properties does not error."""
        for op in BINARY_OP_STRINGS:
            query = f"props['int'] {op} props['int']"
            result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
            assert isinstance(check(result), Number)

    @staticmethod
    def test_r_binary_operations(eval_globals, eval_locals, check):
        """Test right hand binary operations between numbers and properties."""
        for op in BINARY_OP_STRINGS:
            query = f"2 {op} props['int']"
            result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
            assert isinstance(check(result), Number)

    @staticmethod
    def test_number_prefix_operations(eval_globals, eval_locals, check):
        """Test prefix operations on numbers."""
        for op in PREFIX_OP_STRINGS:
            query = f"{op}1"
            result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
            assert isinstance(check(result), Number)

    @staticmethod
    def test_property_prefix_operations(eval_globals, eval_locals, check):
        """Test prefix operations on properties."""
        for op in PREFIX_OP_STRINGS:
            query = f"{op}props['int']"
            result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
            assert isinstance(check(result), Number)

    @staticmethod
    def test_regex_nested_props(eval_globals, eval_locals, check):
        """Test regex on nested properties."""
        query = "props['nesting']['fib'][4]"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == 5

    @staticmethod
    def test_regex_str_props(eval_globals, eval_locals, check):
        """Test regex on string properties."""
        query = "regexp('Hello', props['string'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == "Hello"

    @staticmethod
    def test_regex_str_str(eval_globals, eval_locals, check):
        """Test regex on string and string."""
        query = "regexp('Hello', 'Hello world!')"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == "Hello"

    @staticmethod
    def test_regex_props_str(eval_globals, eval_locals, check):
        """Test regex on property and string."""
        query = "regexp(props['string'], 'Hello world!')"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == "Hello world!"

    @staticmethod
    def test_regex_ignore_case(eval_globals, eval_locals, check):
        """Test regex with ignorecase flag."""
        query = "regexp('hello', props['string'], re.IGNORECASE)"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == "Hello"

    @staticmethod
    def test_regex_no_match(eval_globals, eval_locals, check):
        """Test regex with no match."""
        query = "regexp('Yello', props['string'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) is None

    @staticmethod
    def test_has_key(eval_globals, eval_locals, check):
        """Test has_key function."""
        query = "has_key(props, 'foo')"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is False

    @staticmethod
    def test_is_none(eval_globals, eval_locals, check):
        """Test is_none function."""
        query = "is_none(props['null'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_is_not_none(eval_globals, eval_locals, check):
        """Test is_not_none function."""
        query = "is_not_none(props['int'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_nested_has_key(eval_globals, eval_locals, check):
        """Test nested has_key function."""
        query = "has_key(props['dict'], 'a')"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_list_sum(eval_globals, eval_locals, check):
        """Test sum function on a list."""
        query = "sum(props['list'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == sum(SAMPLE_PROPERTIES["list"])

    @staticmethod
    def test_abs(eval_globals, eval_locals, check):
        """Test abs function."""
        query = "abs(props['neg'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == 1

    @staticmethod
    def test_not(eval_globals, eval_locals, check):
        """Test not operator."""
        query = "not props['bool']"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is False

    @staticmethod
    def test_props_int_keys(eval_globals, eval_locals, check):
        """Test props with int keys."""
        query = "props['list'][1]"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == 1

    @staticmethod
    def test_props_get(eval_globals, eval_locals, check):
        """Test props.get function."""
        query = "is_none(props.get('foo'))"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_props_get_default(eval_globals, eval_locals, check):
        """Test props.get function with default."""
        query = "props.get('foo', 42)"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == 42

    @staticmethod
    def test_in_list(eval_globals, eval_locals, check):
        """Test in operator for list."""
        query = "1 in props.get('list')"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_has_key_exception(eval_globals, eval_locals, check):
        """Test has_key function with exception."""
        query = "has_key(1, 'a')"
        with pytest.raises(TypeError, match="(not iterable)|(Unsupported type)"):
            _ = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123

    @staticmethod
    def test_logical_and(eval_globals, eval_locals, check):
        """Test logical and operator."""
        query = "props['bool'] & is_none(props['null'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_logical_or(eval_globals, eval_locals, check):
        """Test logical or operator."""
        query = "props['bool'] | (props['int'] < 2)"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_nested_logic(eval_globals, eval_locals, check):
        """Test nested logical operators."""
        query = "(props['bool'] | (props['int'] < 2)) & abs(props['neg'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_contains_list(eval_globals, eval_locals, check):
        """Test contains operator for list."""
        query = "1 in props['list']"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_contains_dict(eval_globals, eval_locals, check):
        """Test contains operator for dict."""
        query = "'a' in props['dict']"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_contains_str(eval_globals, eval_locals, check):
        """Test contains operator for str."""
        query = "'Hello' in props['string']"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True
