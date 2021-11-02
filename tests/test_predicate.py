# skipcq: PYL-W0123
import json
import sqlite3
from numbers import Number
from typing import Union

import pytest

from tiatoolbox.annotation.predicate import (
    PY_GLOBALS,
    SQL_GLOBALS,
    Properties,
    json_contains,
    json_list_sum,
    regexp,
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

# Generate Parameterized Tests


def pytest_generate_tests(metafunc):
    """Generate (parameterize) test scenarios.

    Adapted from pytest documentation. For more information on
    parameterized tests see:
    https://docs.pytest.org/en/6.2.x/example/parametrize.html#a-quick-port-of-testscenarios

    """
    # Return if the test is not part of a class
    if metafunc.cls is None:
        return
    idlist = []
    argvalues = []
    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append([x[1] for x in items])
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


def sqlite_eval(query: Union[str, Number]):
    with sqlite3.connect(":memory:") as con:
        con.create_function("REGEXP", 2, regexp)
        con.create_function("REGEXP", 3, regexp)
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
                "eval_locals": {"props": Properties()},
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
    def test_propery_binary_operations(eval_globals, eval_locals, check):
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
        for op in PREFIX_OP_STRINGS:
            query = f"{op}1"
            result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
            assert isinstance(check(result), Number)

    @staticmethod
    def test_property_prefix_operations(eval_globals, eval_locals, check):
        for op in PREFIX_OP_STRINGS:
            query = f"{op}props['int']"
            result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
            assert isinstance(check(result), Number)

    @staticmethod
    def test_regex_nested_props(eval_globals, eval_locals, check):
        query = "props['nesting']['fib'][4]"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == 5

    @staticmethod
    def test_regex_str_props(eval_globals, eval_locals, check):
        query = "regexp('Hello', props['string'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == "Hello"

    @staticmethod
    def test_regex_str_str(eval_globals, eval_locals, check):
        query = "regexp('Hello', 'Hello world!')"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == "Hello"

    @staticmethod
    def test_regex_ignorecase(eval_globals, eval_locals, check):
        query = "regexp('hello', props['string'], re.IGNORECASE)"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == "Hello"

    @staticmethod
    def test_regex_no_match(eval_globals, eval_locals, check):
        query = "regexp('Yello', props['string'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) is None

    @staticmethod
    def test_has_key(eval_globals, eval_locals, check):
        query = "has_key(props, 'foo')"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is False

    @staticmethod
    def test_is_none(eval_globals, eval_locals, check):
        query = "is_none(props['null'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_is_not_none(eval_globals, eval_locals, check):
        query = "is_not_none(props['int'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_nested_has_key(eval_globals, eval_locals, check):
        query = "has_key(props['dict'], 'a')"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_list_sum(eval_globals, eval_locals, check):
        query = "sum(props['list'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == sum(SAMPLE_PROPERTIES["list"])

    @staticmethod
    def test_abs(eval_globals, eval_locals, check):
        query = "abs(props['neg'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == 1

    @staticmethod
    def test_not(eval_globals, eval_locals, check):
        query = "not props['bool']"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is False

    @staticmethod
    def test_props_int_keys(eval_globals, eval_locals, check):
        query = "props['list'][1]"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == 1

    @staticmethod
    def test_props_get(eval_globals, eval_locals, check):
        query = "is_none(props.get('foo'))"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_props_get_default(eval_globals, eval_locals, check):
        query = "props.get('foo', 42)"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert check(result) == 42

    @staticmethod
    def test_in_list(eval_globals, eval_locals, check):
        query = "1 in props.get('list')"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_has_key_exception(eval_globals, eval_locals, check):
        query = "has_key(1, 'a')"
        with pytest.raises(TypeError, match="(not iterable)|(Unsupported type)"):
            _ = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123

    @staticmethod
    def test_logical_and(eval_globals, eval_locals, check):
        query = "props['bool'] & is_none(props['null'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_logical_or(eval_globals, eval_locals, check):
        query = "props['bool'] | (props['int'] < 2)"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True

    @staticmethod
    def test_nested_logic(eval_globals, eval_locals, check):
        query = "(props['bool'] | (props['int'] < 2)) & abs(props['neg'])"
        result = eval(query, eval_globals, eval_locals)  # skipcq: PYL-W0123
        assert bool(check(result)) is True
