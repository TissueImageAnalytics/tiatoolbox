"""Domain specific langauge (DSL) for use in AnnotationStore queries and indexes.

This module facilitates conversion from a restricted subset of python
to another domain specific language, for example SQL. This is done using
`eval` and a set of provided globals and locals. Mainly used for
construction of predicate statements for AnnotationStore queries but
also used in statements for the creation of indexes to accelerate
queries.

This conversion should be assumed to be on a best-effort basis. Not
every expression valid in python can be evaluated to form a valid
matching SQL expression. However, for many common cases this will be
possible. For example, the simple python expression `props["class"] ==
42` can be converted to a valid SQL (SQLite flavour) predicate which
will access the properties JSON column and check that the value under
the key of "class" equals 42.

This predicate statement can be used as part of an SQL query and should
be faster than post-query filtering in python or filtering during the
query via a registered custom function callback.

An additional benefit is that the same input string can be used across
different backends. For example, the previous simple example predicate
string can be evaluated as both a valid python expression and can be
converted to an equivalent valid SQL expression simply by running `eval`
with a different set of globals from this module.

It is important to note that untrusted user input should not be
accepted, as arbitrary code can be run during the parsing of an input
string.

Supported operators and functions:
    - Property access: `props["key"]`
    - Math operations (`+`, `-`, `*`, `/`, `//`, `**`, `%`):
      `props["key"] + 1`
    - Boolean operations (`and`, `or`, `not`): `props["key"] and
      props["key"] == 1`
    - Key checking: `"key" in props`
    - List indexing: `props["key"][0]`
    - List sum: `sum(props["key"])`
    - List contains: `"value" in props["key"]`
    - None check (with a provided function): `is_none(props["key"])`
      `is_not_none(props["key"])`
    - Regex (with a provided function): `regexp(pattern, props["key"])`

Unsupported operations:
    - The `is` operator: `props["key"] is None`
    - Imports: `import re`
    - List length: `len(props["key"])` (support planned)

Compile options:
    Some mathematical functions will not function if the compile option
    `ENABLE_MATH_FUNCTIONS` is not set. These are:

    - `//` (floor division)

"""
import json
import operator
import re
from dataclasses import dataclass
from numbers import Number
from typing import Any, Callable, Optional, Union


@dataclass
class SQLNone:
    """Sentinel object for SQL NULL within expressions."""

    def __str__(self) -> str:
        return "NULL"

    def __repr__(self) -> str:
        return str(self)  # pragma: no cover


class SQLExpression:
    """SQL expression base class."""

    __hash__ = None

    def __repr__(self):
        return str(self)  # pragma: no cover

    def __add__(self, other):
        return SQLTriplet(self, operator.add, other)

    def __radd__(self, other):
        return SQLTriplet(other, operator.add, self)

    def __mul__(self, other):
        return SQLTriplet(self, operator.mul, other)

    def __rmul__(self, other):
        return SQLTriplet(other, operator.mul, self)

    def __sub__(self, other):
        return SQLTriplet(other, operator.sub, self)

    def __rsub__(self, other):
        return SQLTriplet(self, operator.sub, other)

    def __truediv__(self, other):
        return SQLTriplet(self, operator.truediv, other)

    def __rtruediv__(self, other):
        return SQLTriplet(other, operator.truediv, self)

    def __floordiv__(self, other):
        return SQLTriplet(self, operator.floordiv, other)

    def __rfloordiv__(self, other):
        return SQLTriplet(other, operator.floordiv, self)

    def __mod__(self, other):
        return SQLTriplet(self, operator.mod, other)

    def __rmod__(self, other):
        return SQLTriplet(other, operator.mod, self)

    def __gt__(self, other):
        return SQLTriplet(self, operator.gt, other)

    def __ge__(self, other):
        return SQLTriplet(self, operator.ge, other)

    def __lt__(self, other):
        return SQLTriplet(self, operator.lt, other)

    def __le__(self, other):
        return SQLTriplet(self, operator.le, other)

    def __abs__(self):
        return SQLTriplet(self, operator.abs)

    def __eq__(self, other):
        return SQLTriplet(self, operator.eq, other)

    def __ne__(self, other: object):
        return SQLTriplet(self, operator.ne, other)

    def __neg__(self):
        return SQLTriplet(self, operator.neg)

    def __contains__(self, other):
        return SQLTriplet(self, "contains", other)

    def __pow__(self, x):
        return SQLTriplet(self, operator.pow, x)

    def __rpow__(self, x):
        return SQLTriplet(x, operator.pow, self)

    def __and__(self, other):
        return SQLTriplet(self, operator.and_, other)

    def __rand__(self, other):
        return SQLTriplet(other, operator.and_, self)

    def __or__(self, other):
        return SQLTriplet(self, operator.or_, other)

    def __ror__(self, other):
        return SQLTriplet(other, operator.or_, self)


class SQLTriplet(SQLExpression):
    """Representation of an SQL triplet expression (LHS, operator, RHS).

    Attributes:
        lhs (SQLExpression): Left hand side of expression.
        op (str): Operator string.
        rhs (SQLExpression): Right hand side of expression.

    """

    def __init__(
        self,
        lhs: Union["SQLTriplet", str],
        op: Union[Callable, str] = None,
        rhs: Union["SQLTriplet", str] = None,
    ):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs
        self.formatters = {
            operator.mul: lambda a, b: f"({a} * {b})",
            operator.gt: lambda a, b: f"({a} > {b})",
            operator.ge: lambda a, b: f"({a} >= {b})",
            operator.lt: lambda a, b: f"({a} < {b})",
            operator.le: lambda a, b: f"({a} <= {b})",
            operator.add: lambda a, b: f"({a} + {b})",
            operator.sub: lambda a, b: f"({a} - {b})",
            operator.neg: lambda a, _: f"(-{a})",
            operator.truediv: lambda a, b: f"({a} / {b})",
            operator.floordiv: lambda a, b: f"FLOOR({a} / {b})",
            operator.and_: lambda a, b: f"({a} AND {b})",
            operator.or_: lambda a, b: f"({a} OR {b})",
            operator.abs: lambda a, _: f"ABS({a})",
            operator.not_: lambda a, _: f"NOT({a})",
            operator.eq: lambda a, b: f"({a} == {b})",
            operator.ne: lambda a, b: f"({a} != {b})",
            operator.pow: lambda a, p: f"POWER({a}, {p})",
            operator.mod: lambda a, b: f"({a} % {b})",
            "is_none": lambda a, _: f"({a} IS NULL)",
            "is_not_none": lambda a, _: f"({a} IS NOT NULL)",
            "list_sum": lambda a, _: f"LISTSUM({a})",
            "if_null": lambda x, d: f"IFNULL({x}, {d})",
            "contains": lambda j, o: f"CONTAINS({j}, {o})",
            "bool": lambda x, _: f"({x} != 0)",
        }

    def __str__(self) -> str:
        lhs = self.lhs
        rhs = self.rhs
        if isinstance(rhs, str):
            # is this ok? fixes categorical where predicate
            rhs = f'"{rhs}"'
        if lhs and self.op:
            return self.formatters[self.op](lhs, rhs)
        raise ValueError("Invalid SQLTriplet.")


class SQLJSONDictionary(SQLExpression):
    """Representation of an SQL expression to access JSON properties."""

    def __init__(self, acc: str = None) -> None:
        self.acc = acc or ""

    def __str__(self) -> str:
        return f"json_extract(properties, {json.dumps(f'$.{self.acc}')})"

    def __getitem__(self, key: str) -> "SQLJSONDictionary":
        if isinstance(key, (int,)):
            key_str = f"[{key}]"
        else:
            key_str = str(key)
        joiner = "." if self.acc and not isinstance(key, int) else ""
        return SQLJSONDictionary(acc=self.acc + joiner + f"{key_str}")

    def get(self, key, default=None):
        """Returns SQLTriplet specified by key."""
        return SQLTriplet(self[key], "if_null", default or SQLNone())


class SQLRegex(SQLExpression):
    """Representation of an SQL expression to match a string against a regex."""

    def __init__(self, pattern: str, string: str, flags: int = 0) -> None:
        self.pattern = pattern
        self.string = string
        self.flags = flags

    def __str__(self) -> str:
        string = self.string
        pattern = self.pattern
        flags = self.flags
        if isinstance(string, (str, Number)):
            string = json.dumps(string)
        if isinstance(pattern, (str, Number)):
            pattern = json.dumps(pattern)
        if flags != 0:
            return f"REGEXP({pattern}, {string}, {flags})"
        return f"({string} REGEXP {pattern})"

    @classmethod
    def search(cls, pattern: str, string: str, flags: int = 0) -> "SQLRegex":
        """Returns an SQL expression to match a string against a pattern."""
        return SQLRegex(pattern, string, int(flags))


def py_is_none(x: Any) -> bool:
    """Check if x is None."""
    return x is None


def py_is_not_none(x: Any) -> bool:
    """Check if x is not None."""
    return x is not None


def py_regexp(pattern: str, string: str, flags: int = 0) -> Optional[str]:
    """Check if string matches pattern."""
    reg = re.compile(pattern, flags=flags)
    match = reg.search(string)
    if match:
        return match[0]
    return None


def json_list_sum(json_list: str) -> Number:
    """Return the sum of a list of numbers in a JSON string.

    Args:
        json_list: JSON string containing a list of numbers.

    Returns:
        Number:
            The sum of the numbers in the list.

    """
    return sum(json.loads(json_list))


def json_contains(json_str: str, x: object) -> bool:
    """Return True if a JSON string contains x.

    Args:
        json_str: JSON string.
        x: Value to search for.

    Returns:
        bool:
            True if x is in json_str.

    """
    return x in json.loads(json_str)


def sql_is_none(x: Union[SQLExpression, Number, str, bool]) -> SQLTriplet:
    """Check if x is None.

    Returns:
        SQLTriplet:
            SQLTriplet representing None check.

    """
    return SQLTriplet(x, "is_none")


def sql_is_not_none(x: Union[SQLExpression, Number, str, bool]) -> SQLTriplet:
    """Check if x is not None.

    Returns:
        SQLTriplet:
            SQLTriplet representing not None check.

    """
    return SQLTriplet(x, "is_not_none")


def sql_list_sum(x: SQLJSONDictionary) -> SQLTriplet:
    """Return a representation of the sum of a list.

    Args:
        x (SQLJSONDictionary):
            The list to sum.

    Returns:
        SQLTriplet:
            SQLTriplet for a function call to sum the list.

    """
    return SQLTriplet(x, "list_sum")


def sql_has_key(dictionary: SQLJSONDictionary, key: Union[str, int]) -> SQLTriplet:
    """Check if a dictionary has a key.

    Args:
        dictionary (SQLProperties):
            SQLProperties object representing a JSON dict.
        key(str or int):
            Key to check for.

    Returns:
        SQLTriplet:
            SQLTriplet representing key check.

    """
    if not isinstance(dictionary, (SQLJSONDictionary,)):
        raise TypeError("Unsupported type for has_key.")
    return SQLTriplet(dictionary[key], "is_not_none")


# Constants defining the global variables for use in eval() when
# evaluating expressions.

_COMMON_GLOBALS = {
    "__builtins__": {"abs": abs},
    "re": re.RegexFlag,
}
SQL_GLOBALS = {
    "__builtins__": {**_COMMON_GLOBALS["__builtins__"], "sum": sql_list_sum},
    "props": SQLJSONDictionary(),
    "is_none": sql_is_none,
    "is_not_none": sql_is_not_none,
    "regexp": SQLRegex.search,
    "has_key": sql_has_key,
    "re": _COMMON_GLOBALS["re"],
}
PY_GLOBALS = {
    "__builtins__": {**_COMMON_GLOBALS["__builtins__"], "sum": sum},
    "is_none": py_is_none,
    "is_not_none": py_is_not_none,
    "regexp": py_regexp,
    "has_key": lambda a, b: b in a,
    "re": _COMMON_GLOBALS["re"],
}
