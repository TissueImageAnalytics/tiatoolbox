# @title
"""Where the magic happens!"""
import json
import operator
import re
from dataclasses import dataclass
from numbers import Number
from typing import Callable, Optional, Union


@dataclass
class SQLNone:
    def __str__(self) -> str:
        return "NULL"

    def __repr__(self) -> str:
        return str(self)  # pragma: no cover


class SQLExpression:
    def __init__(
        self,
        lhs: Union["SQLExpression", str],
        op: Union[Callable, str] = None,
        rhs: Union["SQLExpression", str] = None,
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
            operator.eq: lambda a, b: f"({a} = {b})",
            operator.ne: lambda a, b: f"({a} != {b})",
            operator.pow: lambda a, p: f"POWER({a}, {p})",
            operator.mod: lambda a, b: f"({a} % {b})",
            "is none": lambda a, _: f"({a} IS NULL)",
            "is not none": lambda a, _: f"({a} IS NOT NULL)",
            "listsum": lambda a, _: f"LISTSUM({a})",
            "ifnull": lambda x, d: f"IFNULL({x}, {d})",
            "contains": lambda j, o: f"CONTAINS({j}, {o})",
            "bool": lambda x, _: f"({x} != 0)",
        }

    def __str__(self) -> str:
        lhs = self.lhs
        rhs = self.rhs
        if lhs and self.op:
            return self.formatters[self.op](lhs, rhs)
        return f" {lhs} "

    def __repr__(self):
        return str(self)  # pragma: no cover

    def __add__(self, other):
        return SQLExpression(self, operator.add, other)

    def __radd__(self, other):
        return SQLExpression(other, operator.add, self)

    def __mul__(self, other):
        return SQLExpression(self, operator.mul, other)

    def __rmul__(self, other):
        return SQLExpression(other, operator.mul, self)

    def __sub__(self, other):
        return SQLExpression(other, operator.sub, self)

    def __rsub__(self, other):
        return SQLExpression(self, operator.sub, other)

    def __truediv__(self, other):
        return SQLExpression(self, operator.truediv, other)

    def __rtruediv__(self, other):
        return SQLExpression(other, operator.truediv, self)

    def __floordiv__(self, other):
        return SQLExpression(self, operator.floordiv, other)

    def __rfloordiv__(self, other):
        return SQLExpression(other, operator.floordiv, self)

    def __mod__(self, other):
        return SQLExpression(self, operator.mod, other)

    def __rmod__(self, other):
        return SQLExpression(other, operator.mod, self)

    def __gt__(self, other):
        return SQLExpression(self, operator.gt, other)

    def __ge__(self, other):
        return SQLExpression(self, operator.ge, other)

    def __lt__(self, other):
        return SQLExpression(self, operator.lt, other)

    def __le__(self, other):
        return SQLExpression(self, operator.le, other)

    def __abs__(self):
        return SQLExpression(self, operator.abs)

    def __not__(self):
        return SQLExpression(self, operator.not_)

    def __eq__(self, other):
        return SQLExpression(self, operator.eq, other)

    def __ne__(self, other: object):
        return SQLExpression(self, operator.ne, other)

    def __neg__(self):
        return SQLExpression(self, operator.neg)

    def __contains__(self, other):
        return SQLExpression(self, "contains", other)

    def __pow__(self, x):
        return SQLExpression(self, operator.pow, x)

    def __rpow__(self, x):
        return SQLExpression(x, operator.pow, self)

    def __and__(self, other):
        return SQLExpression(self, operator.and_, other)

    def __rand__(self, other):
        return SQLExpression(other, operator.and_, self)

    def __or__(self, other):
        return SQLExpression(self, operator.or_, other)

    def __ror__(self, other):
        return SQLExpression(other, operator.or_, self)


class Properties(SQLExpression):
    def __init__(self, acc: str = None) -> None:
        self.acc = acc or ""

    def __str__(self) -> str:
        return f"json_extract(properties, {json.dumps(f'$.{self.acc}')})"

    def __getitem__(self, key: str) -> "Properties":
        if isinstance(key, (int,)):
            key_str = f"[{key}]"
        else:
            key_str = str(key)
        joiner = "." if self.acc and not isinstance(key, (int)) else ""
        return Properties(acc=self.acc + joiner + f"{key_str}")

    def get(self, key, default=None):
        return SQLExpression(self[key], "ifnull", default or SQLNone())


class SQLRegex(SQLExpression):
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
    def search(cls, pattern: str, string: str, flags: int = 0) -> SQLExpression:
        return SQLExpression(SQLRegex(pattern, string, flags))


def is_none(x) -> bool:
    return x is None


def is_not_none(x) -> bool:
    return x is not None


def regexp(pattern: str, string: str, flags: int = 0) -> Optional[str]:
    reg = re.compile(pattern, flags=flags)
    match = reg.search(string)
    if match:
        return match[0]
    return None


def json_list_sum(json_list: str) -> Number:
    return sum(json.loads(json_list))


def json_contains(json_str: str, x: object) -> bool:
    return x in json.loads(json_str)


def sql_is_none(x) -> SQLExpression:
    return SQLExpression(x, "is none")


def sql_is_not_none(x) -> SQLExpression:
    return SQLExpression(x, "is not none")


def sql_list_sum(x) -> SQLExpression:
    return SQLExpression(x, "listsum")


def sql_has_key(a, b) -> SQLExpression:
    if not isinstance(a, (Properties,)):
        raise TypeError("Unsupported type for has_key.")
    return SQLExpression(a[b], "is not none")


_COMMON_GLOBALS = {
    "__builtins__": {"abs": abs},
    "re": re.RegexFlag,
}
SQL_GLOBALS = {
    "__builtins__": {**_COMMON_GLOBALS["__builtins__"], "sum": sql_list_sum},
    "props": Properties(),
    "is_none": sql_is_none,
    "is_not_none": sql_is_not_none,
    "regexp": SQLRegex.search,
    "has_key": sql_has_key,
    "re": _COMMON_GLOBALS["re"],
}
SQL_LOCALS = {}
PY_GLOBALS = {
    "__builtins__": {**_COMMON_GLOBALS["__builtins__"], "sum": sum},
    "is_none": is_none,
    "is_not_none": is_not_none,
    "regexp": regexp,
    "has_key": lambda a, b: b in a,
    "re": _COMMON_GLOBALS["re"],
}
PY_LOCALS = {
    "props": [0, 1],
}

if __name__ == "__main__":
    # Simply access a property value
    query1 = 'props["value"] == 5'
    print(query1, "\n>>>", eval(query1, SQL_GLOBALS, SQL_LOCALS))

    # Nesting
    query2 = 'props["nums"][0] == 0'
    print(query2, "\n>>>", eval(query2, SQL_GLOBALS, SQL_LOCALS))

    # Complex query
    query3 = '(props["nums"][0] == 0) & props["bool"]'
    print(query3, "\n>>>", eval(query3, SQL_GLOBALS, SQL_LOCALS))

    # More complex query
    query4 = (
        '((props["nums"][0] == 0) & props["value"] > 4) | is_none(props["nothing"])'
    )
    print(query4, "\n>>>", eval(query4, SQL_GLOBALS, SQL_LOCALS))

    # Even regex works
    query6 = 'regexp("H.*", props["abc"]["msg"])'
    print(query6, "\n>>>", eval(query6, SQL_GLOBALS, SQL_LOCALS))

    query7 = "is_none(123) & is_not_none(345)"
    print(query7, "\n>>>", eval(query7, SQL_GLOBALS, SQL_LOCALS))

    query8 = "has_key(props, 'abc')"
    print(query8, "\n>>>", eval(query8, SQL_GLOBALS, SQL_LOCALS))

    query9 = "props.get('abc')"
    print(query9, "\n>>>", eval(query9, SQL_GLOBALS, SQL_LOCALS))
