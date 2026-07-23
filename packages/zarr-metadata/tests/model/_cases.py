from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import pytest

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


@dataclass(frozen=True)
class Expect(Generic[TIn, TOut]):
    """A test case with explicit input, expected output, and a human-readable id."""

    input: TIn
    output: TOut
    id: str


@dataclass(frozen=True)
class ExpectFail(Generic[TIn]):
    """A test case that should raise an exception.

    `msg` is a regex matched against the exception text (pytest's native
    `match=` semantics). Leave it `None` to assert only the exception type. Set
    `escape=True` when `msg` is a literal that contains regex metacharacters
    such as `(`, `[`, or `.`; `escape` has no effect when `msg` is `None`.
    """

    input: TIn
    exception: type[Exception]
    id: str
    msg: str | None = None
    escape: bool = False

    def raises(self) -> AbstractContextManager[pytest.ExceptionInfo[Exception]]:
        if self.msg is None:
            return pytest.raises(self.exception)
        pattern = re.escape(self.msg) if self.escape else self.msg
        return pytest.raises(self.exception, match=pattern)


def mutate_nested_containers(value: object) -> None:
    """Recursively mutate every mutable container reachable inside `value`.

    Adds a marker key to every dict and appends a marker to every list,
    descending through tuples. Used to prove a `to_json` document shares no
    mutable state with the model that produced it.
    """
    if isinstance(value, dict):
        for item in value.values():
            mutate_nested_containers(item)
        value["__mutated__"] = "__mutated__"
    elif isinstance(value, list):
        for item in value:
            mutate_nested_containers(item)
        value.append("__mutated__")
    elif isinstance(value, tuple):
        for item in value:
            mutate_nested_containers(item)
