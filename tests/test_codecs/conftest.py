from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Expect[TIn, TOut]:
    """Model an input and an expected output value for a test case."""

    input: TIn
    expected: TOut


@dataclass(frozen=True)
class ExpectErr[TIn]:
    """Model an input and an expected error message for a test case."""

    input: TIn
    msg: str
    exception_cls: type[Exception]
