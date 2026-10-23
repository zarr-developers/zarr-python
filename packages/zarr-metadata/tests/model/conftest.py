"""Fixtures for the model tests."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def interpreter_writes_4300_digits() -> Iterator[None]:
    """Pin the interpreter's default limit on converting an integer to text. An environment may lift the limit (`PYTHONINTMAXSTRDIGITS=0`), and tests of integers past the limit need it in place."""
    limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(limit)
