"""Fixtures for the model tests."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def interpreter_writes_4300_digits() -> Iterator[None]:
    """The interpreter's default limit on writing an integer, pinned: an environment may lift it (`PYTHONINTMAXSTRDIGITS=0`), and a test of what happens past it needs it."""
    limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(limit)
