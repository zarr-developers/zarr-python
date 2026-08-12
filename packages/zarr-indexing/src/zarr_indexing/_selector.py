"""Internal scalar-selector coercion shared by indexing dialects."""

from __future__ import annotations

import operator
from typing import Any, SupportsIndex, cast

import numpy as np


def is_bool_scalar(value: Any) -> bool:
    """Return whether ``value`` is a Python or NumPy boolean scalar."""
    return isinstance(value, (bool, np.bool_))


def as_scalar_index(value: Any) -> int | None:
    """Return a non-boolean scalar selector as an exact Python integer.

    Selector coercion follows Python's ``__index__`` protocol, rather than
    accepting only the concrete integer classes we happen to know about. In
    particular, ``operator.index`` rejects lossy ``__int__``-only objects and
    validates that ``__index__`` really returned an integer.
    """
    if is_bool_scalar(value):
        return None
    # ndarray defines ``__index__`` for its scalar-integer case, but the
    # attribute also makes non-scalar and non-integer arrays look like
    # ``SupportsIndex`` at runtime. Those are array selectors, not malformed
    # scalar selectors, and must continue through array dtype validation.
    if isinstance(value, np.ndarray):
        array = cast("np.ndarray[Any, np.dtype[Any]]", value)
        if array.ndim != 0 or array.dtype.kind not in "iu":
            return None
    if not isinstance(value, SupportsIndex):
        return None
    return operator.index(cast(SupportsIndex, value))


def require_index(value: Any) -> int:
    """Coerce a required slice component through ``__index__``."""
    return operator.index(value)
