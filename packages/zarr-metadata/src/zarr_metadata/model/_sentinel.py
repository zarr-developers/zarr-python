"""The absence sentinel for optional metadata-document keys.

The models observe one invariant: `None` in a model always corresponds to a
JSON `null` in the document (a v2 `compressor`/`filters` value, an unnamed
dimension inside `dimension_names`), and `UNSET` always means the document
key is absent. The two are never interchangeable, so a model value can never leak
into a document as a spelling the writer did not intend.

Check with identity: `if model.dimension_names is UNSET: ...`.

Implementation note: `typing_extensions.Sentinel` (PEP 661, Final as of
2026-04-23, stdlib in Python 3.15) is the intended spelling, but two
independent checker gaps block it for now. Pyright: a confirmed regression
(1.1.405 through at least 1.1.411; worked in <= 1.1.404,
https://github.com/microsoft/pyright/issues/11115) degrades a Sentinel to
`Unknown` when read from any class-body attribute annotation. Mypy (2.1.0):
has not yet implemented PEP 661 — a sentinel in type position is a hard
`[valid-type]` error, so downstream mypy users (zarr-python itself) would
see these fields as `Any`. Pinning a working pyright in this package's CI
would fix neither contributors' IDEs nor downstream checkers reading the
py.typed annotations. The single-member enum gives the same identity
semantics with exact `Literal` narrowing on every checker; switch to
`Sentinel` once the pyright regression is fixed and mypy support lands —
both expected, now that the PEP is Final.
"""

from __future__ import annotations

from enum import Enum
from typing import Final, Literal


class UnsetType(Enum):
    """The type of `UNSET`; use in annotations as `T | UnsetType`."""

    UNSET = "UNSET"

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final[Literal[UnsetType.UNSET]] = UnsetType.UNSET
"""Marks a metadata-document key as absent. Test with `is UNSET`."""
