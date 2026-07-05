"""The absence sentinel for optional metadata-document keys.

The models observe one invariant: `None` in a model always corresponds to a
JSON `null` in the document (a v2 `compressor`/`filters` value, an unnamed
dimension inside `dimension_names`), and `UNSET` always means the document
key is absent. The two are never interchangeable, so a model value can never
leak into a document as a spelling the writer did not intend.

Check with identity: `if model.dimension_names is UNSET: ...`.

Checker support (PEP 661 is Final; stdlib `sentinel` arrives in Python
3.15): ty types this spelling exactly, including `is`/`is not` narrowing.
Pyright supports it but a regression (1.1.405+, tracked as
https://github.com/microsoft/pyright/issues/11115) degrades class-attribute
reads to `Unknown`, so this package pins pyright to the last good version
until the fix lands. Mypy support is in review
(https://github.com/python/mypy/pull/21647); until it merges, mypy-checked
consumers of these fields need a `cast` or `type: ignore` at narrowing
sites. This is a deliberate short-term cost: the sentinel is the standard,
and the checkers are converging on it.
"""

from __future__ import annotations

from typing_extensions import Sentinel

UNSET = Sentinel("UNSET")
"""Marks a metadata-document key as absent (PEP 661 sentinel; usable directly
in type expressions, e.g. `tuple[str, ...] | UNSET`). Test with `is UNSET`."""
