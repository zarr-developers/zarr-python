"""Helpers for validating JSON-decoded metadata.

Most JSON metadata validation is delegated to :func:`msgspec.convert`, which
handles the type coercions Zarr needs (``Literal`` membership, ``int``/``bool``
strictness, list-to-tuple, ``TypedDict`` with ``NotRequired``). :func:`convert`
is a thin wrapper that translates :class:`msgspec.ValidationError` into the
``TypeError`` the rest of the codebase already raises.

msgspec cannot handle two things in Zarr's metadata types:

* the recursive ``JSON`` / ``JSONValue`` aliases, which it rejects at
  schema-build time, and
* PEP 728 ``extra_items=`` extension fields, which it silently drops.

:func:`validate_json_value` is the small hand-written fallback for the first of
those. See https://github.com/zarr-developers/zarr-python/issues/3285.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, cast

import msgspec

if TYPE_CHECKING:
    from zarr.core.common import JSON

__all__ = ["MAX_JSON_DEPTH", "convert", "validate_json_value"]

MAX_JSON_DEPTH: Final = 64
"""Maximum nesting depth accepted by :func:`validate_json_value`."""


def convert(value: object, type_: Any, *, strict: bool = True) -> Any:
    """Validate and coerce ``value`` against ``type_`` via :func:`msgspec.convert`.

    msgspec handles the JSON-shaped coercions Zarr needs but raises
    :class:`msgspec.ValidationError` on a mismatch. We translate that to
    ``TypeError`` so callers can keep raising the exception types the rest of
    Zarr already expects.
    """
    try:
        return msgspec.convert(value, type_, strict=strict)
    except msgspec.ValidationError as exc:
        raise TypeError(str(exc)) from exc


def validate_json_value(value: object, *, max_depth: int = MAX_JSON_DEPTH, _depth: int = 0) -> JSON:
    """Check that ``value`` is a JSON value and return it unchanged.

    msgspec cannot build a schema for Zarr's recursive ``JSON`` / ``JSONValue``
    aliases, so this covers the fields typed that way (``attributes``,
    ``fill_value``, extension-field values). Unlike the previous per-field
    parsers it also enforces ``max_depth``: a pathologically nested document
    could otherwise exhaust the interpreter stack.
    """
    if _depth > max_depth:
        raise ValueError(f"JSON value nesting exceeds the maximum depth of {max_depth}.")
    if value is None or isinstance(value, (bool, int, float, str)):
        return cast("JSON", value)
    if isinstance(value, (list, tuple)):
        for item in value:
            validate_json_value(item, max_depth=max_depth, _depth=_depth + 1)
        return cast("JSON", value)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be str, got {type(key).__name__}.")
            validate_json_value(item, max_depth=max_depth, _depth=_depth + 1)
        return cast("JSON", value)
    raise TypeError(f"Value {value!r} is not a valid JSON value.")
